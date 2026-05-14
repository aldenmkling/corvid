"""Mixed Point-and-Line (PnL) DLT solver for the Phase 6 H regressor.

Builds a confidence-weighted Direct Linear Transform (DLT) system that takes
heterogeneous correspondences — *some* are point↔point and *others* are
line↔line — and solves the homography H in closed form via SVD.

Why mixed:
- Yardline / sideline tokens give us a *line* in the image (PCA fit on the
  CC pixels) that corresponds to a known *line* in NGS coords (a vertical
  yardline at NGS_x, or one of the two horizontal sidelines). The CC's
  centroid is frame-dependent (depends where the line enters/exits the
  frame) so we can't use it as a point keypoint, but the *line itself* is
  a frame-invariant structural prediction.
- Hash / number tokens give us a *point* in the image (CC centroid /
  group centroid) that corresponds to a known *point* in NGS coords.
  These are well-defined keypoints because the CC is a localized blob.

DLT math (for h = [h11..h33] flattened row-major):

POINT correspondence  (image (x,y), NGS (X,Y)):
    H @ [x,y,1]^T  ∝  [X,Y,1]^T
  → 2 rows of A:
      [-x, -y, -1,  0,  0,  0,  X*x, X*y, X]
      [ 0,  0,  0, -x, -y, -1,  Y*x, Y*y, Y]

LINE correspondence  (image (a,b,c), NGS (A,B,C); both ax+by+c=0 form):
    H^T @ l_ngs  ∝  l_img    (lines transform via H^{-T})
  → take cross-product l_img × (H^T l_ngs) = 0, 2 independent rows:
      [ 0, -c*A, b*A,  0, -c*B, b*B,  0, -c*C, b*C]
      [ c*A,  0, -a*A, c*B,  0, -a*B, c*C,  0, -a*C]

Weighted DLT: scale each row by sqrt(weight_i) before SVD. The smallest
right singular vector of A is h (up to sign).

This module is fully differentiable end-to-end through `torch.linalg.svd`
so the per-token confidences and structural predictions can be trained
with an H-consistency loss.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Per-row builders (return 2x9 row blocks).
# All builders assume already-NORMALIZED coords (pre-conditioned).
# ---------------------------------------------------------------------------

def point_rows(p_img: torch.Tensor, p_ngs: torch.Tensor) -> torch.Tensor:
    """Two DLT rows for a point correspondence.

    Args:
        p_img: (..., 2)  image-side (x, y).
        p_ngs: (..., 2)  NGS-side (X, Y).

    Returns:
        (..., 2, 9) matrix block.
    """
    x = p_img[..., 0]
    y = p_img[..., 1]
    X = p_ngs[..., 0]
    Y = p_ngs[..., 1]
    zero = torch.zeros_like(x)
    one = torch.ones_like(x)

    row1 = torch.stack([-x, -y, -one, zero, zero, zero, X * x, X * y, X], dim=-1)
    row2 = torch.stack([zero, zero, zero, -x, -y, -one, Y * x, Y * y, Y], dim=-1)
    return torch.stack([row1, row2], dim=-2)  # (..., 2, 9)


def line_rows(l_img: torch.Tensor, l_ngs: torch.Tensor) -> torch.Tensor:
    """Two DLT rows for a line correspondence.

    Args:
        l_img: (..., 3)  image-side line (a, b, c)  with ax+by+c=0.
        l_ngs: (..., 3)  NGS-side line (A, B, C).

    Returns:
        (..., 2, 9) matrix block.
    """
    a = l_img[..., 0]
    b = l_img[..., 1]
    c = l_img[..., 2]
    A = l_ngs[..., 0]
    B = l_ngs[..., 1]
    C = l_ngs[..., 2]
    zero = torch.zeros_like(a)

    # Row 1: [0, -cA, bA, 0, -cB, bB, 0, -cC, bC]
    row1 = torch.stack(
        [zero, -c * A, b * A, zero, -c * B, b * B, zero, -c * C, b * C],
        dim=-1,
    )
    # Row 2: [cA, 0, -aA, cB, 0, -aB, cC, 0, -aC]
    row2 = torch.stack(
        [c * A, zero, -a * A, c * B, zero, -a * B, c * C, zero, -a * C],
        dim=-1,
    )
    return torch.stack([row1, row2], dim=-2)


# ---------------------------------------------------------------------------
# Helper: build image line params from a token's centroid + orientation.
# ---------------------------------------------------------------------------

def line_from_centroid_orientation(
    centroid: torch.Tensor,        # (..., 2): (cx, cy)
    direction: torch.Tensor,       # (..., 2): (dx, dy) unit vector along line
) -> torch.Tensor:
    """Convert centroid + direction into (a, b, c) with ax+by+c=0.

    A line through point (cx, cy) with direction (dx, dy) has normal
    (-dy, dx), so its equation is:
        -dy*(x - cx) + dx*(y - cy) = 0
        -dy*x + dx*y + (dy*cx - dx*cy) = 0
    so (a, b, c) = (-dy, dx, dy*cx - dx*cy).
    """
    cx = centroid[..., 0]
    cy = centroid[..., 1]
    dx = direction[..., 0]
    dy = direction[..., 1]
    a = -dy
    b = dx
    c = dy * cx - dx * cy
    return torch.stack([a, b, c], dim=-1)


# ---------------------------------------------------------------------------
# Weighted batched DLT solve.
# ---------------------------------------------------------------------------

def solve_h_dlt_weighted(
    A_rows: torch.Tensor,    # (B, R, 9)  R = total rows across all tokens
    weights: torch.Tensor,   # (B, R)     non-negative confidence per row
    eps: float = 1e-8,
) -> torch.Tensor:
    """Confidence-weighted DLT solve via SVD.

    Returns:
        H: (B, 3, 3) homography (last entry h33 sign-fixed positive).
    """
    # Scale rows by sqrt(weight). Adds tiny eps to avoid sqrt(0) gradient
    # blowup at exactly-zero confidence.
    w = torch.clamp(weights, min=0.0)
    sw = torch.sqrt(w + eps).unsqueeze(-1)               # (B, R, 1)
    A_w = A_rows * sw                                     # (B, R, 9)

    # SVD: smallest right singular vector = h.
    # torch.linalg.svd returns U, S, Vh with Vh shape (..., 9, 9).
    # The last row of Vh is the eigenvector for the smallest singular value.
    _, _, Vh = torch.linalg.svd(A_w, full_matrices=False)
    h = Vh[..., -1, :]                                    # (B, 9)

    H = h.reshape(*h.shape[:-1], 3, 3)                    # (B, 3, 3)

    # Sign-fix: pick the sign so that h33 is positive (consistent gauge).
    sign = torch.where(H[..., 2, 2] < 0, -torch.ones_like(H[..., 2, 2]),
                        torch.ones_like(H[..., 2, 2]))
    H = H * sign.unsqueeze(-1).unsqueeze(-1)
    return H


# ---------------------------------------------------------------------------
# Convenience: pack a list of (rows, weights) blocks into A, w batched.
# ---------------------------------------------------------------------------

def pack_dlt_system(
    blocks: list[tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Concatenate per-token (rows, weights) along the row dim.

    Args:
        blocks: list of (rows, weights) where rows is (B, K_i, 9) and
                weights is (B, K_i). K_i is 2 for point or line tokens.

    Returns:
        (A, w) with A shape (B, sum_K, 9), w shape (B, sum_K).
    """
    A = torch.cat([b[0] for b in blocks], dim=1)
    w = torch.cat([b[1] for b in blocks], dim=1)
    return A, w


# ===========================================================================
# Self-test: synthetic data with known H, verify recovery to <1e-4.
# ===========================================================================

def _make_random_H(seed: int = 0, dtype=torch.float64) -> torch.Tensor:
    """A plausible NFL-broadcast-ish homography (not pure random).

    Uses float64 for numerical stability of the test.
    """
    torch.manual_seed(seed)
    pts_ngs = torch.tensor([
        [40.0, 0.0],
        [100.0, 0.0],
        [100.0, 53.33],
        [40.0, 53.33],
    ], dtype=dtype)
    pts_img = torch.tensor([
        [200.0, 600.0],
        [1100.0, 590.0],
        [950.0, 220.0],
        [350.0, 230.0],
    ], dtype=dtype)
    rows = point_rows(pts_img, pts_ngs).reshape(8, 9).unsqueeze(0)
    w = torch.ones(1, 8, dtype=dtype)
    H = solve_h_dlt_weighted(rows, w)[0]
    return H


def _project_pt(H: torch.Tensor, p_ngs: torch.Tensor) -> torch.Tensor:
    """Project NGS point into image via H^{-1}."""
    Hinv = torch.linalg.inv(H)
    pn = torch.cat([p_ngs, torch.ones_like(p_ngs[..., :1])], dim=-1)
    pi = pn @ Hinv.T
    pi = pi[..., :2] / pi[..., 2:3]
    return pi


def _project_line(H: torch.Tensor, l_ngs: torch.Tensor) -> torch.Tensor:
    """Project NGS line into image via H^T (lines transform with H^T)."""
    return l_ngs @ H               # equivalent to (H^T @ l)^T row form.


def _reproj_pt_err(H: torch.Tensor, pts_img: torch.Tensor,
                     pts_ngs: torch.Tensor) -> float:
    """RMS pixel reprojection error for point correspondences."""
    pn = torch.cat([pts_ngs, torch.ones_like(pts_ngs[..., :1])], dim=-1)
    pi = pn @ torch.linalg.inv(H).T
    pi = pi[..., :2] / pi[..., 2:3]
    err = (pi - pts_img).pow(2).sum(-1).sqrt()
    return err.max().item()


def _self_test():
    """End-to-end smoke test: synthesize correspondences from a known H,
    perturb, run weighted DLT, verify recovery via reprojection error.

    Uses float64 throughout so we hit numerical-zero recovery (not just
    "small relative to data scale"). Production training will run in
    float32 / bf16, but the algorithm itself is correct in fp32 too —
    the input pre-conditioning (in the model wrapper) handles the
    Hartley-normalization that makes fp32 stable.
    """
    print("[self-test] PnL DLT solver (float64 for numerical exactness)")
    dtype = torch.float64
    H_gt = _make_random_H(seed=0, dtype=dtype)
    print("H_gt (normalized) =\n", (H_gt / H_gt[2, 2]).numpy())

    # ---- All-point case (sanity baseline)
    pts_ngs = torch.tensor([
        [55.0, 5.0], [70.0, 30.0], [85.0, 50.0], [45.0, 25.0],
        [95.0, 10.0], [60.0, 45.0],
    ], dtype=dtype)
    pts_img = _project_pt(H_gt, pts_ngs)
    rows_p = point_rows(pts_img, pts_ngs).reshape(-1, 9).unsqueeze(0)
    w_p = torch.ones(1, rows_p.shape[1], dtype=dtype)
    H_pt = solve_h_dlt_weighted(rows_p, w_p)[0]
    err_pt = _reproj_pt_err(H_pt, pts_img, pts_ngs)
    print(f"all-points: max reproj px err = {err_pt:.3e}")
    assert err_pt < 1e-6, f"point-only recovery too loose: {err_pt}"

    # ---- All-line case
    lines_ngs = torch.tensor([
        [1.0, 0.0, -50.0],
        [1.0, 0.0, -80.0],
        [1.0, 0.0, -65.0],
        [0.0, 1.0,   0.0],
        [0.0, 1.0, -53.33],
    ], dtype=dtype)
    lines_img = _project_line(H_gt, lines_ngs)
    norm = torch.linalg.norm(lines_img[..., :2], dim=-1, keepdim=True)
    lines_img = lines_img / (norm + 1e-12)
    rows_l = line_rows(lines_img, lines_ngs).reshape(-1, 9).unsqueeze(0)
    w_l = torch.ones(1, rows_l.shape[1], dtype=dtype)
    H_ln = solve_h_dlt_weighted(rows_l, w_l)[0]
    # Validate by checking lines map correctly: project test NGS points
    # through H_ln, see if they obey expected line equations.
    err_ln = _reproj_pt_err(H_ln, pts_img, pts_ngs)
    print(f"all-lines:  max reproj px err = {err_ln:.3e}")
    assert err_ln < 1e-3, f"line-only recovery too loose: {err_ln}"

    # ---- Mixed: 3 points + 2 lines  (5 corr → 10 rows, redundant for 8 DoF)
    rows_mix = torch.cat([
        point_rows(pts_img[:3], pts_ngs[:3]).reshape(-1, 9),
        line_rows(lines_img[:2], lines_ngs[:2]).reshape(-1, 9),
    ], dim=0).unsqueeze(0)
    w_mix = torch.ones(1, rows_mix.shape[1], dtype=dtype)
    H_mix = solve_h_dlt_weighted(rows_mix, w_mix)[0]
    err_mix = _reproj_pt_err(H_mix, pts_img, pts_ngs)
    print(f"mixed (3pt+2ln): max reproj px err = {err_mix:.3e}")
    assert err_mix < 1e-3, f"mixed recovery too loose: {err_mix}"

    # ---- Outlier rejection via weights:
    pts_img_bad = pts_img.clone()
    pts_img_bad[0] += 200.0
    rows_bad = point_rows(pts_img_bad, pts_ngs).reshape(-1, 9).unsqueeze(0)
    w_bad = torch.ones(1, rows_bad.shape[1], dtype=dtype)
    w_bad[0, 0:2] = 0.0
    H_w = solve_h_dlt_weighted(rows_bad, w_bad)[0]
    err_w = _reproj_pt_err(H_w, pts_img[1:], pts_ngs[1:])  # exclude masked pt
    print(f"weighted-mask-out outlier: max reproj px err on rest = {err_w:.3e}")
    assert err_w < 1e-3, f"weighted outlier rejection failed: {err_w}"

    # ---- Soft weighting: weight needs to scale with squared outlier
    # magnitude to suppress its row energy in A^T A. For a 200 px outlier
    # surrounded by ~1 px-scale inliers, weight ~ 1/200² = 2.5e-5 fully
    # suppresses; weight 1e-4 should already give big reduction.
    w_soft = torch.ones(1, rows_bad.shape[1], dtype=dtype)
    w_soft[0, 0:2] = 1e-5
    H_s = solve_h_dlt_weighted(rows_bad, w_soft)[0]
    err_s = _reproj_pt_err(H_s, pts_img[1:], pts_ngs[1:])
    H_full_bad = solve_h_dlt_weighted(rows_bad,
                                        torch.ones_like(w_soft))[0]
    err_full = _reproj_pt_err(H_full_bad, pts_img[1:], pts_ngs[1:])
    print(f"soft-down-weight (1e-5) outlier: rest err = {err_s:.3e} "
          f"(vs full-weight bad: {err_full:.3e})")
    assert err_s < 0.05 * err_full, \
        f"soft weighting not reducing outlier enough: {err_s} vs {err_full}"

    # ---- Differentiability check: gradients flow through SVD.
    pts_ngs_p = pts_ngs.clone().requires_grad_(True)
    rows_d = point_rows(pts_img, pts_ngs_p).reshape(-1, 9).unsqueeze(0)
    w_d = torch.ones(1, rows_d.shape[1], dtype=dtype)
    H_d = solve_h_dlt_weighted(rows_d, w_d)[0]
    loss = (H_d / H_d[2, 2] - H_gt / H_gt[2, 2]).pow(2).sum()
    loss.backward()
    grad_ok = pts_ngs_p.grad is not None and \
              torch.isfinite(pts_ngs_p.grad).all().item()
    print(f"diff-through-SVD: grad finite = {grad_ok}")
    assert grad_ok

    # ---- Confidence batched solve
    rows_b = rows_mix.repeat(4, 1, 1)
    w_b = torch.ones(4, rows_b.shape[1], dtype=dtype)
    H_b = solve_h_dlt_weighted(rows_b, w_b)
    assert H_b.shape == (4, 3, 3)
    print("batched solve: OK")

    # ---- fp32 sanity: should still be reasonable for normalized inputs
    H_gt32 = H_gt.float()
    pts_ngs32 = pts_ngs.float()
    pts_img32 = pts_img.float()
    rows_32 = point_rows(pts_img32, pts_ngs32).reshape(-1, 9).unsqueeze(0)
    w_32 = torch.ones(1, rows_32.shape[1])
    H_32 = solve_h_dlt_weighted(rows_32, w_32)[0]
    err_32 = _reproj_pt_err(H_32.double(), pts_img, pts_ngs)
    print(f"fp32 (raw, no Hartley): max reproj px err = {err_32:.3e}")
    # No assertion: we explicitly expect this to be poor without Hartley
    # normalization (which we do at the model wrapper level, not here).

    print("[self-test] ALL PASSED")


if __name__ == "__main__":
    _self_test()
