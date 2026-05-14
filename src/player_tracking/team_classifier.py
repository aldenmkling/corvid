"""
Layer 4: post-tracking team classification.

Each PlayerTrajectory is sampled at N evenly-spaced measured frames; the
inner box of each detection is cropped (skip box_margin_frac on each side
to avoid background pixels), converted to HSV, and an HSV histogram is
computed. Histograms are averaged per track to form a signature, then
KMeans (k=2) splits all tracks into two clusters → 'team_A' / 'team_B'.

Tracks with < n_samples_per_track measured points are tagged 'unknown'
(insufficient evidence). Cluster labels are deterministic — the cluster
containing the smallest track_id becomes team_A.

Runs ONCE after tracking is complete; not part of the per-frame tracker.
"""

from __future__ import annotations

import os

import cv2
import numpy as np
from sklearn.cluster import KMeans

from .tracker import PlayerTrajectory


# HSV histogram bins: 8 hue × 8 sat × 4 val = 256 bins per crop.
_H_BINS = 8
_S_BINS = 8
_V_BINS = 4


def _crop_inner_box(frame: np.ndarray, xyxy: np.ndarray,
                    box_margin_frac: float) -> np.ndarray | None:
    """Return the inner box% of `frame` defined by xyxy (image space).

    Skips `box_margin_frac` on each side to avoid background bleed. Returns
    None if the resulting crop is empty / out of bounds.
    """
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])
    bw = x2 - x1
    bh = y2 - y1
    if bw <= 1 or bh <= 1:
        return None
    mx = bw * box_margin_frac
    my = bh * box_margin_frac
    ix1 = int(max(0, np.floor(x1 + mx)))
    iy1 = int(max(0, np.floor(y1 + my)))
    ix2 = int(min(w, np.ceil(x2 - mx)))
    iy2 = int(min(h, np.ceil(y2 - my)))
    if ix2 - ix1 < 2 or iy2 - iy1 < 2:
        return None
    return frame[iy1:iy2, ix1:ix2]


def _hsv_histogram(crop_bgr: np.ndarray) -> np.ndarray:
    """Compute a normalized HSV histogram over the crop. Returns a flat
    (H_BINS*S_BINS*V_BINS,) float32 vector summing to 1."""
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist(
        [hsv], [0, 1, 2], None,
        [_H_BINS, _S_BINS, _V_BINS],
        [0, 180, 0, 256, 0, 256],
    )
    hist = hist.flatten().astype(np.float32)
    s = hist.sum()
    if s > 0:
        hist /= s
    return hist


def classify_teams(
    trajectories: dict[int, PlayerTrajectory],
    video_path: str,
    n_samples_per_track: int = 8,
    box_margin_frac: float = 0.2,
) -> dict[int, str]:
    """Cluster trajectories into two teams via jersey-color histograms.

    Args:
        trajectories: track_id -> PlayerTrajectory (post-tracking output).
        video_path: path to the source MP4 (sideline clip).
        n_samples_per_track: number of evenly-spaced measured frames per
            trajectory to sample.
        box_margin_frac: fraction of each box side to skip when cropping.

    Returns:
        dict[track_id -> 'team_A' | 'team_B' | 'unknown'].
        - 'unknown' for trajectories with fewer than n_samples_per_track
          measured points (insufficient evidence) OR for which we couldn't
          recover any usable crops.
        - Team A is deterministically the cluster containing the smallest
          successfully-clustered track_id; the other becomes Team B.
    """
    if not trajectories:
        return {}
    if not os.path.exists(video_path):
        return {tid: "unknown" for tid in trajectories}

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {tid: "unknown" for tid in trajectories}

    # Build a per-track plan: which frame_idx and which xyxy box to sample.
    # Need both: frame_idx (to read the video) and xyxy (to crop).
    # PlayerTrajectory stores foot_point pixel_xy but not the box; we
    # reconstruct the sampling by walking the trajectory's MEASURED points
    # (interrupted=False) — for those points, pixel_xy is finite. The
    # actual box is not stored, so we rebuild it by looking up the box
    # from the detection cache via foot_point matching is brittle. Instead,
    # we'll fall back to a fixed-size pseudo-box around foot_point using
    # the median measured player height from the clip — but a much simpler
    # plan: scan the source video and crop a box around the foot point
    # using a heuristic height in pixels (player ~80 px tall). Imprecise,
    # but the histogram is dominated by the jersey colors regardless.

    # ── Better approach: re-derive boxes from the per-frame detection
    # cache by nearest-foot-point lookup. But to stay self-contained and
    # cheap, we approximate the box from the foot_point + a heuristic
    # height (read from the clip's typical player size — ~75 px tall here).

    # Heuristic: half-width = 22 px, height = 80 px. The 0.2 margin will
    # crop the inner 60% so torso pixels (jersey) dominate.
    HALF_W = 22.0
    H_BOX = 80.0

    # Group desired (frame_idx, foot_point, track_id) reads.
    plan: dict[int, list[tuple[int, np.ndarray]]] = {}  # frame_idx -> [(tid, foot_xy)]
    track_ok: dict[int, bool] = {}
    for tid, traj in trajectories.items():
        meas = [(p.frame_idx, p.pixel_xy) for p in traj.points
                if not p.interrupted and p.pixel_xy is not None
                and np.isfinite(p.pixel_xy).all()]
        if len(meas) < n_samples_per_track:
            track_ok[tid] = False
            continue
        track_ok[tid] = True
        # evenly-spaced indices over the measured points
        idxs = np.linspace(0, len(meas) - 1, n_samples_per_track).astype(int)
        for k in idxs:
            fi, foot = meas[int(k)]
            plan.setdefault(int(fi), []).append((tid, foot))

    # Read the video sequentially, picking off frames in `plan`.
    histograms: dict[int, list[np.ndarray]] = {tid: [] for tid in trajectories}
    sorted_frames = sorted(plan.keys())
    target_idx = 0
    fi = 0
    while target_idx < len(sorted_frames):
        target = sorted_frames[target_idx]
        # Read forward until we land on `target`.
        while fi <= target:
            ok, frame = cap.read()
            if not ok:
                fi = -1
                break
            if fi == target:
                # Crop each requested foot-point.
                for tid, foot in plan[target]:
                    cx, cy = float(foot[0]), float(foot[1])
                    box = np.array([
                        cx - HALF_W,
                        cy - H_BOX,    # top of the box ~80 px above feet
                        cx + HALF_W,
                        cy,
                    ])
                    crop = _crop_inner_box(frame, box, box_margin_frac)
                    if crop is None:
                        continue
                    hist = _hsv_histogram(crop)
                    histograms[tid].append(hist)
            fi += 1
        if fi == -1:
            break
        target_idx += 1
    cap.release()

    # Build per-track signatures.
    track_ids: list[int] = []
    sig_rows: list[np.ndarray] = []
    for tid in sorted(trajectories.keys()):
        if not track_ok.get(tid, False):
            continue
        hs = histograms.get(tid, [])
        if not hs:
            continue
        sig = np.mean(np.stack(hs, axis=0), axis=0).astype(np.float32)
        track_ids.append(tid)
        sig_rows.append(sig)

    out: dict[int, str] = {tid: "unknown" for tid in trajectories}
    if len(sig_rows) < 2:
        return out

    X = np.stack(sig_rows, axis=0)
    km = KMeans(n_clusters=2, n_init=10, random_state=0)
    labels = km.fit_predict(X)

    # Determinism: cluster containing the smallest track_id becomes team_A.
    smallest_tid = min(track_ids)
    smallest_idx = track_ids.index(smallest_tid)
    a_label = int(labels[smallest_idx])

    for tid, lab in zip(track_ids, labels):
        out[tid] = "team_A" if int(lab) == a_label else "team_B"
    return out


def select_long_tracks(trajectories: dict[int, PlayerTrajectory],
                          min_meas_frac: float = 0.5,
                          n_valid_frames: int | None = None,
                          ) -> set[int]:
    """Return the set of track_ids that have ≥ min_meas_frac of valid
    frames worth of MEASURED (interrupted=False) points.

    If n_valid_frames is None, uses the max measured count across all
    tracks as the implicit "valid frames" reference (so the cut is
    relative to the longest track in the clip).
    """
    counts: dict[int, int] = {}
    for tid, traj in trajectories.items():
        counts[tid] = sum(1 for p in traj.points if not p.interrupted)
    if not counts:
        return set()
    ref = n_valid_frames if n_valid_frames is not None else max(counts.values())
    threshold = max(1, int(ref * min_meas_frac))
    return {tid for tid, c in counts.items() if c >= threshold}


def classify_teams_by_position(trajectories: dict[int, PlayerTrajectory],
                                  snap_frame_idx: int = 0,
                                  search_window: int = 30,
                                  long_track_ids: set[int] | None = None,
                                  ) -> dict[int, str]:
    """Split tracks into two teams by their FIRST measured NGS-x.

    At/near the snap, offense and defense line up on opposite sides of
    the line of scrimmage. The LoS divides them — and since each team
    has 11 players, the LoS x-coordinate is the *median* of all 22
    first-observed x-positions. So a median split on first-observed x
    yields a clean 11/11 partition without any color processing.

    Why "first measured" and not "at frame 0":
      - Tracks come into view at staggered frames (some players occluded
        in early frames, some refs / late-arrivals appear later).
      - Each track's earliest measurement is the closest snapshot of its
        starting formation we can get.

    Why median split and not K-means:
      - 1D K-means splits at the largest gap, which on a typical play
        is the WR/CB spread rather than the LoS gap (the line of
        scrimmage gap is small — ~1 yd — relative to the formation
        spread). Median split is robust to that.
      - Median by definition gives 11/11 (with even number of tracks).
      - team_A = below-median (lower NGS-x); team_B = above-median.

    Fails cleanly:
      - Special-teams formations (kickoff/punt) where teams span the
        full field width — the median split isn't necessarily LoS-
        aligned but still gives a reasonable left/right partition.
      - If fewer than 4 tracks have measured x, returns all 'unknown'.

    Args:
        trajectories: dict[track_id, PlayerTrajectory]
        snap_frame_idx, search_window: kept for API compatibility, not
            currently used (we accept the EARLIEST measured point per
            track regardless of frame).

    Returns:
        dict[track_id, 'team_A' | 'team_B' | 'unknown']
    """
    first_x: dict[int, float] = {}
    for tid, traj in trajectories.items():
        if long_track_ids is not None and tid not in long_track_ids:
            continue
        for pt in traj.points:
            if pt.field_xy is None or pt.interrupted:
                continue
            first_x[tid] = float(pt.field_xy[0])
            break

    if len(first_x) < 4:
        return {tid: "unknown" for tid in trajectories}

    median_x = float(np.median(list(first_x.values())))
    out: dict[int, str] = {}
    for tid, x in first_x.items():
        out[tid] = "team_A" if x < median_x else "team_B"
    for tid in trajectories:
        if tid not in out:
            out[tid] = "unknown"
    return out







def classify_teams_team_colors(trajectories: dict[int, PlayerTrajectory],
                                  video_path: str,
                                  n_samples_per_track: int = 12,
                                  long_track_ids: set[int] | None = None,
                                  max_pixels_for_clustering: int = 50000,
                                  ) -> tuple[dict[int, str], dict[int, float]]:
    """Identify the two team colors via global pixel clustering, then
    assign each track to the closer color.

    No jersey-region crop — uses the full detection box. Chromatic
    masking still drops grass / white / dark pixels first, so we cluster
    only the saturated chromatic pixels (mostly jersey body color).

    Steps:
      1. Per long-track, sample N evenly-spaced measured frames where
         the actual detection xyxy is available.
      2. Crop the FULL detection box, mask out grass/white/dark, keep
         only chromatic pixels.
      3. Pool all chromatic pixels across all sampled boxes globally.
      4. Cluster pooled pixels into 2 modes (k-means in HSV space, but
         hue weighted up since hue is the team-distinctive axis).
         Subsample to `max_pixels_for_clustering` for speed.
      5. For each track, compute mean distance from its chromatic pixels
         to each of the 2 team-color centers.
      6. Assign the track to the closer team-color cluster.

    Why this beats per-track histogram clustering: histograms compress
    each track to a 36-vector and lose pixel-level information. By
    clustering at the PIXEL level globally, we identify what the actual
    team colors *are* in this clip, then ask each track which one it
    belongs to. Less sensitive to per-track sample variance.

    Returns (labels, confidences). Confidence = (d_other - d_own) / d_own;
    larger = farther from boundary.
    """
    labels: dict[int, str] = {tid: "unknown" for tid in trajectories}
    confidences: dict[int, float] = {tid: 0.0 for tid in trajectories}
    if not trajectories or not os.path.exists(video_path):
        return labels, confidences

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return labels, confidences
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    plan: dict[int, list[tuple[int, np.ndarray]]] = {}
    track_ok: dict[int, bool] = {}
    for tid, traj in trajectories.items():
        if long_track_ids is not None and tid not in long_track_ids:
            track_ok[tid] = False
            continue
        meas = [(p.frame_idx, p.xyxy) for p in traj.points
                if not p.interrupted and p.xyxy is not None]
        if len(meas) < n_samples_per_track:
            track_ok[tid] = False
            continue
        track_ok[tid] = True
        idxs = np.linspace(0, len(meas) - 1, n_samples_per_track).astype(int)
        for k in idxs:
            fi, xyxy = meas[int(k)]
            plan.setdefault(int(fi), []).append((tid, xyxy))

    # Phase 1: gather chromatic pixels per track
    per_track_pixels: dict[int, list[np.ndarray]] = {tid: [] for tid in trajectories}
    sorted_frames = sorted(plan.keys())
    target_idx = 0
    fi = 0
    while target_idx < len(sorted_frames):
        target = sorted_frames[target_idx]
        while fi <= target:
            ok, frame = cap.read()
            if not ok:
                fi = -1
                break
            if fi == target:
                for tid, xyxy in plan[target]:
                    region = _box_crop(xyxy, frame_h, frame_w)
                    if region is None:
                        continue
                    y1, y2, x1, x2 = region
                    crop = frame[y1:y2, x1:x2]
                    chrom = _chromatic_pixels(crop)
                    if chrom is not None and len(chrom) >= 5:
                        per_track_pixels[tid].append(chrom)
            fi += 1
        if fi == -1:
            break
        target_idx += 1
    cap.release()

    # Concatenate per track + filter out tracks with too little data
    per_track: dict[int, np.ndarray] = {}
    for tid, parts in per_track_pixels.items():
        if not parts:
            continue
        cat = np.concatenate(parts, axis=0)
        if len(cat) < 50:
            continue
        per_track[tid] = cat.astype(np.float32)

    if len(per_track) < 2:
        return labels, confidences

    # Phase 2: cluster pooled pixels into 2 team-color modes.
    # Hue is the team-distinctive axis; saturation/value vary with
    # lighting. Weight features so hue dominates the clustering.
    # OpenCV HSV: H ∈ [0,180], S ∈ [0,255], V ∈ [0,255].
    HUE_WEIGHT = 4.0
    SV_WEIGHT = 1.0

    def _featurize(hsv: np.ndarray) -> np.ndarray:
        f = hsv.astype(np.float32).copy()
        # Convert hue to (cos, sin) so the circle wraps correctly,
        # then scale by HUE_WEIGHT.
        h_rad = (f[:, 0:1] / 180.0) * 2.0 * np.pi
        h_cos = np.cos(h_rad) * (180.0 * HUE_WEIGHT)
        h_sin = np.sin(h_rad) * (180.0 * HUE_WEIGHT)
        sv = f[:, 1:3] * SV_WEIGHT
        return np.concatenate([h_cos, h_sin, sv], axis=1)

    pooled = np.concatenate(list(per_track.values()), axis=0)
    if len(pooled) > max_pixels_for_clustering:
        idx = np.random.RandomState(0).choice(
            len(pooled), max_pixels_for_clustering, replace=False)
        pooled = pooled[idx]

    pooled_feat = _featurize(pooled)
    km = KMeans(n_clusters=2, n_init=10, random_state=0).fit(pooled_feat)
    centers_feat = km.cluster_centers_  # (2, 4)

    # Recover (approximate) HSV centers for diagnostics.
    def _unfeat(c: np.ndarray) -> np.ndarray:
        h_cos = c[0] / (180.0 * HUE_WEIGHT)
        h_sin = c[1] / (180.0 * HUE_WEIGHT)
        h_deg = (np.degrees(np.arctan2(h_sin, h_cos)) % 360) / 2.0
        s = c[2] / SV_WEIGHT
        v = c[3] / SV_WEIGHT
        return np.array([h_deg, s, v])

    team_colors_hsv = np.stack([_unfeat(c) for c in centers_feat], axis=0)

    # Phase 3: assign each track by which team color is closer (in
    # weighted feature space) on average.
    track_ids: list[int] = []
    cluster_assignments: list[int] = []
    track_distances: list[tuple[float, float]] = []
    for tid in sorted(per_track.keys()):
        feat = _featurize(per_track[tid])
        d_a = float(np.linalg.norm(feat - centers_feat[0], axis=1).mean())
        d_b = float(np.linalg.norm(feat - centers_feat[1], axis=1).mean())
        cluster_assignments.append(0 if d_a < d_b else 1)
        track_distances.append((d_a, d_b))
        track_ids.append(tid)

    # Determinism: cluster containing the smallest track_id becomes team_A.
    smallest_tid = min(track_ids)
    smallest_idx = track_ids.index(smallest_tid)
    a_label = cluster_assignments[smallest_idx]
    for tid, lab, (d_a, d_b) in zip(track_ids, cluster_assignments, track_distances):
        labels[tid] = "team_A" if lab == a_label else "team_B"
        d_own = d_a if lab == 0 else d_b
        d_other = d_b if lab == 0 else d_a
        confidences[tid] = (d_other - d_own) / (d_own + 1e-6)

    return labels, confidences


def classify_teams_color_pca(trajectories: dict[int, PlayerTrajectory],
                                video_path: str,
                                n_samples_per_track: int = 12,
                                long_track_ids: set[int] | None = None,
                                ) -> tuple[dict[int, str], dict[int, float]]:
    """Per-track HSV signature, baseline-subtract, PCA → 1D, median-split.

    The team-distinctive direction in feature space is the principal
    component of the residual signatures (residual = signature − cohort
    mean). Projecting each track onto PC1 collapses each track to a
    single scalar that orders them along the team-color axis. Median
    split then enforces 11/11 by construction (the strong prior we have
    for NFL plays: exactly 11 vs 11).

    Uses the FULL detection box (no jersey crop). Chromatic masking
    drops grass/white/dark before featurizing.

    Each track's signature is a 24-dim vector: 12-bin hue histogram
    (sum=1) concatenated with 12-bin (S,V) joint histogram normalized
    to unit area. Hue captures the primary color axis; (S,V) captures
    the saturation/value axis where look-alike hues separate.
    """
    labels: dict[int, str] = {tid: "unknown" for tid in trajectories}
    confidences: dict[int, float] = {tid: 0.0 for tid in trajectories}
    if not trajectories or not os.path.exists(video_path):
        return labels, confidences

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return labels, confidences
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    plan: dict[int, list[tuple[int, np.ndarray]]] = {}
    track_ok: dict[int, bool] = {}
    for tid, traj in trajectories.items():
        if long_track_ids is not None and tid not in long_track_ids:
            track_ok[tid] = False
            continue
        meas = [(p.frame_idx, p.xyxy) for p in traj.points
                if not p.interrupted and p.xyxy is not None]
        if len(meas) < n_samples_per_track:
            track_ok[tid] = False
            continue
        track_ok[tid] = True
        idxs = np.linspace(0, len(meas) - 1, n_samples_per_track).astype(int)
        for k in idxs:
            fi, xyxy = meas[int(k)]
            plan.setdefault(int(fi), []).append((tid, xyxy))

    per_track_pixels: dict[int, list[np.ndarray]] = {tid: [] for tid in trajectories}
    sorted_frames = sorted(plan.keys())
    target_idx = 0
    fi = 0
    while target_idx < len(sorted_frames):
        target = sorted_frames[target_idx]
        while fi <= target:
            ok, frame = cap.read()
            if not ok:
                fi = -1
                break
            if fi == target:
                for tid, xyxy in plan[target]:
                    region = _box_crop(xyxy, frame_h, frame_w)
                    if region is None:
                        continue
                    y1, y2, x1, x2 = region
                    crop = frame[y1:y2, x1:x2]
                    chrom = _chromatic_pixels(crop)
                    if chrom is not None and len(chrom) >= 5:
                        per_track_pixels[tid].append(chrom)
            fi += 1
        if fi == -1:
            break
        target_idx += 1
    cap.release()

    H_BINS = 12
    SV_BINS = 4   # 4 S × 3 V grid → 12 bins (we use 4*3=12 to match H_BINS)

    def _signature(chrom_pixels: np.ndarray) -> np.ndarray | None:
        if len(chrom_pixels) < 20:
            return None
        h_hist, _ = np.histogram(chrom_pixels[:, 0], bins=H_BINS, range=(0, 180))
        h_hist = h_hist.astype(np.float32)
        h_hist /= max(h_hist.sum(), 1.0)
        sv_hist, _, _ = np.histogram2d(
            chrom_pixels[:, 1], chrom_pixels[:, 2],
            bins=[SV_BINS, 3], range=[[0, 256], [50, 220]])
        sv_hist = sv_hist.astype(np.float32)
        sv_hist = sv_hist.flatten()
        sv_hist /= max(sv_hist.sum(), 1.0)
        return np.concatenate([h_hist, sv_hist], axis=0)

    track_ids: list[int] = []
    sigs: list[np.ndarray] = []
    for tid in sorted(per_track_pixels.keys()):
        if not track_ok.get(tid, False):
            continue
        parts = per_track_pixels[tid]
        if not parts:
            continue
        cat = np.concatenate(parts, axis=0)
        sig = _signature(cat)
        if sig is None:
            continue
        track_ids.append(tid)
        sigs.append(sig)

    if len(sigs) < 2:
        return labels, confidences

    H = np.stack(sigs, axis=0)
    baseline = H.mean(axis=0, keepdims=True)
    R = H - baseline   # (N, n_bins)

    # PCA in residual space → first principal direction
    # (right singular vector for largest singular value of R).
    U, S_vals, Vt = np.linalg.svd(R, full_matrices=False)
    pc1 = Vt[0]                            # (n_bins,)
    proj = R @ pc1                          # (N,) — 1D coordinate per track

    # Median-split: enforces a balanced partition (the prior is 11/11).
    median = float(np.median(proj))
    cluster_lbls = (proj >= median).astype(int)

    # Determinism: cluster containing the smallest track_id becomes team_A.
    smallest_tid = min(track_ids)
    smallest_idx = track_ids.index(smallest_tid)
    a_label = int(cluster_lbls[smallest_idx])

    # Confidence: |proj − median| / std(proj). Larger = farther from
    # boundary along PC1.
    proj_std = float(proj.std()) + 1e-6
    for tid, lab, p in zip(track_ids, cluster_lbls, proj):
        labels[tid] = "team_A" if int(lab) == a_label else "team_B"
        confidences[tid] = abs(float(p) - median) / proj_std

    return labels, confidences


def classify_teams_color_residual(trajectories: dict[int, PlayerTrajectory],
                                      video_path: str,
                                      n_samples_per_track: int = 12,
                                      long_track_ids: set[int] | None = None,
                                      ) -> tuple[dict[int, str], dict[int, float]]:
    """Cluster tracks by jersey HUE, after masking out grass / white /
    dark pixels and subtracting the cohort baseline.

    The team color is genuinely a small fraction of pixels in the box
    (helmet + pants + grass + arms + numbers all eat into it). Without
    aggressive masking + jersey-region focus + baseline subtraction,
    histogram clustering is dominated by noise that's common across
    both teams.

    Pipeline:
      1. Filter to long-tracks (skip spurious short trajectories).
      2. Per-track sample N evenly-spaced MEASURED frames where the
         actual detection xyxy is available.
      3. Crop the jersey region (y ∈ [0.15, 0.50] of box height,
         skip 20% on each side horizontally).
      4. HSV-mask: drop grass (green H ∈ [35,85], S>60), drop
         white (S<40, V>150), drop dark (V<50), drop glare.
      5. Build a 36-bin hue histogram over remaining chromatic pixels.
      6. Average histograms per track, subtract cohort baseline (mean
         across tracks), KMeans k=2 on residuals.

    Returns (labels, confidences). Confidences = (d_other - d_own) /
    d_own — larger = farther from cluster boundary.
    """
    labels: dict[int, str] = {tid: "unknown" for tid in trajectories}
    confidences: dict[int, float] = {tid: 0.0 for tid in trajectories}
    if not trajectories or not os.path.exists(video_path):
        return labels, confidences

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return labels, confidences
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Plan per-track frame samples + actual xyxy.
    plan: dict[int, list[tuple[int, np.ndarray]]] = {}
    track_ok: dict[int, bool] = {}
    for tid, traj in trajectories.items():
        if long_track_ids is not None and tid not in long_track_ids:
            track_ok[tid] = False
            continue
        meas = [(p.frame_idx, p.xyxy) for p in traj.points
                if not p.interrupted and p.xyxy is not None]
        if len(meas) < n_samples_per_track:
            track_ok[tid] = False
            continue
        track_ok[tid] = True
        idxs = np.linspace(0, len(meas) - 1, n_samples_per_track).astype(int)
        for k in idxs:
            fi, xyxy = meas[int(k)]
            plan.setdefault(int(fi), []).append((tid, xyxy))

    histograms: dict[int, list[np.ndarray]] = {tid: [] for tid in trajectories}
    sorted_frames = sorted(plan.keys())
    target_idx = 0
    fi = 0
    while target_idx < len(sorted_frames):
        target = sorted_frames[target_idx]
        while fi <= target:
            ok, frame = cap.read()
            if not ok:
                fi = -1
                break
            if fi == target:
                for tid, xyxy in plan[target]:
                    region = _jersey_region(xyxy, frame_h, frame_w)
                    if region is None:
                        continue
                    y1, y2, x1, x2 = region
                    crop = frame[y1:y2, x1:x2]
                    chrom = _chromatic_pixels(crop)
                    if chrom is None or len(chrom) < 5:
                        continue
                    hist = _hue_histogram(chrom)
                    histograms[tid].append(hist)
            fi += 1
        if fi == -1:
            break
        target_idx += 1
    cap.release()

    track_ids: list[int] = []
    sigs: list[np.ndarray] = []
    for tid in sorted(trajectories.keys()):
        if not track_ok.get(tid, False):
            continue
        hs = histograms.get(tid, [])
        if not hs:
            continue
        sig = np.mean(np.stack(hs, axis=0), axis=0).astype(np.float32)
        track_ids.append(tid)
        sigs.append(sig)

    if len(sigs) < 2:
        return labels, confidences

    H = np.stack(sigs, axis=0)
    baseline = H.mean(axis=0, keepdims=True)
    R = H - baseline

    km = KMeans(n_clusters=2, n_init=10, random_state=0)
    cluster_lbls = km.fit_predict(R)
    centers = km.cluster_centers_

    for i, tid in enumerate(track_ids):
        d_own = float(np.linalg.norm(R[i] - centers[cluster_lbls[i]]))
        d_other = float(np.linalg.norm(R[i] - centers[1 - cluster_lbls[i]]))
        confidences[tid] = (d_other - d_own) / (d_own + 1e-6)

    smallest_tid = min(track_ids)
    smallest_idx = track_ids.index(smallest_tid)
    a_label = int(cluster_lbls[smallest_idx])
    for tid, lab in zip(track_ids, cluster_lbls):
        labels[tid] = "team_A" if int(lab) == a_label else "team_B"
    return labels, confidences


def classify_teams_hybrid(trajectories: dict[int, PlayerTrajectory],
                              video_path: str,
                              snap_frame_idx: int = 0,
                              n_samples_per_track: int = 8,
                              long_track_ids: set[int] | None = None,
                              ) -> dict[int, str]:
    """Combine position-based and color-residual classifiers.

    Position is rock-solid for players far from the line of scrimmage
    (WRs, safeties, etc.) but degrades for players near the LoS (the
    center is the canonical failure mode — they're right at the median
    and one yard of pre-snap motion flips them).

    Color-residual is the inverse: very confident for players whose
    team has a distinctive color signature, less reliable for players
    whose visible region is mostly common features (helmet + pants).

    Combiner:
      1. Filter to long-tracks (drops spurious late-clip refs/sideline
         figures from the median split and the cluster centroid).
      2. Run BOTH classifiers on the long-track subset.
      3. Where they agree on a label → use that.
      4. Where they disagree on a track → use color-residual IF the
         track's color confidence is above a threshold; else use
         position. Position is the safer default near the boundary
         (because the median is robust); color wins decisively when
         it has a clear signal.

    Tracks not in long_track_ids → 'unknown'.
    """
    if long_track_ids is None:
        long_track_ids = select_long_tracks(trajectories)

    pos_labels = classify_teams_by_position(
        trajectories, snap_frame_idx=snap_frame_idx,
        long_track_ids=long_track_ids)
    color_labels, color_conf = classify_teams_color_residual(
        trajectories, video_path,
        n_samples_per_track=n_samples_per_track,
        long_track_ids=long_track_ids)

    # Reconcile: position and color may use independent A/B labels (each
    # is deterministic by smallest track-id). Align them by counting
    # majority agreement on long tracks.
    long_tids = [tid for tid in long_track_ids
                 if pos_labels.get(tid) in ("team_A", "team_B")
                 and color_labels.get(tid) in ("team_A", "team_B")]
    if not long_tids:
        return {tid: "unknown" for tid in trajectories}
    n_agree = sum(1 for tid in long_tids
                  if pos_labels[tid] == color_labels[tid])
    if n_agree < len(long_tids) - n_agree:
        # Color labels are inverted relative to position. Flip color.
        color_labels = {tid: ("team_B" if v == "team_A" else
                                "team_A" if v == "team_B" else v)
                        for tid, v in color_labels.items()}

    # Tiebreaker threshold: color_conf > 0.05 means the track sits
    # noticeably farther from the boundary in residual space than at it.
    CONF_THRESH = 0.05

    out: dict[int, str] = {tid: "unknown" for tid in trajectories}
    for tid in long_track_ids:
        p = pos_labels.get(tid, "unknown")
        c = color_labels.get(tid, "unknown")
        if p == c and p != "unknown":
            out[tid] = p
        elif p != "unknown" and c != "unknown":
            # disagreement — trust color when confident, else position
            if color_conf.get(tid, 0.0) >= CONF_THRESH:
                out[tid] = c
            else:
                out[tid] = p
        elif p != "unknown":
            out[tid] = p
        elif c != "unknown":
            out[tid] = c
    return out
