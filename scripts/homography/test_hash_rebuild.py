"""
Step-by-step hash mark detection rebuild.

Saves debug images at each stage so we can verify visually.
Run: python scripts/test_hash_rebuild.py
"""

import cv2
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from scripts.homography.yard_lines import detect_yard_lines

OUT_DIR = "output/homography_test/hash_rebuild"
os.makedirs(OUT_DIR, exist_ok=True)

PLAYS = [1, 5, 34, 50, 75, 100, 150]
CLIP_DIR = "videos/clips/2019092204"


def load_frame(play_num: int) -> np.ndarray:
    """Load frame 1 (skip frame 0) from a play's sideline video."""
    path = f"{CLIP_DIR}/play_{play_num:03d}/sideline.mp4"
    cap = cv2.VideoCapture(path)
    cap.read()  # skip frame 0
    ret, frame = cap.read()  # frame 1
    cap.release()
    if not ret:
        raise RuntimeError(f"Cannot read frame from {path}")
    return frame


def step1_yard_lines(play_num: int, frame: np.ndarray):
    """Step 1: Detect yard lines and save overlay."""
    result = detect_yard_lines(frame)
    if result is None:
        print(f"  Play {play_num}: No yard lines detected!")
        return None

    # Draw overlay
    vis = frame.copy()
    for i, line in enumerate(result.lines):
        x_top, y_top, x_bot, y_bot = line
        cv2.line(vis, (int(x_top), int(y_top)), (int(x_bot), int(y_bot)),
                 (0, 255, 0), 2)
        # Label with grid position
        mx = int((x_top + x_bot) / 2)
        my = int(frame.shape[0] / 2)
        cv2.putText(vis, str(result.grid[i]), (mx - 5, my),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    cv2.imwrite(f"{OUT_DIR}/play{play_num:03d}_s1_yardlines.png", vis)
    print(f"  Play {play_num}: {len(result.lines)} yard lines, "
          f"period={result.period_px:.1f}px, {result.elapsed_ms:.0f}ms")
    return result


def step2_edge_cleanup(play_num: int, frame: np.ndarray, yl_result):
    """Step 2: YL removal → density filter → band mask → hash_canny.

    Order: remove yard line edges first, then density filter globally,
    then restrict to 16px band around yard lines.
    """
    h, w = frame.shape[:2]
    gray = yl_result.gray
    canny = yl_result.canny

    # Sobel gradient → edge direction (rotate by 90°)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    edge_angle = np.arctan2(sobely, sobelx) + np.pi / 2

    BAND_WIDTH = 12
    ANGLE_TOL = np.radians(25)
    DENSITY_THRESH = 0.06

    # --- Step 2a: Remove yard-line-parallel edges (globally in band) ---
    angle_removal = np.zeros((h, w), dtype=bool)
    band_mask = np.zeros((h, w), dtype=np.uint8)

    for line in yl_result.lines:
        x_top, y_top, x_bot, y_bot = line
        ldx = x_bot - x_top
        ldy = y_bot - y_top
        length = np.sqrt(ldx**2 + ldy**2)
        nx = -ldy / length
        ny = ldx / length
        line_angle = np.arctan2(ldy, ldx)

        line_band = np.zeros((h, w), dtype=np.uint8)
        n_steps = int(length * 2)
        for step in range(n_steps + 1):
            t = step / n_steps
            cx = x_top + t * ldx
            cy = y_top + t * ldy
            for sign in [-1, 1]:
                for dist in range(0, BAND_WIDTH + 1):
                    px = int(cx + sign * dist * nx)
                    py = int(cy + sign * dist * ny)
                    if 0 <= px < w and 0 <= py < h:
                        line_band[py, px] = 255
        band_mask = cv2.bitwise_or(band_mask, line_band)

        # Remove edges parallel to this yard line
        angle_diff = edge_angle - line_angle
        angle_diff = (angle_diff + np.pi) % (2 * np.pi) - np.pi
        match = (
            (np.abs(angle_diff) < ANGLE_TOL)
            | (np.abs(angle_diff - np.pi) < ANGLE_TOL)
            | (np.abs(angle_diff + np.pi) < ANGLE_TOL)
        )
        angle_removal[(line_band > 0) & match] = True

    canny_no_yl = canny.copy()
    canny_no_yl[angle_removal] = 0

    # --- Step 2b: Density filter (6%, applied globally) ---
    dens = cv2.boxFilter(canny_no_yl.astype(np.float32) / 255.0, -1, (31, 31))
    sparse = (dens < DENSITY_THRESH).astype(np.uint8) * 255
    clean = cv2.bitwise_and(canny_no_yl, sparse)

    # --- Step 2c: Restrict to band around yard lines ---
    hash_canny = cv2.bitwise_and(clean, band_mask)

    # Save debug images
    cv2.imwrite(f"{OUT_DIR}/play{play_num:03d}_s2a_canny_raw.png", canny)
    cv2.imwrite(f"{OUT_DIR}/play{play_num:03d}_s2b_canny_no_yl.png", canny_no_yl)
    cv2.imwrite(f"{OUT_DIR}/play{play_num:03d}_s2c_clean.png", clean)
    cv2.imwrite(f"{OUT_DIR}/play{play_num:03d}_s2d_band_mask.png", band_mask)
    cv2.imwrite(f"{OUT_DIR}/play{play_num:03d}_s2e_hash_canny.png", hash_canny)

    edge_count = np.count_nonzero(hash_canny)
    print(f"  Play {play_num}: hash_canny has {edge_count} edge pixels")
    return hash_canny


def step3_hash_walk(play_num: int, frame: np.ndarray, yl_result, hash_canny):
    """Step 3: Walk along each yard line, find paired hash marks."""
    h, w = frame.shape[:2]
    INNER_DIST = 6
    OUTER_DIST = 14
    CLUSTER_GAP = 4
    CLUSTER_MIN = 2
    CLUSTER_MATCH = 12
    d_mid = (INNER_DIST + OUTER_DIST) / 2

    all_line_hashes = {}

    for i, line in enumerate(yl_result.lines):
        x_top, y_top, x_bot, y_bot = line
        ldx = x_bot - x_top
        ldy = y_bot - y_top
        length = np.sqrt(ldx**2 + ldy**2)
        nx_l = -ldy / length
        ny_l = ldx / length
        n_steps = int(length)
        if n_steps < 10:
            continue

        # Expected step offset for tilted lines (0.64 correction for perspective)
        expected_offset = (
            0.64 * 2 * d_mid * ny_l * n_steps / ldy if abs(ldy) > 1e-6 else 0.0
        )

        left_steps = set()
        right_steps = set()

        for step in range(n_steps + 1):
            t = step / n_steps
            cx = x_top + t * ldx
            cy = y_top + t * ldy
            for dist in range(INNER_DIST, OUTER_DIST + 1):
                px_c = int(cx - dist * nx_l)
                py_c = int(cy - dist * ny_l)
                if 0 <= px_c < w and 0 <= py_c < h and hash_canny[py_c, px_c] > 0:
                    left_steps.add(step)
                    break
            for dist in range(INNER_DIST, OUTER_DIST + 1):
                px_c = int(cx + dist * nx_l)
                py_c = int(cy + dist * ny_l)
                if 0 <= px_c < w and 0 <= py_c < h and hash_canny[py_c, px_c] > 0:
                    right_steps.add(step)
                    break

        # Cluster
        def cluster_steps(steps):
            if not steps:
                return []
            sorted_s = sorted(steps)
            groups = [[sorted_s[0]]]
            for k in range(1, len(sorted_s)):
                if sorted_s[k] - sorted_s[k - 1] <= CLUSTER_GAP:
                    groups[-1].append(sorted_s[k])
                else:
                    groups.append([sorted_s[k]])
            return groups

        l_clusters = [g for g in cluster_steps(left_steps) if len(g) >= CLUSTER_MIN]
        r_clusters = [g for g in cluster_steps(right_steps) if len(g) >= CLUSTER_MIN]
        l_centers = [(min(g) + max(g)) / 2.0 for g in l_clusters]
        r_centers = [(min(g) + max(g)) / 2.0 for g in r_clusters]

        # Match left/right clusters (subtract expected offset, check tolerance)
        CLUSTER_MATCH = 5
        hash_ts = []
        used_r = set()
        for lc in l_centers:
            adjusted_lc = lc - expected_offset
            best_ri = None
            best_dist = CLUSTER_MATCH + 1
            for ri, rc in enumerate(r_centers):
                if ri in used_r:
                    continue
                d = abs(adjusted_lc - rc)
                if d < best_dist:
                    best_dist = d
                    best_ri = ri
            if best_ri is not None and best_dist <= CLUSTER_MATCH:
                used_r.add(best_ri)
                hash_ts.append((lc + r_centers[best_ri]) / 2 / n_steps)

        # Filter out edge-of-frame noise
        hash_ts = [t for t in hash_ts if 0.15 < t < 0.85]
        if hash_ts:
            all_line_hashes[i] = hash_ts

    # Print per-line results
    for li, ts in sorted(all_line_hashes.items()):
        ts_str = ", ".join(f"{t:.3f}" for t in sorted(ts))
        print(f"    Line {li} (grid {yl_result.grid[li]}): t = [{ts_str}]")

    return all_line_hashes


def step4_classify(play_num, all_line_hashes, yl_result):
    """Step 4: Classify hashes as far/near, compute spread."""
    T_TOL = 0.15

    if not all_line_hashes:
        print(f"  Play {play_num}: No hashes found!")
        return None

    # Find confident pairs (lines with exactly 2 hashes)
    confident_pairs = {}
    far_ts = []
    near_ts = []

    for li, ts in all_line_hashes.items():
        if len(ts) == 2:
            ts_sorted = sorted(ts)
            confident_pairs[li] = (ts_sorted[0], ts_sorted[1])
            far_ts.append(ts_sorted[0])
            near_ts.append(ts_sorted[1])

    if far_ts and near_ts:
        far_median = float(np.median(far_ts))
        near_median = float(np.median(near_ts))
    else:
        # Fallback clustering
        all_ts = sorted(
            t for ts in all_line_hashes.values() for t in ts
            if 0.15 < t < 0.85
        )
        if len(all_ts) < 2:
            print(f"  Play {play_num}: Not enough t-values for classification")
            return None

        t_clusters = [[all_ts[0]]]
        for t_val in all_ts[1:]:
            if t_val - t_clusters[-1][-1] < 0.05:
                t_clusters[-1].append(t_val)
            else:
                t_clusters.append([t_val])

        if len(t_clusters) < 2:
            print(f"  Play {play_num}: Only 1 t-cluster, can't classify")
            return None
        t_clusters.sort(key=len, reverse=True)
        top2 = sorted(t_clusters[:2], key=lambda c: np.mean(c))
        far_median = float(np.median(top2[0]))
        near_median = float(np.median(top2[1]))

    mid_t = (far_median + near_median) / 2
    pair_gap = near_median - far_median

    # Classify
    final_hashes = {}
    for li, ts in all_line_hashes.items():
        entry = {}
        far_cands = [t for t in ts if t < mid_t]
        near_cands = [t for t in ts if t >= mid_t]

        if far_cands:
            best = min(far_cands, key=lambda t: abs(t - far_median))
            if abs(best - far_median) < T_TOL:
                entry["far"] = best
        if near_cands:
            best = min(near_cands, key=lambda t: abs(t - near_median))
            if abs(best - near_median) < T_TOL:
                entry["near"] = best
        if entry:
            final_hashes[li] = entry

    # Interpolate missing
    for entry in final_hashes.values():
        if "far" not in entry and "near" in entry:
            entry["far"] = entry["near"] - pair_gap
        elif "near" not in entry and "far" in entry:
            entry["near"] = entry["far"] + pair_gap

    # Compute spread (pixel distance between far/near hash positions)
    far_pixels = []
    near_pixels = []
    for li, entry in final_hashes.items():
        line = yl_result.lines[li]
        x_top, y_top, x_bot, y_bot = line
        ldy = y_bot - y_top
        if "far" in entry:
            far_pixels.append(y_top + entry["far"] * ldy)
        if "near" in entry:
            near_pixels.append(y_top + entry["near"] * ldy)

    if far_pixels:
        far_spread = max(far_pixels) - min(far_pixels)
    else:
        far_spread = float('inf')
    if near_pixels:
        near_spread = max(near_pixels) - min(near_pixels)
    else:
        near_spread = float('inf')

    print(f"  Play {play_num}: {len(confident_pairs)} confident pairs, "
          f"{len(final_hashes)} total lines with hashes")
    print(f"    far_median={far_median:.3f}, near_median={near_median:.3f}, "
          f"gap={pair_gap:.3f}")
    print(f"    far_spread={far_spread:.1f}px, near_spread={near_spread:.1f}px")

    return final_hashes, far_median, near_median


def step5_overlay(play_num, frame, yl_result, final_hashes):
    """Step 5: Draw hash marks on frame for visual verification."""
    vis = frame.copy()

    # Draw yard lines
    for line in yl_result.lines:
        x_top, y_top, x_bot, y_bot = line
        cv2.line(vis, (int(x_top), int(y_top)), (int(x_bot), int(y_bot)),
                 (0, 255, 0), 1)

    # Draw hash marks
    for li, entry in final_hashes.items():
        line = yl_result.lines[li]
        x_top, y_top, x_bot, y_bot = line
        ldx = x_bot - x_top
        ldy = y_bot - y_top

        for label, color in [("far", (255, 0, 0)), ("near", (0, 0, 255))]:
            if label in entry:
                t = entry[label]
                px = int(x_top + t * ldx)
                py = int(y_top + t * ldy)
                cv2.circle(vis, (px, py), 5, color, -1)
                cv2.putText(vis, label[0].upper(), (px + 7, py + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    cv2.imwrite(f"{OUT_DIR}/play{play_num:03d}_s5_hashes.png", vis)


HASH_SPACING_YD = 18.5 / 3  # 18'6" = 6.167 yards


def step6_rectify(play_num, frame, yl_result, final_hashes):
    """Step 6: Rectify using hash-yard line intersections as correspondences."""
    h, w = frame.shape[:2]
    scale = 20.0       # pixels per yard in output
    margin_x = 100.0
    margin_y = 100.0

    src_pts = []
    dst_pts = []

    for li, entry in final_hashes.items():
        line = yl_result.lines[li]
        gpos = yl_result.grid[li]
        x_top, y_top, x_bot, y_bot = line
        ldx = x_bot - x_top
        ldy = y_bot - y_top
        target_x = margin_x + gpos * 5 * scale

        if "far" in entry:
            t = entry["far"]
            src_pts.append([x_top + t * ldx, y_top + t * ldy])
            dst_pts.append([target_x, margin_y])
        if "near" in entry:
            t = entry["near"]
            src_pts.append([x_top + t * ldx, y_top + t * ldy])
            dst_pts.append([target_x, margin_y + HASH_SPACING_YD * scale])

    if len(src_pts) < 4:
        print(f"  Play {play_num}: Only {len(src_pts)} correspondences, need 4+")
        return

    src = np.array(src_pts, np.float64)
    dst = np.array(dst_pts, np.float64)
    H, _ = cv2.findHomography(src, dst, 0)
    if H is None:
        print(f"  Play {play_num}: findHomography failed")
        return

    # Warp corners to find output bounds
    corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float64)
    corners_h = np.hstack([corners, np.ones((4, 1))])
    wc = (H @ corners_h.T).T
    wc = wc[:, :2] / wc[:, 2:3]

    x_min = int(np.floor(wc[:, 0].min()))
    x_max = int(np.ceil(wc[:, 0].max()))
    y_min = int(np.floor(wc[:, 1].min()))
    y_max = int(np.ceil(wc[:, 1].max()))

    # Translation to keep everything positive
    tx = float(-x_min)
    ty = float(-y_min)
    T = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]], dtype=np.float64)
    H_full = T @ H

    out_w = x_max - x_min
    out_h = y_max - y_min
    warped = cv2.warpPerspective(frame, H_full, (out_w, out_h))

    # Draw grid lines on the warped image for verification
    vis = warped.copy()
    max_grid = max(yl_result.grid)
    for g in range(max_grid + 1):
        x = int(margin_x + g * 5 * scale + tx)
        if 0 <= x < out_w:
            cv2.line(vis, (x, 0), (x, out_h), (0, 255, 0), 1)
    # Draw hash rows
    for y_off in [0, HASH_SPACING_YD * scale]:
        y = int(margin_y + y_off + ty)
        if 0 <= y < out_h:
            cv2.line(vis, (0, y), (out_w, y), (0, 200, 255), 1)

    cv2.imwrite(f"{OUT_DIR}/play{play_num:03d}_s6_rectified.png", vis)

    n_far = sum(1 for e in final_hashes.values() if "far" in e)
    n_near = sum(1 for e in final_hashes.values() if "near" in e)
    print(f"  Play {play_num}: rectified {out_w}x{out_h}, "
          f"{len(src_pts)} pts ({n_far} far, {n_near} near)")


def main():
    print("=== Hash Mark Detection Rebuild ===\n")

    for play_num in PLAYS:
        print(f"--- Play {play_num} ---")
        try:
            frame = load_frame(play_num)
        except RuntimeError as e:
            print(f"  {e}")
            continue

        # Step 1: Yard lines
        yl_result = step1_yard_lines(play_num, frame)
        if yl_result is None:
            continue

        # Step 2: Edge cleanup
        hash_canny = step2_edge_cleanup(play_num, frame, yl_result)

        # Step 3: Hash walk
        all_line_hashes = step3_hash_walk(play_num, frame, yl_result, hash_canny)

        # Step 4: Classify
        result = step4_classify(play_num, all_line_hashes, yl_result)
        if result is None:
            continue

        final_hashes, far_median, near_median = result

        # Step 5: Overlay
        step5_overlay(play_num, frame, yl_result, final_hashes)

        # Step 6: Rectify
        step6_rectify(play_num, frame, yl_result, final_hashes)

        print()


if __name__ == "__main__":
    main()
