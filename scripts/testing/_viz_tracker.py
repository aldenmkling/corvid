"""Render player-tracking visualization. Stacked output:
  top    = source frame (undistorted) with bboxes + track ID labels
  bottom = rectified field canvas with player dots colored by track ID
           (or, with --use-teams, colored by team_A / team_B / unknown)

Without --use-teams, each track gets a stable per-id color (hash of
track_id → HSV hue). With --use-teams, we run a second pass after
tracking that calls classify_teams() and recolors by team affiliation.
Run as: python scripts/testing/_viz_tracker.py --clip <path> --out <path>
        [--use-teams]
"""
import argparse
import os
import sys

import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from src.detector import RFDETRDetector, get_or_build_detection_cache
from src.tracker import PlayerTracker
from src.team_classifier import (
    classify_teams_by_position, classify_teams_color_pca,
    classify_teams_hybrid, select_long_tracks,
)
from src.homography.rectify import (
    compute_homographies, get_or_build_homography_cache, build_rectify_warp,
    DISPLAY_X_LEFT, PX_PER_YARD, RECT_W, RECT_H,
    NGS_X_LEFT_GOAL, NGS_X_RIGHT_GOAL,
)
from src.homography.field_model import FIELD_WIDTH, HASH_Y_NEAR, HASH_Y_FAR


# Fixed BGR colors for the team-colored render.
_TEAM_COLOR = {
    "team_A": (60, 60, 220),   # red
    "team_B": (220, 100, 60),  # blue
    "unknown": (160, 160, 160),
}


def color_for_track(track_id: int) -> tuple[int, int, int]:
    """Stable BGR color from track_id via HSV golden-ratio hashing."""
    hue = (track_id * 47) % 180   # spread hues
    hsv = np.uint8([[[hue, 220, 240]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def color_for_player(track_id: int,
                      team_labels: dict[int, str] | None) -> tuple[int, int, int]:
    """Pick a BGR color for a given track. Falls back to per-id hash when
    team_labels is None or the id is missing from the map."""
    if team_labels is None:
        return color_for_track(track_id)
    lab = team_labels.get(track_id, "unknown")
    return _TEAM_COLOR.get(lab, _TEAM_COLOR["unknown"])


def draw_field_dot(canvas: np.ndarray, field_xy: np.ndarray,
                     track_id: int, conf: float, interrupted: bool,
                     team_labels: dict[int, str] | None = None):
    """Draw a player dot on the rectified canvas at NGS field_xy."""
    x_yd, y_yd = float(field_xy[0]), float(field_xy[1])
    if not (DISPLAY_X_LEFT <= x_yd <= 120 and 0 <= y_yd <= FIELD_WIDTH):
        return
    px = int((x_yd - DISPLAY_X_LEFT) * PX_PER_YARD)
    py = int(y_yd * PX_PER_YARD)
    col = color_for_player(track_id, team_labels)
    if interrupted:
        # hollow circle for predicted-only frames
        cv2.circle(canvas, (px, py), 6, col, 1, cv2.LINE_AA)
    else:
        cv2.circle(canvas, (px, py), 6, col, -1, cv2.LINE_AA)
        cv2.circle(canvas, (px, py), 7, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(canvas, str(track_id), (px + 8, py + 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1, cv2.LINE_AA)


def render_rectified_with_tracks(frame_u, H, fi_points, team_labels=None):
    """Rectified field canvas + per-frame player dots.

    `fi_points`: dict[track_id -> TrajectoryPoint] for the CURRENT frame.
    Includes both measured (interrupted=False) and predicted-forward
    (interrupted=True) tracks. Drawing all of them gives stable,
    moving dots regardless of momentary occlusion.
    """
    canvas = np.zeros((RECT_H, RECT_W, 3), dtype=np.uint8)
    if H is not None:
        Hw = build_rectify_warp(H)
        canvas = cv2.warpPerspective(frame_u, Hw, (RECT_W, RECT_H))
        canvas = (canvas * 0.5).astype(np.uint8)
    # Yardlines + sidelines + hash rows for reference
    for x_yd in range(int(NGS_X_LEFT_GOAL), int(NGS_X_RIGHT_GOAL) + 1, 5):
        x_px = int((x_yd - DISPLAY_X_LEFT) * PX_PER_YARD)
        cv2.line(canvas, (x_px, 0), (x_px, RECT_H - 1),
                  (0, 180, 0), 1, cv2.LINE_AA)
    for y_yd in (0, FIELD_WIDTH, HASH_Y_NEAR, HASH_Y_FAR):
        y_px = int(y_yd * PX_PER_YARD)
        col = (220, 220, 0) if y_yd in (0, FIELD_WIDTH) else (140, 60, 60)
        cv2.line(canvas, (0, y_px), (RECT_W - 1, y_px),
                  col, 1, cv2.LINE_AA)
    # Per-track dots for THIS frame (not the trajectory's last point).
    for tid, pt in fi_points.items():
        if pt.field_xy is None:
            continue
        draw_field_dot(canvas, pt.field_xy, tid,
                        pt.confidence, pt.interrupted,
                        team_labels=team_labels)
    return canvas


def render_source_with_boxes(frame_u, players, trajectories, team_labels=None,
                                long_track_ids=None):
    """Source frame (undistorted) with bboxes + track ID labels.

    If long_track_ids is provided, only renders boxes for those tracks
    (drops the spurious short tracks from the source view too).
    """
    out = frame_u.copy()
    for p in players:
        if long_track_ids is not None and p.track_id not in long_track_ids:
            continue
        col = color_for_player(p.track_id, team_labels)
        x1, y1, x2, y2 = [int(v) for v in p.xyxy]
        cv2.rectangle(out, (x1, y1), (x2, y2), col, 2)
        # Track ID label
        cv2.putText(out, str(p.track_id), (x1, max(y1 - 4, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2, cv2.LINE_AA)
        # Foot-point dot
        fx, fy = int(p.foot_point[0]), int(p.foot_point[1])
        cv2.circle(out, (fx, fy), 3, col, -1, cv2.LINE_AA)
    return out


def stack(top, bot):
    h_t, w_t = top.shape[:2]
    h_b, w_b = bot.shape[:2]
    if w_t != w_b:
        bot = cv2.resize(bot, (w_t, int(h_b * w_t / w_b)))
        h_b, w_b = bot.shape[:2]
    out = np.zeros((h_t + h_b, w_t, 3), dtype=np.uint8)
    out[:h_t] = top
    out[h_t:] = bot
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--detector-weights", default=os.path.join(
        PROJECT_ROOT, "models/rfdetr_best_ema.pth"))
    ap.add_argument("--device", default="mps")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--use-teams", action="store_true",
                    help="run team classification after tracking and color "
                         "dots / boxes by team (red=A, blue=B, gray=unknown)")
    args = ap.parse_args()

    print(f"  clip: {args.clip}")
    print(f"  loading or building homography cache ...")
    rect = get_or_build_homography_cache(
        args.clip, device=args.device, verbose=True)
    if args.max_frames is not None:
        rect["n_frames"] = min(rect["n_frames"], args.max_frames)
        rect["valid_until"] = min(rect["valid_until"], args.max_frames)
    Hs = rect["Hs"]
    K = rect["K"]; dist = rect["dist"]
    valid_until = rect["valid_until"]
    fps = rect["fps"]
    w, h = rect["image_size"]
    n_frames = rect["n_frames"]
    qf = rect["quality_flag"]
    print(f"  H pass: g0={rect['g0_x']:.1f}  valid_until={valid_until}/{n_frames}  "
          f"questionable={qf['questionable']}")
    if K is not None and abs(rect["dist"][0]) > 1e-6:
        undist_x, undist_y = cv2.initUndistortRectifyMap(
            K, dist, None, K, (w, h), cv2.CV_32FC1)
    else:
        undist_x = undist_y = None

    print(f"  loading or building detection cache ...")
    dets_cached = get_or_build_detection_cache(
        clip_path=args.clip,
        weights=args.detector_weights,
        device=args.device,
        conf_thresh=0.3,
        verbose=True,
    )

    # ── Pass 1: run the tracker (no frame writes). Per-frame results are
    # cached so we can replay them in pass 2 with team-colored dots.
    print(f"  pass 1/2 — tracking ...")
    tracker = PlayerTracker()
    per_frame_results = []
    cap = cv2.VideoCapture(args.clip)
    fi = 0
    while fi < n_frames:
        ok, frame = cap.read()
        if not ok:
            break
        H = Hs[fi] if fi < valid_until and fi < len(Hs) else None
        det = dets_cached[fi]
        result = tracker.update(det, frame, H=H, K=K, dist=dist)
        per_frame_results.append(result)
        fi += 1
        if fi % 60 == 0:
            print(f"    track frame {fi}/{n_frames}  "
                  f"active={len(result)}  trajs={len(tracker.trajectories)}")
    cap.release()
    trajs_final = tracker.get_trajectories()

    # Filter to long-tracks (drop spurious short trajectories — refs,
    # sideline figures, brief mis-detections). Render only these.
    long_track_ids = select_long_tracks(trajs_final, min_meas_frac=0.5,
                                            n_valid_frames=valid_until)
    n_total = len(trajs_final)
    print(f"  trajectory filter: {len(long_track_ids)}/{n_total} are "
          f"long tracks; dropping {n_total - len(long_track_ids)} short")

    # Build per-frame index of LONG-TRACK points only.
    points_by_frame = [{} for _ in range(n_frames)]
    for tid, traj in trajs_final.items():
        if tid not in long_track_ids:
            continue
        for pt in traj.points:
            if 0 <= pt.frame_idx < n_frames:
                points_by_frame[pt.frame_idx][tid] = pt

    # ── Optional Layer 4: team classification.
    # Color via PCA + median split on chromatic-pixel histograms gives
    # 11/11 on the canonical clip with 100% agreement vs position.
    # Position-based is an alt for special-teams plays where teams
    # span the full field width.
    team_labels = None
    if args.use_teams:
        print(f"  classifying teams (color PCA + median split) ...")
        team_labels, conf = classify_teams_color_pca(
            trajectories=trajs_final,
            video_path=args.clip,
            n_samples_per_track=12,
            long_track_ids=long_track_ids,
        )
        n_a = sum(1 for v in team_labels.values() if v == "team_A")
        n_b = sum(1 for v in team_labels.values() if v == "team_B")
        n_unk = sum(1 for v in team_labels.values() if v == "unknown")
        print(f"    team_A={n_a}  team_B={n_b}  unknown={n_unk}")

    # ── Pass 2: re-iterate the video and render the viz.
    print(f"  pass 2/2 — rendering ...")
    cap = cv2.VideoCapture(args.clip)
    writer = None
    fi = 0
    while fi < n_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if undist_x is not None:
            frame_u = cv2.remap(frame, undist_x, undist_y, cv2.INTER_LINEAR)
        else:
            frame_u = frame.copy()

        H = Hs[fi] if fi < valid_until and fi < len(Hs) else None
        result = per_frame_results[fi]

        top = render_source_with_boxes(frame_u, result.players, trajs_final,
                                        team_labels=team_labels,
                                        long_track_ids=long_track_ids)
        bot = render_rectified_with_tracks(frame_u, H, points_by_frame[fi],
                                            team_labels=team_labels)
        hud_team = " (team-colored)" if args.use_teams else ""
        cv2.putText(top, f"frame {fi}/{n_frames}  tracks_active={len(result)}  "
                          f"trajs={len(trajs_final)}{hud_team}",
                    (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 2, cv2.LINE_AA)
        canvas = stack(top, bot)

        if writer is None:
            os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
            writer = cv2.VideoWriter(
                args.out, cv2.VideoWriter_fourcc(*"mp4v"),
                fps, (canvas.shape[1], canvas.shape[0]))
        writer.write(canvas)

        fi += 1
        if fi % 60 == 0:
            print(f"    render frame {fi}/{n_frames}")
    cap.release()
    if writer is not None:
        writer.release()

    # Summary
    longs = [t for t in trajs_final.values()
              if sum(1 for p in t.points if not p.interrupted) >= valid_until * 0.5]
    print(f"  done: {len(trajs_final)} total trajectories, "
          f"{len(longs)} long (>=50% of valid frames)")
    print(f"  out: {args.out}")


if __name__ == "__main__":
    main()
