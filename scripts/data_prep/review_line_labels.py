#!/usr/bin/env python3
"""
Interactive hand-review of self-sup line labels.

Shows each frame with its cyan/yellow overlay (with snap applied, matching
the dataset builder). Keystrokes:
  space / y / → : keep whole frame
  d / n        : reject whole frame
  c            : crop — drag a box to keep only that region, press enter to submit
  b / ←        : back one
  s            : skip (no decision, advance)
  r            : redraw / reset current decision
  q / ESC      : quit + save

Decisions saved to JSON after every keystroke. Resume by re-running the same
command; it picks up at the first undecided frame. Decision format:
  {fid: {"status": "keep"}}
  {fid: {"status": "reject"}}
  {fid: {"status": "crop", "crop": [x, y, w, h]}}

Usage:
  python scripts/data_prep/review_line_labels.py
"""

import argparse
import csv
import json
import os
import pickle
import sys

import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts/data_prep"))

from build_line_dataset import render_masks, grab_frame


def load_candidates(pool_dir, min_white, require_both_sides):
    path = os.path.join(pool_dir, "summary_hquality.csv")
    if not os.path.exists(path):
        print(f"missing {path}. Run filter_h_quality.py first.")
        sys.exit(1)
    rows = list(csv.DictReader(open(path)))
    out = []
    for r in rows:
        if r.get("yard_white_min") in (None, "", "None"):
            continue
        try:
            ywm = float(r["yard_white_min"])
        except ValueError:
            continue
        if ywm < min_white:
            continue
        if require_both_sides and int(r.get("both_sides_covered", "0") or "0") != 1:
            continue
        out.append(r)
    return out


def normalize_decision(v):
    """Accept legacy string form ('keep'/'reject') or new dict form.
    Always return a dict."""
    if v is None:
        return None
    if isinstance(v, str):
        return {"status": v}
    return v


def render_overlay(r, pool_dir, clips_dir):
    """Return (overlay_bgr, raw_frame)."""
    mp4 = os.path.join(clips_dir, r["game"], r["play"], f"{r['angle']}.mp4")
    frame = grab_frame(mp4, int(r["frame_idx"]))
    if frame is None:
        return None, None
    with open(os.path.join(pool_dir, r["h_path"]), "rb") as f:
        hd = pickle.load(f)
    # Pass the full frame (not .shape) so snap runs — matches build_line_dataset.
    yard, side = render_masks(frame, hd["H"], hd["k1"], hd["k2"])
    ov = frame.copy()
    color = np.zeros_like(ov)
    color[yard > 0] = (255, 255, 0)
    color[side > 0] = (0, 255, 255)
    any_m = (yard > 0) | (side > 0)
    ov[any_m] = cv2.addWeighted(ov, 0.3, color, 0.7, 0)[any_m]
    return ov, frame


def annotate(frame, idx, total, fid, decision, ywm):
    h, w = frame.shape[:2]
    bar = np.zeros((60, w, 3), dtype=np.uint8)
    if decision is None:
        status, color = "?", (255, 255, 255)
    else:
        s = decision.get("status", "?")
        status = s.upper()
        color = {"KEEP": (0, 255, 0), "REJECT": (0, 0, 255),
                 "CROP": (0, 255, 255)}.get(status, (255, 255, 255))
    cv2.putText(bar, f"[{idx+1}/{total}] {status}  {fid}  ywm={ywm:.2f}",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
    cv2.putText(bar, "space/y keep  d reject  c crop  b back  s skip  r reset  q quit",
                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    return np.vstack([bar, frame])


def _aspect_locked_rect(anchor, cursor, target_aspect, img_shape):
    """Given drag anchor + current cursor, return an (x, y, w, h) rect with
    aspect = target_aspect (width / height). Expand the shorter dimension
    to match, anchor at the drag start, clamp to image bounds."""
    H_img, W_img = img_shape[:2]
    ax, ay = anchor
    cx, cy = cursor
    raw_w = abs(cx - ax)
    raw_h = abs(cy - ay)
    if raw_w == 0 or raw_h == 0:
        return None
    # Expand the short side to match target aspect
    if raw_w / raw_h < target_aspect:
        w = int(raw_h * target_aspect)
        h = raw_h
    else:
        w = raw_w
        h = int(raw_w / target_aspect)
    sign_x = 1 if cx >= ax else -1
    sign_y = 1 if cy >= ay else -1
    fx = ax + sign_x * w
    fy = ay + sign_y * h
    x0 = min(ax, fx); y0 = min(ay, fy)
    # Clamp top-left
    x0 = max(0, min(x0, W_img - 1))
    y0 = max(0, min(y0, H_img - 1))
    # Shrink if the rect exceeds the frame (maintain aspect)
    if x0 + w > W_img:
        w = W_img - x0
        h = int(w / target_aspect)
    if y0 + h > H_img:
        h = H_img - y0
        w = int(h * target_aspect)
    if w < 5 or h < 5:
        return None
    return (x0, y0, w, h)


def run_crop(win, overlay, target_aspect):
    """Mouse-drag crop with live aspect-ratio lock.
    target_aspect = width / height.  Returns (x, y, w, h) or None."""
    state = {"anchor": None, "cursor": None, "dragging": False, "rect": None}

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            state["anchor"] = (x, y)
            state["cursor"] = (x, y)
            state["dragging"] = True
        elif event == cv2.EVENT_MOUSEMOVE and state["dragging"]:
            state["cursor"] = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            state["cursor"] = (x, y)
            state["dragging"] = False

    cv2.setMouseCallback(win, on_mouse)

    try:
        while True:
            disp = overlay.copy()
            if state["anchor"] and state["cursor"]:
                r = _aspect_locked_rect(state["anchor"], state["cursor"],
                                         target_aspect, overlay.shape)
                if r is not None:
                    state["rect"] = r
                    x, y, w, h = r
                    cv2.rectangle(disp, (x, y), (x + w, y + h), (0, 255, 255), 2)
            cv2.putText(disp, "CROP MODE  drag to select  Enter=submit  ESC=cancel  R=retry",
                        (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.imshow(win, disp)
            key = cv2.waitKey(20) & 0xFF
            if key in (13, 10):       # Enter
                return state["rect"]
            if key == 27:             # ESC
                return None
            if key in (ord("r"), ord("R")):
                state["anchor"] = None
                state["cursor"] = None
                state["rect"] = None
    finally:
        cv2.setMouseCallback(win, lambda *a, **k: None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool-dir",
                    default=os.path.join(PROJECT_ROOT, "output/self_sup_pool_10k"))
    ap.add_argument("--clips-dir",
                    default=os.path.join(PROJECT_ROOT, "videos/clips"))
    ap.add_argument("--min-white", type=float, default=0.7)
    ap.add_argument("--require-both-sides", action="store_true", default=True)
    ap.add_argument("--no-require-both-sides", action="store_false",
                    dest="require_both_sides")
    ap.add_argument("--out", default=None,
                    help="Decisions JSON (default: <pool>/review_decisions.json)")
    args = ap.parse_args()

    decisions_path = args.out or os.path.join(args.pool_dir, "review_decisions.json")
    decisions = {}
    if os.path.exists(decisions_path):
        loaded = json.load(open(decisions_path))
        decisions = {k: normalize_decision(v) for k, v in loaded.items()}
        print(f"loaded {len(decisions)} prior decisions from {decisions_path}")

    cands = load_candidates(args.pool_dir, args.min_white, args.require_both_sides)
    print(f"{len(cands)} candidates after auto-filter")

    idx = 0
    for i, r in enumerate(cands):
        if r["frame_id"] not in decisions:
            idx = i
            break

    win = "review"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 1280, 780)

    while 0 <= idx < len(cands):
        r = cands[idx]
        fid = r["frame_id"]
        ov, _ = render_overlay(r, args.pool_dir, args.clips_dir)
        if ov is None:
            print(f"SKIP (could not render): {fid}")
            idx += 1
            continue

        prior = decisions.get(fid)
        # If a prior crop exists, draw the saved box on the overlay for context
        if prior and prior.get("status") == "crop" and "crop" in prior:
            x, y, w, h = prior["crop"]
            cv2.rectangle(ov, (x, y), (x + w, y + h), (0, 255, 255), 3)

        shown = annotate(ov, idx, len(cands), fid, prior,
                         float(r["yard_white_min"]))
        cv2.imshow(win, shown)
        key = cv2.waitKey(0) & 0xFF

        if key in (ord(" "), ord("y"), 83):
            decisions[fid] = {"status": "keep"}; idx += 1
        elif key in (ord("d"), ord("n")):
            decisions[fid] = {"status": "reject"}; idx += 1
        elif key == ord("c"):
            H_img, W_img = ov.shape[:2]
            target_aspect = W_img / H_img      # lock crop to frame aspect
            crop = run_crop(win, ov, target_aspect)
            if crop is not None:
                decisions[fid] = {"status": "crop",
                                   "crop": [crop[0], crop[1], crop[2], crop[3]]}
                idx += 1
            # If cancelled, stay on the current frame.
        elif key in (ord("b"), 81):
            idx = max(0, idx - 1)
        elif key == ord("s"):
            idx += 1
        elif key == ord("r"):
            # Reset: clear decision on current frame, stay here.
            decisions.pop(fid, None)
        elif key in (ord("q"), 27):
            break

        json.dump(decisions, open(decisions_path, "w"), indent=2)

    cv2.destroyAllWindows()
    n_keep = sum(1 for v in decisions.values() if v.get("status") == "keep")
    n_reject = sum(1 for v in decisions.values() if v.get("status") == "reject")
    n_crop = sum(1 for v in decisions.values() if v.get("status") == "crop")
    print(f"\n{n_keep + n_reject + n_crop}/{len(cands)} decided  "
          f"({n_keep} keep, {n_reject} reject, {n_crop} crop)")
    print(f"decisions → {decisions_path}")


if __name__ == "__main__":
    main()
