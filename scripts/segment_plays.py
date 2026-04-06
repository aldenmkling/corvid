#!/usr/bin/env python3
"""
Segment an All-22 game video into per-play clips (sideline + endzone).

Each play in All-22 film follows the pattern:
  [scoreboard] → sideline wide → endzone tight → [scoreboard] → ...

Detection:
  1. Find all hard cuts via local-ratio spike detection on frame diffs.
  2. Classify each segment between cuts as "field" or "non-field" (scoreboard)
     using HSV green percentage.
  3. Group consecutive field segments into plays: the first field segment
     after a non-field (or start) is sideline, the next is endzone.
  4. If no scoreboards exist, field segments alternate SL/EZ, and a new
     play starts each time EZ→SL occurs.

Usage:
  python scripts/segment_plays.py --video <path> --game-id <id> [--output data/clips/]
  python scripts/segment_plays.py --video <path> --game-id <id> --preview
"""

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass

import cv2
import numpy as np


# ── Configuration ──────────────────────────────────────────────────────────

CUT_RATIO_THRESHOLD = 4.0    # frame diff must be >= this × local median
CUT_MIN_DIFF = 15.0           # absolute minimum diff for a cut
CUT_NEIGHBOR_WINDOW = 15      # frames on each side for local median (~0.5s)
CUT_MIN_GAP_S = 0.3           # minimum seconds between detected cuts
GREEN_FIELD_THRESHOLD = 0.20  # min green % to classify segment as field
MIN_SEGMENT_DURATION_S = 1.0  # ignore segments shorter than this


# ── Data structures ────────────────────────────────────────────────────────

@dataclass
class Segment:
    start_s: float
    end_s: float
    is_field: bool

    @property
    def duration(self) -> float:
        return self.end_s - self.start_s


@dataclass
class PlayClip:
    play_num: int
    sideline_start: float
    sideline_end: float
    endzone_start: float
    endzone_end: float


# ── Hard cut detection ─────────────────────────────────────────────────────

def detect_cuts(video_path: str) -> list[float]:
    """Detect hard cuts by finding frames where diff spikes relative to neighbors."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Cannot open {video_path}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_s = total_frames / fps
    print(f"Video: {duration_s:.0f}s ({duration_s/60:.1f} min), {fps:.2f} fps")
    print(f"Scanning {total_frames} frames for hard cuts...")

    prev_gray = None
    diffs = []

    frame_num = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(
            cv2.resize(frame, (320, 180)), cv2.COLOR_BGR2GRAY
        ).astype(np.float32)
        if prev_gray is not None:
            diff = float(np.mean(np.abs(gray - prev_gray)))
            diffs.append(diff)
        prev_gray = gray
        frame_num += 1

        if frame_num % (int(fps) * 300) == 0:
            pct = frame_num / total_frames * 100
            print(f"  {pct:.0f}% ({frame_num/fps:.0f}s / {duration_s:.0f}s)")

    cap.release()
    diffs_arr = np.array(diffs)

    # Find local-ratio spikes
    w = CUT_NEIGHBOR_WINDOW
    cut_frames = []

    for i in range(w, len(diffs_arr) - w):
        if diffs_arr[i] < CUT_MIN_DIFF:
            continue
        neighborhood = np.concatenate([diffs_arr[i - w : i], diffs_arr[i + 1 : i + w + 1]])
        local_median = np.median(neighborhood)
        if local_median <= 0:
            continue
        ratio = diffs_arr[i] / local_median
        if ratio >= CUT_RATIO_THRESHOLD:
            # Check minimum gap from previous cut
            t = (i + 1) / fps
            if not cut_frames or t - cut_frames[-1] > CUT_MIN_GAP_S:
                cut_frames.append(t)

    print(f"  Found {len(cut_frames)} hard cuts")
    return cut_frames


# ── Segment classification ─────────────────────────────────────────────────

def classify_segments(
    video_path: str, cuts: list[float], duration_s: float
) -> list[Segment]:
    """Classify each segment between cuts as field or non-field."""
    cap = cv2.VideoCapture(video_path)

    # Build segment boundaries: [0, cut1, cut2, ..., end]
    boundaries = [0.0] + cuts + [duration_s]
    segments = []

    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end = boundaries[i + 1]
        dur = end - start

        if dur < MIN_SEGMENT_DURATION_S:
            # Too short — classify based on neighbors later
            segments.append(Segment(start, end, is_field=False))
            continue

        # Sample the middle of the segment
        mid = (start + end) / 2
        cap.set(cv2.CAP_PROP_POS_MSEC, mid * 1000)
        ret, frame = cap.read()
        if not ret:
            segments.append(Segment(start, end, is_field=False))
            continue

        # HSV green percentage on center crop
        h, w = frame.shape[:2]
        center = frame[h // 4 : 3 * h // 4, w // 5 : 4 * w // 5]
        hsv = cv2.cvtColor(center, cv2.COLOR_BGR2HSV)
        green_mask = cv2.inRange(hsv, (35, 30, 40), (85, 255, 255))
        green_pct = float(np.mean(green_mask > 0))

        segments.append(Segment(start, end, is_field=green_pct > GREEN_FIELD_THRESHOLD))

    cap.release()
    return segments


# ── Play grouping ──────────────────────────────────────────────────────────

def group_plays(segments: list[Segment]) -> list[PlayClip]:
    """
    Group segments into plays.

    With scoreboards: non-field segments are play separators.
    Each play = 2 consecutive field segments (sideline then endzone).

    Without scoreboards: field segments alternate SL/EZ.
    A new play starts at each pair.
    """
    field_segments = [s for s in segments if s.is_field and s.duration >= MIN_SEGMENT_DURATION_S]
    non_field_segments = [s for s in segments if not s.is_field and s.duration >= MIN_SEGMENT_DURATION_S]

    has_scoreboards = len(non_field_segments) > 0
    print(f"  {len(field_segments)} field segments, {len(non_field_segments)} non-field segments")
    print(f"  Scoreboard clips: {'yes' if has_scoreboards else 'no'}")

    if has_scoreboards:
        # Group field segments between non-field separators
        # Each group of 2 consecutive field segments = 1 play (SL + EZ)
        plays = []
        i = 0
        play_num = 1
        while i < len(field_segments) - 1:
            sl = field_segments[i]
            ez = field_segments[i + 1]

            # Verify they're from the same play: EZ should start right after SL
            # (no non-field segment between them)
            gap = ez.start_s - sl.end_s
            if gap < 1.0:
                # These are paired: SL then EZ within the same play
                plays.append(PlayClip(
                    play_num=play_num,
                    sideline_start=sl.start_s,
                    sideline_end=sl.end_s,
                    endzone_start=ez.start_s,
                    endzone_end=ez.end_s,
                ))
                play_num += 1
                i += 2
            else:
                # There's a gap (scoreboard) between them — sl is orphaned
                # This can happen if the SL→EZ cut wasn't detected
                print(f"  Warning: orphan field segment at {sl.start_s:.1f}-{sl.end_s:.1f}s, skipping")
                i += 1
    else:
        # No scoreboards — field segments alternate SL/EZ
        plays = []
        play_num = 1
        for i in range(0, len(field_segments) - 1, 2):
            sl = field_segments[i]
            ez = field_segments[i + 1]
            plays.append(PlayClip(
                play_num=play_num,
                sideline_start=sl.start_s,
                sideline_end=sl.end_s,
                endzone_start=ez.start_s,
                endzone_end=ez.end_s,
            ))
            play_num += 1

    return plays


# ── Clip extraction ────────────────────────────────────────────────────────

TRIM_INWARD_S = 0.4  # shave off boundary frames to avoid bleed


def extract_clip(video_path: str, start_s: float, end_s: float, output_path: str) -> None:
    """Extract a clip using re-encode for frame-accurate cuts, with inward trim."""
    start_s = start_s + TRIM_INWARD_S
    end_s = end_s - TRIM_INWARD_S
    duration = end_s - start_s
    if duration <= 0:
        return
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{start_s:.3f}",
        "-i", video_path,
        "-t", f"{duration:.3f}",
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-an",
        output_path,
    ]
    subprocess.run(cmd, capture_output=True, check=True)


def extract_plays(
    video_path: str, plays: list[PlayClip], output_dir: str, game_id: str
) -> dict:
    """Extract all play clips and write manifest."""
    game_dir = os.path.join(output_dir, game_id)
    os.makedirs(game_dir, exist_ok=True)

    manifest = {
        "game_id": game_id,
        "source_video": os.path.basename(video_path),
        "total_plays": len(plays),
        "plays": [],
    }

    for play in plays:
        play_dir = os.path.join(game_dir, f"play_{play.play_num:03d}")
        os.makedirs(play_dir, exist_ok=True)

        sl_dur = play.sideline_end - play.sideline_start
        ez_dur = play.endzone_end - play.endzone_start

        print(
            f"  Play {play.play_num:3d}: "
            f"SL {play.sideline_start:.1f}-{play.sideline_end:.1f}s ({sl_dur:.1f}s), "
            f"EZ {play.endzone_start:.1f}-{play.endzone_end:.1f}s ({ez_dur:.1f}s)"
        )

        sideline_path = os.path.join(play_dir, "sideline.mp4")
        endzone_path = os.path.join(play_dir, "endzone.mp4")

        extract_clip(video_path, play.sideline_start, play.sideline_end, sideline_path)
        extract_clip(video_path, play.endzone_start, play.endzone_end, endzone_path)

        manifest["plays"].append({
            "play_num": play.play_num,
            "sideline": {
                "file": "sideline.mp4",
                "start_s": round(play.sideline_start, 2),
                "end_s": round(play.sideline_end, 2),
                "duration_s": round(sl_dur, 2),
            },
            "endzone": {
                "file": "endzone.mp4",
                "start_s": round(play.endzone_start, 2),
                "end_s": round(play.endzone_end, 2),
                "duration_s": round(ez_dur, 2),
            },
        })

    manifest_path = os.path.join(game_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest: {manifest_path}")
    return manifest


# ── Preview ────────────────────────────────────────────────────────────────

def preview_timeline(
    cuts: list[float],
    segments: list[Segment],
    plays: list[PlayClip],
    duration_s: float,
    output_path: str,
) -> None:
    """Generate a timeline visualization."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping preview")
        return

    fig, ax = plt.subplots(figsize=(24, 3))

    # Draw segments
    for seg in segments:
        color = "#90EE90" if seg.is_field else "#D3D3D3"
        ax.axvspan(seg.start_s, seg.end_s, alpha=0.4, color=color)

    # Draw plays
    for play in plays:
        ax.axvspan(play.sideline_start, play.sideline_end, alpha=0.6, color="blue")
        ax.axvspan(play.endzone_start, play.endzone_end, alpha=0.6, color="orange")
        mid = (play.sideline_start + play.endzone_end) / 2
        ax.text(mid, 0.5, str(play.play_num), fontsize=5, ha="center", va="center")

    # Draw cut lines
    for c in cuts:
        ax.axvline(x=c, color="red", linewidth=0.3, alpha=0.5)

    ax.set_xlim(0, duration_s)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Time (s)")
    ax.set_title(f"Blue=Sideline, Orange=Endzone, Gray=Non-field ({len(plays)} plays)")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Preview saved: {output_path}")
    plt.close()


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Segment All-22 game video into per-play clips"
    )
    parser.add_argument("--video", required=True, help="Path to game MP4")
    parser.add_argument("--game-id", required=True, help="Game ID (e.g. 2019092204)")
    parser.add_argument(
        "--output", default="videos/clips/",
        help="Output directory (default: videos/clips/)",
    )
    parser.add_argument(
        "--preview", action="store_true",
        help="Generate timeline visualization only, no clip extraction",
    )
    args = parser.parse_args()

    video_path = os.path.abspath(args.video)
    if not os.path.exists(video_path):
        print(f"ERROR: Video not found: {video_path}")
        sys.exit(1)

    print(f"Processing: {os.path.basename(video_path)}")
    print(f"Game ID:    {args.game_id}\n")

    # Step 1: Detect hard cuts
    cuts = detect_cuts(video_path)

    # Get video duration
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration_s = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
    cap.release()

    # Step 2: Classify segments between cuts
    print("\nClassifying segments...")
    segments = classify_segments(video_path, cuts, duration_s)

    # Step 3: Group into plays
    print("\nGrouping plays...")
    plays = group_plays(segments)
    print(f"  Found {len(plays)} plays")

    if not plays:
        print("No plays found!")
        sys.exit(1)

    # Summary
    sl_durs = [p.sideline_end - p.sideline_start for p in plays]
    ez_durs = [p.endzone_end - p.endzone_start for p in plays]
    print(f"\n  Sideline: {np.mean(sl_durs):.1f}s avg ({np.min(sl_durs):.1f}-{np.max(sl_durs):.1f}s)")
    print(f"  Endzone:  {np.mean(ez_durs):.1f}s avg ({np.min(ez_durs):.1f}-{np.max(ez_durs):.1f}s)")

    if args.preview:
        os.makedirs(args.output, exist_ok=True)
        preview_path = os.path.join(args.output, f"{args.game_id}_timeline.png")
        preview_timeline(cuts, segments, plays, duration_s, preview_path)
        return

    # Step 4: Extract clips
    print(f"\nExtracting clips to {args.output}...")
    extract_plays(video_path, plays, args.output, args.game_id)
    print(f"\nDone! {len(plays)} plays → {os.path.join(args.output, args.game_id)}/")


if __name__ == "__main__":
    main()
