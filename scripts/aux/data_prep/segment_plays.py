#!/usr/bin/env python3
"""
Segment an All-22 game video into per-play clips (sideline + endzone).

Each play in All-22 film follows the pattern:
  [scoreboard] → sideline wide → endzone tight → [scoreboard] → ...

Detection uses a YOLO-cls view classifier trained on 2019 game frames:
  1. Classify 1 frame/second as sideline, endzone, or scoreboard.
  2. Find transitions where the confident label changes.
  3. Group consecutive sideline+endzone segments into plays.
  4. Extract clips using ffmpeg stream copy with boundary trims.

Usage:
  python scripts/segment_plays.py --video <path> --game-id <id> [--output videos/clips/]
  python scripts/segment_plays.py --video <path> --game-id <id> --preview
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

import cv2
import numpy as np

# Log file for detached runs
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output", "logs")
PID_FILE = os.path.join(LOG_DIR, "segment_plays.pid")


# ── Configuration ──────────────────────────────────────────────────────────

VIEW_MODEL_PATH = "models/view_classifier.pt"
CONFIDENCE_THRESHOLD = 0.7    # below this, carry forward previous label
SMOOTHING_WINDOW = 3          # majority vote window (frames at 1fps = seconds)
MIN_SEGMENT_DURATION_S = 3.0  # merge segments shorter than this into neighbor

START_TRIM_S = 1.0            # trim off start of all clips
END_TRIM_S = 2.0              # trim off end of all clips


# ── Data structures ────────────────────────────────────────────────────────

@dataclass
class Segment:
    start_s: float
    end_s: float
    view_type: str  # "sideline", "endzone", or "scoreboard"

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


# ── View classification ───────────────────────────────────────────────────

def classify_frames(video_path: str, model_path: str) -> list[tuple[int, str, float]]:
    """Classify 1 frame/second using YOLO-cls view classifier.

    Reads sequentially and skips frames for speed (avoids slow seeking).
    Returns list of (second, class_name, confidence).
    """
    from ultralytics import YOLO
    model = YOLO(model_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Cannot open {video_path}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_s = total_frames / fps
    step = round(fps)  # 1 frame per second

    print(f"Video: {duration_s:.0f}s ({duration_s/60:.1f} min), {fps:.2f} fps")
    print(f"Classifying 1 frame/second ({int(duration_s)} frames)...")

    classifications = []
    frame_num = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_num % step == 0:
            sec = round(frame_num / fps)
            small = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
            result = model(small, verbose=False)[0]
            pred = result.names[result.probs.top1]
            conf = float(result.probs.top1conf)
            classifications.append((sec, pred, conf))

            if sec % 500 == 0 and sec > 0:
                print(f"  {sec}/{int(duration_s)}s...")

        frame_num += 1

    cap.release()
    print(f"  Classified {len(classifications)} frames")
    return classifications


# ── Segment detection ─────────────────────────────────────────────────────

def find_segments(classifications: list[tuple[int, str, float]],
                  duration_s: float) -> list[Segment]:
    """Find view segments by detecting transitions in confident classifications.

    Low-confidence frames carry forward the previous label.
    Majority-vote smoothing prevents single-frame glitches from triggering cuts.
    Short segments are merged into neighbors.
    """
    if not classifications:
        return []

    # Step 1: Confidence gating — low-confidence frames inherit previous label
    gated = []
    prev_label = classifications[0][1]
    for sec, pred, conf in classifications:
        if conf >= CONFIDENCE_THRESHOLD:
            prev_label = pred
        gated.append(prev_label)

    # Step 2: Majority-vote smoothing over SMOOTHING_WINDOW frames
    W = SMOOTHING_WINDOW
    smoothed_labels = []
    for i in range(len(gated)):
        window = gated[max(0, i - W // 2):i + W // 2 + 1]
        most_common = Counter(window).most_common(1)[0][0]
        smoothed_labels.append(most_common)

    labels = [(classifications[i][0], smoothed_labels[i]) for i in range(len(classifications))]

    # Step 3: Find runs of same label → segments
    segments = []
    run_start = labels[0][0]
    run_label = labels[0][1]

    for i in range(1, len(labels)):
        sec, label = labels[i]
        if label != run_label:
            # End of run
            segments.append(Segment(
                start_s=float(run_start),
                end_s=float(sec),
                view_type=run_label,
            ))
            run_start = sec
            run_label = label

    # Final segment
    segments.append(Segment(
        start_s=float(run_start),
        end_s=duration_s,
        view_type=run_label,
    ))

    # Filter short segments
    segments = [s for s in segments if s.duration >= MIN_SEGMENT_DURATION_S]

    return segments


# ── Play grouping ──────────────────────────────────────────────────────────

def group_plays(segments: list[Segment]) -> list[PlayClip]:
    """Group segments into plays.

    Each play = consecutive sideline + endzone pair.
    Scoreboard segments are skipped (play separators).
    """
    field_segments = [s for s in segments if s.view_type in ("sideline", "endzone")]
    scoreboard_segments = [s for s in segments if s.view_type == "scoreboard"]

    print(f"  {len(field_segments)} field segments "
          f"({sum(1 for s in field_segments if s.view_type == 'sideline')} SL, "
          f"{sum(1 for s in field_segments if s.view_type == 'endzone')} EZ)")
    if scoreboard_segments:
        print(f"  {len(scoreboard_segments)} scoreboard segments")

    plays = []
    play_num = 1
    i = 0

    while i < len(field_segments) - 1:
        sl = field_segments[i]
        ez = field_segments[i + 1]

        if sl.view_type == "sideline" and ez.view_type == "endzone":
            # Valid SL→EZ pair
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
            # Orphan segment (e.g. two sidelines in a row) — skip it
            print(f"  Warning: orphan {sl.view_type} segment at "
                  f"{sl.start_s:.1f}-{sl.end_s:.1f}s, skipping")
            i += 1

    # Handle last segment if unpaired
    if i == len(field_segments) - 1:
        last = field_segments[i]
        if last.view_type == "sideline":
            # Final play with sideline only (end of game)
            plays.append(PlayClip(
                play_num=play_num,
                sideline_start=last.start_s,
                sideline_end=last.end_s,
                endzone_start=0,
                endzone_end=0,
            ))

    return plays


# ── Clip extraction ────────────────────────────────────────────────────────

TARGET_FPS = 30  # Normalize all clips to 30fps (2024 games are 60fps)


def extract_clip(video_path: str, start_s: float, duration_s: float,
                 output_path: str) -> None:
    """Extract a clip, downsampling to 30fps if needed.

    Uses stream copy when source is already 30fps, re-encodes only when
    the source is higher (e.g. 60fps from 2024 games).
    """
    if duration_s <= 0:
        return
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{start_s:.3f}",
        "-i", video_path,
        "-t", f"{duration_s:.3f}",
        "-r", str(TARGET_FPS),
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-an",
        output_path,
    ]
    subprocess.run(cmd, capture_output=True, check=True)


def extract_plays(
    video_path: str, plays: list[PlayClip], output_dir: str, game_id: str,
    workers: int = 4,
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

    tasks = []
    for play in plays:
        play_dir = os.path.join(game_dir, f"play_{play.play_num:03d}")
        os.makedirs(play_dir, exist_ok=True)

        # Sideline clip
        if play.sideline_start > 0 or play.sideline_end > 0:
            sl_dur = play.sideline_end - play.sideline_start
            sl_start = play.sideline_start + START_TRIM_S
            sl_duration = sl_dur - START_TRIM_S - END_TRIM_S

            if sl_duration > 1.0:
                tasks.append((video_path, sl_start, sl_duration,
                              os.path.join(play_dir, "sideline.mp4")))

        # Endzone clip
        if play.endzone_end > 0:
            ez_dur = play.endzone_end - play.endzone_start
            ez_start = play.endzone_start + START_TRIM_S
            ez_duration = ez_dur - START_TRIM_S - END_TRIM_S

            if ez_duration > 1.0:
                tasks.append((video_path, ez_start, ez_duration,
                              os.path.join(play_dir, "endzone.mp4")))

        sl_dur_raw = play.sideline_end - play.sideline_start
        ez_dur_raw = play.endzone_end - play.endzone_start if play.endzone_end > 0 else 0

        manifest["plays"].append({
            "play_num": play.play_num,
            "sideline": {
                "file": "sideline.mp4",
                "start_s": round(play.sideline_start, 2),
                "end_s": round(play.sideline_end, 2),
                "duration_s": round(sl_dur_raw, 2),
            },
            "endzone": {
                "file": "endzone.mp4",
                "start_s": round(play.endzone_start, 2),
                "end_s": round(play.endzone_end, 2),
                "duration_s": round(ez_dur_raw, 2),
            } if play.endzone_end > 0 else None,
        })

    print(f"\n  Extracting {len(tasks)} clips ({workers} workers, stream copy)...")
    completed = 0
    errors = 0

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(extract_clip, *t): t for t in tasks}
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                errors += 1
                task = futures[future]
                print(f"  Error extracting {task[3]}: {e}")
            completed += 1
            if completed % 50 == 0:
                print(f"  {completed}/{len(tasks)} clips extracted...")

    print(f"  {completed - errors}/{len(tasks)} clips extracted successfully")
    if errors:
        print(f"  {errors} errors")

    manifest_path = os.path.join(game_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest: {manifest_path}")
    return manifest


# ── Preview ────────────────────────────────────────────────────────────────

def preview_timeline(
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

    colors = {"sideline": "blue", "endzone": "orange", "scoreboard": "#D3D3D3"}
    for seg in segments:
        ax.axvspan(seg.start_s, seg.end_s, alpha=0.5,
                    color=colors.get(seg.view_type, "gray"))

    for play in plays:
        mid = (play.sideline_start + (play.endzone_end or play.sideline_end)) / 2
        ax.text(mid, 0.5, str(play.play_num), fontsize=5, ha="center", va="center")

    ax.set_xlim(0, duration_s)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Time (s)")
    ax.set_title(f"Blue=Sideline, Orange=Endzone, Gray=Scoreboard ({len(plays)} plays)")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Preview saved: {output_path}")
    plt.close()


# ── Detached launch / check ───────────────────────────────────────────────

def log_path_for_game(game_id: str) -> str:
    return os.path.join(LOG_DIR, f"segment_{game_id}.log")


def launch_detached(video_path: str, game_id: str, output: str, model: str,
                    workers: int) -> None:
    """Launch segmentation as a detached background process with log file."""
    os.makedirs(LOG_DIR, exist_ok=True)
    log_file = log_path_for_game(game_id)

    script_path = os.path.abspath(__file__)
    # Use the same Python interpreter that's running this script (respects venv)
    python_path = sys.executable

    # Build command as list to avoid shell quoting issues with spaces in paths
    cmd_args = [
        python_path, "-u", script_path,
        "--video", video_path,
        "--game-id", game_id,
        "--output", output,
        "--model", model,
        "--workers", str(workers),
    ]

    # Launch with nohup via Popen, redirect stdout/stderr to log file
    with open(log_file, "w") as log_f:
        proc = subprocess.Popen(
            cmd_args,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            start_new_session=True,  # detach from parent session (like nohup)
        )

    # Save PID for checking later
    with open(PID_FILE, "a") as f:
        f.write(f"{game_id}:{proc.pid}\n")

    print(f"Launched segmentation for {game_id} (PID {proc.pid})")
    print(f"  Log: {log_file}")


def check_segmentation() -> None:
    """Check progress of running segmentation jobs."""
    if not os.path.exists(PID_FILE):
        print("No segmentation jobs found.")
        return

    with open(PID_FILE) as f:
        jobs = [line.strip().split(":", 1) for line in f if line.strip()]

    for game_id, pid in jobs:
        log_file = log_path_for_game(game_id)

        # Check if process is still running
        try:
            os.kill(int(pid), 0)
            running = True
        except (OSError, ValueError):
            running = False

        # Check log for completion
        done = False
        last_lines = ""
        if os.path.exists(log_file):
            with open(log_file) as f:
                content = f.read()
            done = "Done!" in content
            lines = content.strip().split("\n")
            last_lines = "\n".join(lines[-5:])

        status = "RUNNING" if running else ("COMPLETED" if done else "STOPPED/FAILED")
        print(f"[{game_id}] {status} (PID {pid})")
        print(f"  Log: {log_file}")
        if last_lines:
            for line in last_lines.split("\n"):
                print(f"    {line}")
        print()

    # Clean up PID file if all jobs are done
    all_done = True
    for game_id, pid in jobs:
        try:
            os.kill(int(pid), 0)
            all_done = False
        except (OSError, ValueError):
            pass
    if all_done:
        os.remove(PID_FILE)


# ── Main ───────────────────────────────────────────────────────────────────

def run_segmentation(args):
    """Run segmentation directly (called either interactively or from nohup)."""
    video_path = os.path.abspath(args.video)
    if not os.path.exists(video_path):
        print(f"ERROR: Video not found: {video_path}")
        sys.exit(1)

    model_path = args.model
    if not os.path.exists(model_path):
        print(f"ERROR: View classifier model not found: {model_path}")
        print("Train with: data/labels/view_classifier/ training data")
        sys.exit(1)

    print(f"Processing: {os.path.basename(video_path)}")
    print(f"Game ID:    {args.game_id}\n")

    # Step 1: Classify frames
    classifications = classify_frames(video_path, model_path)

    # Get duration
    cap = cv2.VideoCapture(video_path)
    duration_s = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    # Step 2: Find segments from transitions
    print("\nFinding view segments...")
    segments = find_segments(classifications, duration_s)

    # Step 3: Group into plays
    print("\nGrouping plays...")
    plays = group_plays(segments)
    print(f"  Found {len(plays)} plays")

    if not plays:
        print("No plays found!")
        sys.exit(1)

    # Summary
    sl_durs = [p.sideline_end - p.sideline_start for p in plays]
    ez_durs = [p.endzone_end - p.endzone_start for p in plays if p.endzone_end > 0]
    print(f"\n  Sideline: {np.mean(sl_durs):.1f}s avg "
          f"({np.min(sl_durs):.1f}-{np.max(sl_durs):.1f}s)")
    if ez_durs:
        print(f"  Endzone:  {np.mean(ez_durs):.1f}s avg "
              f"({np.min(ez_durs):.1f}-{np.max(ez_durs):.1f}s)")

    if args.preview:
        os.makedirs(args.output, exist_ok=True)
        preview_path = os.path.join(args.output, f"{args.game_id}_timeline.png")
        preview_timeline(segments, plays, duration_s, preview_path)
        return

    # Step 4: Extract clips
    print(f"\nExtracting clips to {args.output}...")
    extract_plays(video_path, plays, args.output, args.game_id,
                  workers=args.workers)
    print(f"\nDone! {len(plays)} plays → {os.path.join(args.output, args.game_id)}/")


def main():
    parser = argparse.ArgumentParser(
        description="Segment All-22 game video into per-play clips"
    )
    parser.add_argument("--video", help="Path to game MP4")
    parser.add_argument("--game-id", help="Game ID (e.g. 2019092204)")
    parser.add_argument(
        "--output", default="videos/clips/",
        help="Output directory (default: videos/clips/)",
    )
    parser.add_argument(
        "--model", default=VIEW_MODEL_PATH,
        help=f"Path to view classifier weights (default: {VIEW_MODEL_PATH})",
    )
    parser.add_argument(
        "--preview", action="store_true",
        help="Generate timeline visualization only, no clip extraction",
    )
    parser.add_argument(
        "--workers", type=int, default=4,
        help="Parallel ffmpeg workers for extraction (default: 4)",
    )
    parser.add_argument(
        "--detach", action="store_true",
        help="Launch in background with nohup (survives session disconnect)",
    )
    parser.add_argument(
        "--check", action="store_true",
        help="Check progress of running segmentation jobs",
    )
    args = parser.parse_args()

    if args.check:
        check_segmentation()
        return

    if not args.video or not args.game_id:
        parser.error("--video and --game-id are required (unless using --check)")

    if args.detach:
        launch_detached(
            video_path=os.path.abspath(args.video),
            game_id=args.game_id,
            output=args.output,
            model=args.model,
            workers=args.workers,
        )
    else:
        run_segmentation(args)


if __name__ == "__main__":
    main()
