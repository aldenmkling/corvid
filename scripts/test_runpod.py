#!/usr/bin/env python3
"""
Test RunPod integration by running YOLO inference on a single play clip.

Creates a pod, uploads a clip + model, runs detection on every frame,
draws boxes, saves the annotated video, downloads it, and terminates the pod.

Usage:
  python scripts/test_runpod.py
  python scripts/test_runpod.py --clip videos/clips/2024090802/play_050/sideline.mp4
  python scripts/test_runpod.py --gpu-type "NVIDIA RTX A4000"  # cheaper GPU for testing
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_FILE = os.path.join(PROJECT_ROOT, ".env")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")

# Inference script that runs on the pod
INFERENCE_SCRIPT = '''
#!/usr/bin/env python3
"""Run YOLOv12x inference on a video clip and save annotated output."""
import cv2
import sys
import time
from ultralytics import YOLO

model_path = sys.argv[1]
input_path = sys.argv[2]
output_path = sys.argv[3]
threshold = float(sys.argv[4]) if len(sys.argv) > 4 else 0.5

print(f"Loading YOLO model: {model_path}")
model = YOLO(model_path)

cap = cv2.VideoCapture(input_path)
fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

print(f"Processing {total} frames at {fps:.1f}fps ({w}x{h})...")
t0 = time.time()
frame_num = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, imgsz=1280, device="cuda", verbose=False)[0]
    boxes = results.boxes
    mask = boxes.conf >= threshold
    boxes = boxes[mask]

    # Draw boxes
    for box, conf in zip(boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy()):
        x1, y1, x2, y2 = box.astype(int)
        color = (0, 255, 0) if conf >= 0.7 else (0, 165, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{conf:.2f}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    # Add frame counter
    n_det = len(boxes)
    cv2.putText(frame, f"Frame {frame_num}/{total} | {n_det} detections",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    out.write(frame)
    frame_num += 1

    if frame_num % 30 == 0:
        elapsed = time.time() - t0
        fps_actual = frame_num / elapsed
        print(f"  [{frame_num}/{total}] {fps_actual:.1f} fps")

cap.release()
out.release()

elapsed = time.time() - t0
print(f"\\nDone! {frame_num} frames in {elapsed:.1f}s ({frame_num/elapsed:.1f} fps)")
print(f"Output: {output_path}")
'''


def load_api_key():
    from dotenv import dotenv_values
    config = dotenv_values(ENV_FILE)
    key = config.get("RUNPOD_API_KEY", "")
    if not key or key == "paste_your_key_here":
        print("ERROR: Set your RunPod API key in .env")
        sys.exit(1)
    return key


def ssh_cmd(host, port, user="root"):
    return ["ssh", "-o", "StrictHostKeyChecking=no", "-p", str(port), f"{user}@{host}"]


def scp_cmd(host, port, user="root"):
    return ["scp", "-o", "StrictHostKeyChecking=no", "-P", str(port)]


def elapsed_since(t0):
    return f"{time.time() - t0:.1f}s"


def main():
    parser = argparse.ArgumentParser(description="Test RunPod with YOLO inference")
    parser.add_argument("--clip", default=None,
                        help="Path to clip (default: auto-picks a sideline clip)")
    parser.add_argument("--gpu-type", default="NVIDIA GeForce RTX 4090",
                        help="GPU type (default: RTX 4090)")
    parser.add_argument("--threshold", type=float, default=0.3,
                        help="Detection confidence threshold (default: 0.3)")
    args = parser.parse_args()

    # Find a clip if not specified
    if args.clip is None:
        clips_dir = os.path.join(PROJECT_ROOT, "videos", "clips", "2024090802")
        if os.path.exists(clips_dir):
            plays = sorted([d for d in os.listdir(clips_dir) if d.startswith("play_")])
            if plays:
                mid_play = plays[len(plays) // 2]
                args.clip = os.path.join(clips_dir, mid_play, "sideline.mp4")

    if not args.clip or not os.path.exists(args.clip):
        print(f"ERROR: No clip found. Specify with --clip")
        sys.exit(1)

    clip_size_mb = os.path.getsize(args.clip) / 1e6
    model_path = os.path.join(PROJECT_ROOT, "models", "best.pt")
    model_size_mb = os.path.getsize(model_path) / 1e6 if os.path.exists(model_path) else 0

    clip_name = os.path.basename(os.path.dirname(args.clip))
    print(f"{'='*60}")
    print(f"  RunPod Integration Test — YOLO Inference")
    print(f"{'='*60}")
    print(f"  Clip:       {args.clip}")
    print(f"  Clip size:  {clip_size_mb:.1f} MB")
    print(f"  Model:      {model_path}")
    print(f"  Model size: {model_size_mb:.1f} MB")
    print(f"  GPU:        {args.gpu_type}")
    print(f"  Threshold:  {args.threshold}")
    print(f"{'='*60}\n")

    pipeline_start = time.time()

    # Step 1: Create pod
    import runpod
    runpod.api_key = load_api_key()

    t0 = time.time()
    print("[1/6] Creating pod...")
    try:
        pod = runpod.create_pod(
            name="all22-test-inference",
            image_name="runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04",
            gpu_type_id=args.gpu_type,
            gpu_count=1,
            container_disk_in_gb=20,
            ports="22/tcp",
            start_ssh=True,
        )
    except Exception as e:
        print(f"  ERROR creating pod: {e}")
        sys.exit(1)

    pod_id = pod["id"]
    print(f"  Pod ID: {pod_id}")
    print(f"  API response: {json.dumps(pod, indent=2)}")
    print(f"  ⏱ Create call took {elapsed_since(t0)}")

    # Step 2: Wait for ready
    t0 = time.time()
    print("\n[2/6] Waiting for pod to start...")
    ssh_host = ssh_port = None
    for i in range(120):  # 10 min timeout
        time.sleep(5)
        try:
            info = runpod.get_pod(pod_id)
        except Exception as e:
            print(f"\n  ERROR polling pod: {e}")
            continue

        status = info.get("desiredStatus", "?")
        runtime = info.get("runtime")
        machine = info.get("machine", {})
        gpu_name = machine.get("gpuDisplayName", "?")

        if runtime:
            uptime = runtime.get("uptimeInSeconds", 0)
            ports = runtime.get("ports", [])
            gpu_util = runtime.get("gpus", [{}])

            # Look for SSH port
            for p in ports:
                if p.get("privatePort") == 22:
                    ssh_host = p.get("ip")
                    ssh_port = p.get("publicPort")

            if ssh_host and uptime:
                print(f"\n  ✓ Pod ready!")
                print(f"    GPU: {gpu_name}")
                print(f"    SSH: root@{ssh_host}:{ssh_port}")
                print(f"    Uptime: {uptime}s")
                print(f"    Ports: {json.dumps(ports)}")
                print(f"  ⏱ Startup took {elapsed_since(t0)}")
                break

            # Show runtime info even if not fully ready
            sys.stdout.write(f"\r  Status: {status} | GPU: {gpu_name} | uptime: {uptime}s | ports: {len(ports)} | ({(i+1)*5}s elapsed)")
            sys.stdout.flush()
        else:
            sys.stdout.write(f"\r  Status: {status} | GPU: {gpu_name} | no runtime yet | ({(i+1)*5}s elapsed)")
            sys.stdout.flush()

    if not ssh_host:
        print(f"\n  ERROR: Pod didn't start after {(i+1)*5}s")
        print(f"  Last pod state: {json.dumps(info, indent=2, default=str)}")
        print("  Terminating...")
        runpod.terminate_pod(pod_id)
        sys.exit(1)

    # Wait for SSH daemon to be fully ready
    t0 = time.time()
    print(f"\n  Waiting for SSH to accept connections...")
    for attempt in range(12):
        time.sleep(5)
        result = subprocess.run(
            ssh_cmd(ssh_host, ssh_port) + ["echo 'SSH OK'"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            print(f"  ✓ SSH connected ({elapsed_since(t0)})")
            break
        sys.stdout.write(f"\r  SSH attempt {attempt+1}/12...")
        sys.stdout.flush()
    else:
        print(f"\n  ERROR: SSH never connected. Terminating...")
        runpod.terminate_pod(pod_id)
        sys.exit(1)

    target = f"root@{ssh_host}"
    ssh = ssh_cmd(ssh_host, ssh_port)
    scp = scp_cmd(ssh_host, ssh_port)

    try:
        # Step 3: Install ultralytics
        t0 = time.time()
        print(f"\n[3/6] Installing ultralytics...")
        result = subprocess.run(
            ssh + ["pip install ultralytics==8.4.31 -q 2>&1 | tail -3"],
            capture_output=True, text=True,
        )
        print(f"  {result.stdout.strip()}")
        if result.returncode != 0:
            print(f"  STDERR: {result.stderr}")
        print(f"  ⏱ Install took {elapsed_since(t0)}")

        # Step 4: Upload clip, model, and inference script
        t0 = time.time()
        print(f"\n[4/6] Uploading files...")

        # Write inference script to temp file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(INFERENCE_SCRIPT)
            infer_script_path = f.name

        if not os.path.exists(model_path):
            print(f"  ERROR: Model not found at {model_path}")
            raise FileNotFoundError(model_path)

        for label, local, remote in [
            ("clip", args.clip, "/workspace/input.mp4"),
            ("model", model_path, "/workspace/best.pt"),
            ("script", infer_script_path, "/workspace/infer.py"),
        ]:
            t1 = time.time()
            size = os.path.getsize(local) / 1e6
            subprocess.run(scp + [local, f"{target}:{remote}"], check=True)
            speed = size / (time.time() - t1) if time.time() - t1 > 0 else 0
            print(f"  ✓ {label}: {size:.1f}MB ({speed:.1f} MB/s)")

        os.unlink(infer_script_path)
        print(f"  ⏱ Upload took {elapsed_since(t0)}")

        # Step 5: Run inference
        t0 = time.time()
        print(f"\n[5/6] Running inference...\n")
        subprocess.run(
            ssh + [f"cd /workspace && python infer.py best.pt input.mp4 output.mp4 {args.threshold}"],
        )
        print(f"\n  ⏱ Inference took {elapsed_since(t0)}")

        # Step 6: Download result
        t0 = time.time()
        print(f"\n[6/6] Downloading annotated video...")
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        output_file = os.path.join(OUTPUT_DIR, f"yolo_test_{clip_name}.mp4")

        # Check if output exists on pod
        result = subprocess.run(
            ssh + ["ls -lh /workspace/output.mp4"],
            capture_output=True, text=True,
        )
        print(f"  Remote file: {result.stdout.strip()}")

        subprocess.run(
            scp + [f"{target}:/workspace/output.mp4", output_file],
            check=True,
        )
        local_size = os.path.getsize(output_file) / 1e6
        print(f"  ✓ Downloaded {local_size:.1f}MB → {output_file}")
        print(f"  ⏱ Download took {elapsed_since(t0)}")

    finally:
        # Always terminate
        t0 = time.time()
        print(f"\nTerminating pod {pod_id}...")
        runpod.terminate_pod(pod_id)
        print(f"  ⏱ Terminate took {elapsed_since(t0)}")

    total_time = time.time() - pipeline_start
    print(f"\n{'='*60}")
    print(f"  Test complete!")
    print(f"  Total pipeline time: {total_time:.0f}s ({total_time/60:.1f} min)")
    print(f"  Output: {output_file}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
