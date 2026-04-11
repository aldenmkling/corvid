#!/usr/bin/env python3
"""
Run detection inference on RunPod and download the result.

Creates a cheap GPU pod, uploads a clip + weights + inference script,
runs detection, downloads the annotated video, and terminates.

Usage:
  python scripts/runpod_infer.py --clip videos/clips/2019092204/play_050/sideline.mp4
  python scripts/runpod_infer.py --check     # check progress
  python scripts/runpod_infer.py --download   # grab result + terminate
"""

import argparse
import json
import os
import subprocess
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from scripts.launch_runpod import (
    init_runpod, save_pod_state, load_pod_state,
    get_ssh_info, ssh_is_reachable, wait_for_pod, terminate_pod,
)

STATE_FILE = os.path.join(PROJECT_ROOT, ".runpod_infer.json")
LOG_DIR = os.path.join(PROJECT_ROOT, "output", "logs")


def save_infer_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def load_infer_state():
    if not os.path.exists(STATE_FILE):
        return None
    with open(STATE_FILE) as f:
        return json.load(f)


def run_inference(args):
    rp = init_runpod()

    clip_path = os.path.abspath(args.clip)
    weights_path = os.path.join(PROJECT_ROOT, "models", "rfdetr_best_ema.pth")
    script_path = os.path.join(PROJECT_ROOT, "scripts", "infer_viz.py")
    reqs_path = os.path.join(PROJECT_ROOT, "scripts", "requirements_runpod.txt")

    for path, name in [(clip_path, "clip"), (weights_path, "weights"), (script_path, "script")]:
        if not os.path.exists(path):
            print(f"ERROR: {name} not found: {path}")
            sys.exit(1)

    clip_name = os.path.basename(clip_path)
    output_name = os.path.splitext(clip_name)[0] + "_detected.mp4"

    # Create pod (use cheaper GPU for inference — RTX 4090 is fine)
    print(f"Creating RunPod pod for inference...")
    gpu_type = args.gpu_type
    print(f"  GPU: {gpu_type}")

    pod = rp.create_pod(
        name="all22-rfdetr-inference",
        image_name="runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04",
        gpu_type_id=gpu_type,
        gpu_count=1,
        volume_in_gb=0,
        container_disk_in_gb=20,
        ports="22/tcp",
        start_ssh=True,
    )

    pod_id = pod["id"]
    print(f"Pod created: {pod_id}")

    # Save to BOTH state files so --terminate works from launch_runpod.py too
    save_pod_state({"id": pod_id, "gpu": gpu_type})

    info = wait_for_pod(rp, pod_id, gpu_type, timeout_sec=300, container_timeout_sec=120)
    if not info:
        # Container stuck — terminate and retry once on a different machine
        print("  Terminating stuck pod and retrying...")
        rp.terminate_pod(pod_id)
        pod = rp.create_pod(
            name="all22-rfdetr-inference",
            image_name="runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04",
            gpu_type_id=gpu_type,
            gpu_count=1,
            volume_in_gb=0,
            container_disk_in_gb=20,
            ports="22/tcp",
            start_ssh=True,
        )
        pod_id = pod["id"]
        print(f"  Retry pod created: {pod_id}")
        save_pod_state({"id": pod_id, "gpu": gpu_type})
        info = wait_for_pod(rp, pod_id, gpu_type, timeout_sec=300, container_timeout_sec=120)
        if not info:
            print("  Second attempt also failed. Terminating.")
            rp.terminate_pod(pod_id)
            return

    state = load_pod_state()
    ssh_host = state["ssh_host"]
    ssh_port = state["ssh_port"]
    ssh_target = f"root@{ssh_host}"
    ssh_opts = ["-o", "StrictHostKeyChecking=no", "-p", str(ssh_port)]
    scp_opts = ["-o", "StrictHostKeyChecking=no", "-P", str(ssh_port)]

    # Save inference state
    save_infer_state({
        "pod_id": pod_id,
        "ssh_host": ssh_host,
        "ssh_port": ssh_port,
        "clip": clip_path,
        "output_name": output_name,
    })

    # Upload files
    print(f"\nUploading files...")
    print(f"  Script: infer_viz.py")
    subprocess.run(
        ["scp"] + scp_opts + [script_path, reqs_path, f"{ssh_target}:/workspace/"],
        check=True,
    )
    print(f"  Clip: {clip_name}")
    subprocess.run(
        ["scp"] + scp_opts + [clip_path, f"{ssh_target}:/workspace/{clip_name}"],
        check=True,
    )
    print(f"  Weights: rfdetr_best_ema.pth (128MB)")
    subprocess.run(
        ["scp"] + scp_opts + [weights_path, f"{ssh_target}:/workspace/rfdetr_best_ema.pth"],
        check=True,
    )

    # Install deps
    print(f"\nInstalling dependencies...")
    subprocess.run(
        ["ssh"] + ssh_opts + [ssh_target,
         "cd /workspace && python -m venv venv --system-site-packages"
         " && source venv/bin/activate"
         " && pip install -q rfdetr opencv-python-headless"],
        check=True,
    )

    # Launch inference detached
    infer_cmd = (
        f"cd /workspace && source venv/bin/activate && "
        f"python -u infer_viz.py "
        f"--weights rfdetr_best_ema.pth "
        f"--video {clip_name} "
        f"--output {output_name}"
    )
    detached_cmd = f"nohup bash -c '{infer_cmd}' > /workspace/infer.log 2>&1 &"

    print(f"\nLaunching inference (detached)...")
    subprocess.run(["ssh"] + ssh_opts + [ssh_target, detached_cmd], check=True)

    time.sleep(2)
    result = subprocess.run(
        ["ssh"] + ssh_opts + [ssh_target, "pgrep -f infer_viz.py"],
        capture_output=True, text=True,
    )
    if result.returncode == 0:
        print(f"✓ Inference running (PID {result.stdout.strip().split()[0]})")
        print(f"\nNext steps:")
        print(f"  python scripts/runpod_infer.py --check      # monitor progress")
        print(f"  python scripts/runpod_infer.py --download    # grab result + terminate")
    else:
        print(f"⚠ Inference process not found. Check log:")
        log = subprocess.run(
            ["ssh"] + ssh_opts + [ssh_target, "cat /workspace/infer.log"],
            capture_output=True, text=True,
        )
        print(log.stdout[-500:] if log.stdout else "(empty log)")


def check_progress():
    state = load_infer_state()
    if not state:
        print("No inference job found.")
        return

    ssh_host = state["ssh_host"]
    ssh_port = state["ssh_port"]
    ssh_target = f"root@{ssh_host}"
    ssh_opts = ["-o", "StrictHostKeyChecking=no", "-p", str(ssh_port)]

    # Check if running
    proc = subprocess.run(
        ["ssh"] + ssh_opts + [ssh_target, "pgrep -f infer_viz.py"],
        capture_output=True, text=True,
    )
    running = proc.returncode == 0

    # Get log
    log = subprocess.run(
        ["ssh"] + ssh_opts + [ssh_target, "tail -10 /workspace/infer.log"],
        capture_output=True, text=True,
    )

    status = "RUNNING" if running else "FINISHED"
    print(f"Inference: {status}")
    print(f"--- Log ---")
    print(log.stdout if log.stdout else "(no output)")


def download_result():
    state = load_infer_state()
    if not state:
        print("No inference job found.")
        return

    ssh_host = state["ssh_host"]
    ssh_port = state["ssh_port"]
    output_name = state["output_name"]
    ssh_target = f"root@{ssh_host}"
    scp_opts = ["-o", "StrictHostKeyChecking=no", "-P", str(ssh_port)]

    local_output = os.path.join(PROJECT_ROOT, "output", output_name)
    os.makedirs(os.path.dirname(local_output), exist_ok=True)

    print(f"Downloading {output_name}...")
    subprocess.run(
        ["scp"] + scp_opts + [f"{ssh_target}:/workspace/{output_name}", local_output],
        check=True,
    )
    print(f"Saved: {local_output}")

    # Also grab the log
    log_path = os.path.join(LOG_DIR, "infer_viz.log")
    os.makedirs(LOG_DIR, exist_ok=True)
    subprocess.run(
        ["scp"] + scp_opts + [f"{ssh_target}:/workspace/infer.log", log_path],
        capture_output=True,
    )

    print(f"\nTerminating pod...")
    terminate_pod()

    # Clean up state
    os.remove(STATE_FILE)
    print(f"\nDone! Open: {local_output}")


def main():
    parser = argparse.ArgumentParser(description="Run RF-DETR inference on RunPod")
    parser.add_argument("--clip", help="Path to video clip")
    parser.add_argument("--check", action="store_true", help="Check inference progress")
    parser.add_argument("--download", action="store_true", help="Download result and terminate")
    parser.add_argument("--gpu-type", default="NVIDIA GeForce RTX 5090",
                        help="GPU type (default: RTX 5090)")
    args = parser.parse_args()

    if args.check:
        check_progress()
    elif args.download:
        download_result()
    elif args.clip:
        run_inference(args)
    else:
        parser.error("Provide --clip, --check, or --download")


if __name__ == "__main__":
    main()
