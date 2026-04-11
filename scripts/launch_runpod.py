#!/usr/bin/env python3
"""
Launch RF-DETR training on RunPod.

Automates the full workflow:
  1. Create a GPU pod on RunPod
  2. Wait for it to be ready
  3. Upload dataset and training script via SCP
  4. Run training via SSH
  5. Download trained weights
  6. Terminate the pod

Usage:
  python scripts/launch_runpod.py --dataset dataset/ --epochs 50
  python scripts/launch_runpod.py --status           # check running pod
  python scripts/launch_runpod.py --terminate         # shut down pod

Requires:
  - .env file with RUNPOD_API_KEY=your_key
  - pip install runpod python-dotenv
"""

import argparse
import json
import os
import subprocess
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_FILE = os.path.join(PROJECT_ROOT, ".env")
POD_STATE_FILE = os.path.join(PROJECT_ROOT, ".runpod_pod.json")


def load_api_key():
    """Load RunPod API key from .env file without exposing it."""
    from dotenv import dotenv_values
    config = dotenv_values(ENV_FILE)
    key = config.get("RUNPOD_API_KEY", "")
    if not key or key == "paste_your_key_here":
        print("ERROR: Set your RunPod API key in .env")
        print(f"  File: {ENV_FILE}")
        sys.exit(1)
    return key


def init_runpod():
    """Initialize RunPod SDK with API key."""
    import runpod
    runpod.api_key = load_api_key()
    return runpod


def save_pod_state(pod_info):
    """Save pod info to local file for later reference."""
    with open(POD_STATE_FILE, "w") as f:
        json.dump(pod_info, f, indent=2)


def load_pod_state():
    """Load saved pod info."""
    if not os.path.exists(POD_STATE_FILE):
        return None
    with open(POD_STATE_FILE) as f:
        return json.load(f)


def get_ssh_info(pod):
    """Extract SSH connection info from pod details."""
    runtime = pod.get("runtime")
    if not runtime:
        return None, None, None

    ports = runtime.get("ports") or []
    for port in ports:
        if port.get("privatePort") == 22:
            return port.get("ip"), port.get("publicPort"), "root"

    return None, None, None


def ssh_is_reachable(host, port, timeout=5):
    """Test if SSH port is accepting connections and responsive."""
    import socket
    try:
        with socket.create_connection((host, port), timeout=timeout) as sock:
            # SSH servers send a banner like "SSH-2.0-OpenSSH_8.9"
            sock.settimeout(timeout)
            banner = sock.recv(256).decode("utf-8", errors="ignore")
            return banner.startswith("SSH-")
    except (socket.timeout, ConnectionRefusedError, OSError):
        return False


def wait_for_pod(rp, pod_id, gpu_type, timeout_sec=600):
    """Wait for a pod to be ready with SSH accessible. Returns pod info or None."""
    print("Waiting for pod to start...")
    start = time.time()
    phase = "api"  # api -> ssh

    while time.time() - start < timeout_sec:
        elapsed = int(time.time() - start)
        info = rp.get_pod(pod_id)
        status = info.get("desiredStatus", "unknown")

        if status == "EXITED":
            print(f"\n  Pod exited unexpectedly. Check RunPod dashboard.")
            return None

        # Phase 1: wait for API to report runtime with SSH port info
        if phase == "api":
            ssh_host, ssh_port, ssh_user = get_ssh_info(info)
            if ssh_host and ssh_port:
                print(f"\n  Container running. Waiting for SSH...")
                phase = "ssh"
                save_pod_state({
                    "id": pod_id,
                    "gpu": gpu_type,
                    "ssh_host": ssh_host,
                    "ssh_port": ssh_port,
                    "ssh_user": ssh_user,
                })
            else:
                sys.stdout.write(f"\r  [{elapsed}s] Status: {status} (waiting for container)...")
                sys.stdout.flush()
                time.sleep(5)
                continue

        # Phase 2: wait for SSH to actually accept connections
        if phase == "ssh":
            if ssh_is_reachable(ssh_host, ssh_port):
                print(f"  SSH ready! ({elapsed}s total)")
                print(f"  SSH: ssh {ssh_user}@{ssh_host} -p {ssh_port}")
                return info
            else:
                sys.stdout.write(f"\r  [{elapsed}s] SSH port not ready yet...")
                sys.stdout.flush()
                time.sleep(3)

    print(f"\n  Timeout after {timeout_sec}s. Check RunPod dashboard.")
    return None


def create_pod(args):
    """Create a new RunPod GPU pod."""
    rp = init_runpod()

    print(f"Creating RunPod pod...")
    print(f"  GPU: {args.gpu_type}")
    print(f"  GPU count: {args.gpu_count}")
    print(f"  Disk: {args.disk_size}GB")
    print()

    pod = rp.create_pod(
        name="all22-rfdetr-training",
        image_name="runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04",
        gpu_type_id=args.gpu_type,
        gpu_count=args.gpu_count,
        volume_in_gb=0,
        container_disk_in_gb=args.disk_size,
        ports="22/tcp",
        start_ssh=True,
    )

    pod_id = pod["id"]
    print(f"Pod created: {pod_id}")
    save_pod_state({"id": pod_id, "gpu": args.gpu_type})

    return wait_for_pod(rp, pod_id, args.gpu_type)


def upload_dataset(args):
    """Upload dataset and training script to the pod via SCP."""
    state = load_pod_state()
    if not state or "ssh_host" not in state:
        print("ERROR: No running pod found. Run with --create first.")
        sys.exit(1)

    ssh_host = state["ssh_host"]
    ssh_port = state["ssh_port"]
    ssh_user = state["ssh_user"]
    ssh_target = f"{ssh_user}@{ssh_host}"
    ssh_opts = ["-o", "StrictHostKeyChecking=no", "-p", str(ssh_port)]
    # SCP uses -P (uppercase) for port, SSH uses -p (lowercase)
    scp_opts = ["-o", "StrictHostKeyChecking=no", "-P", str(ssh_port)]

    dataset_dir = os.path.abspath(args.dataset)
    if not os.path.exists(dataset_dir):
        print(f"ERROR: Dataset not found: {dataset_dir}")
        sys.exit(1)

    print(f"Uploading to pod...")

    # Upload training script
    train_script = os.path.join(PROJECT_ROOT, "scripts", "train_rfdetr.py")
    requirements = os.path.join(PROJECT_ROOT, "scripts", "requirements_runpod.txt")

    print("  Uploading training script...")
    subprocess.run(
        ["scp"] + scp_opts + [train_script, requirements, f"{ssh_target}:/workspace/"],
        check=True,
    )

    # Upload dataset as tarball (much faster than scp -r with many small files)
    print(f"  Packing dataset...")
    tar_path = os.path.join(PROJECT_ROOT, ".dataset_upload.tar.gz")
    subprocess.run(
        ["tar", "-czf", tar_path, "-C", os.path.dirname(dataset_dir), os.path.basename(dataset_dir)],
        check=True,
    )
    tar_size_mb = os.path.getsize(tar_path) / 1024 / 1024
    print(f"  Uploading dataset ({tar_size_mb:.0f}MB tarball)...")
    subprocess.run(
        ["scp"] + scp_opts + [tar_path, f"{ssh_target}:/workspace/dataset.tar.gz"],
        check=True,
    )
    os.remove(tar_path)
    print(f"  Extracting on pod...")
    subprocess.run(
        ["ssh"] + ssh_opts + [ssh_target, "cd /workspace && tar -xzf dataset.tar.gz && rm dataset.tar.gz"],
        check=True,
    )

    # Install dependencies
    print("  Installing dependencies (this can take a few minutes)...")
    proc = subprocess.Popen(
        ["ssh"] + ssh_opts + [ssh_target,
         "cd /workspace && python -m venv venv --system-site-packages"
         " && source venv/bin/activate"
         " && pip install -r requirements_runpod.txt"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    for line in proc.stdout:
        print(f"    {line}", end="", flush=True)
    proc.wait()
    if proc.returncode != 0:
        print("  ⚠ Dependency install failed!")
        sys.exit(1)

    print("  ✓ Upload and install complete!")


def run_training(args):
    """Launch training detached on the pod via nohup.

    Training runs independently of the local SSH session, so it survives
    Claude session timeouts or network disconnects. Progress is written
    to /workspace/train.log on the pod. Use --check-training to read it.
    """
    state = load_pod_state()
    if not state or "ssh_host" not in state:
        print("ERROR: No running pod found. Run with --create first.")
        sys.exit(1)

    ssh_host = state["ssh_host"]
    ssh_port = state["ssh_port"]
    ssh_user = state["ssh_user"]
    ssh_target = f"{ssh_user}@{ssh_host}"
    ssh_opts = ["-o", "StrictHostKeyChecking=no", "-p", str(ssh_port)]

    train_args = (
        f" --dataset dataset"
        f" --epochs {args.epochs}"
        f" --batch-size {args.batch_size}"
        f" --grad-accum {args.grad_accum}"
        f" --resolution {args.resolution}"
        f" --devices {args.gpu_count}"
    )

    if args.gpu_count > 1:
        train_args += " --strategy ddp"
        train_cmd = f"cd /workspace && source venv/bin/activate && python -m torch.distributed.run --nproc_per_node={args.gpu_count} train_rfdetr.py{train_args}"
    else:
        train_cmd = f"cd /workspace && source venv/bin/activate && python -u train_rfdetr.py{train_args}"

    # Wrap in nohup so training survives SSH disconnect
    detached_cmd = f"nohup bash -c '{train_cmd}' > /workspace/train.log 2>&1 &"

    print(f"Launching training (detached)...")
    print(f"  Training command: {train_cmd}")
    print(f"  Log file: /workspace/train.log")
    print()

    subprocess.run(
        ["ssh"] + ssh_opts + [ssh_target, detached_cmd],
        check=True,
    )

    # Brief pause, then verify the process started
    time.sleep(3)
    result = subprocess.run(
        ["ssh"] + ssh_opts + [ssh_target, "pgrep -f train_rfdetr.py"],
        capture_output=True, text=True,
    )
    if result.returncode == 0:
        pid = result.stdout.strip().split('\n')[0]
        print(f"✓ Training launched (PID {pid})")
        print(f"  Monitor with: python scripts/launch_runpod.py --check-training")
        print(f"  Or SSH in:    ssh {ssh_user}@{ssh_host} -p {ssh_port}")
        print(f"                tail -f /workspace/train.log")

        # Save training state
        state["training_pid"] = pid
        state["training_started"] = time.strftime("%Y-%m-%d %H:%M:%S")
        save_pod_state(state)
        return True
    else:
        print("⚠ Training process not found — may have failed to start.")
        print("  Check: ssh into pod and look at /workspace/train.log")
        return False


def check_training(args=None):
    """Check training progress by reading the log file on the pod."""
    state = load_pod_state()
    if not state or "ssh_host" not in state:
        print("ERROR: No running pod found.")
        sys.exit(1)

    ssh_host = state["ssh_host"]
    ssh_port = state["ssh_port"]
    ssh_user = state["ssh_user"]
    ssh_target = f"{ssh_user}@{ssh_host}"
    ssh_opts = ["-o", "StrictHostKeyChecking=no", "-p", str(ssh_port)]

    # Check if training is still running
    proc_check = subprocess.run(
        ["ssh"] + ssh_opts + [ssh_target, "pgrep -f train_rfdetr.py"],
        capture_output=True, text=True,
    )
    is_running = proc_check.returncode == 0

    # Get best mAP line
    best_map = subprocess.run(
        ["ssh"] + ssh_opts + [ssh_target, "grep 'Best EMA' /workspace/train.log | tail -1"],
        capture_output=True, text=True,
    )

    # Get latest val metrics
    latest_val = subprocess.run(
        ["ssh"] + ssh_opts + [ssh_target, "grep -c 'Val — Overall' /workspace/train.log"],
        capture_output=True, text=True,
    )

    # Get tail of log
    log_tail = subprocess.run(
        ["ssh"] + ssh_opts + [ssh_target, "tail -30 /workspace/train.log"],
        capture_output=True, text=True,
    )

    # Check for completion message
    completed = subprocess.run(
        ["ssh"] + ssh_opts + [ssh_target, "grep 'Training complete' /workspace/train.log"],
        capture_output=True, text=True,
    )

    epochs_done = latest_val.stdout.strip() if latest_val.returncode == 0 else "?"
    status = "RUNNING" if is_running else ("COMPLETED" if completed.stdout.strip() else "STOPPED/FAILED")

    print(f"Training Status: {status}")
    if state.get("training_started"):
        print(f"  Started: {state['training_started']}")
    print(f"  Epochs completed: {epochs_done}")
    if best_map.stdout.strip():
        print(f"  {best_map.stdout.strip()}")
    print()
    print("--- Last 30 lines of train.log ---")
    print(log_tail.stdout if log_tail.returncode == 0 else "(could not read log)")

    return status


def download_weights(args):
    """Download trained weights from the pod."""
    state = load_pod_state()
    if not state or "ssh_host" not in state:
        print("ERROR: No running pod found.")
        sys.exit(1)

    ssh_host = state["ssh_host"]
    ssh_port = state["ssh_port"]
    ssh_user = state["ssh_user"]
    ssh_target = f"{ssh_user}@{ssh_host}"
    scp_opts = ["-o", "StrictHostKeyChecking=no", "-P", str(ssh_port)]

    output_dir = os.path.join(PROJECT_ROOT, "models")
    os.makedirs(output_dir, exist_ok=True)

    print("Downloading trained weights...")
    for weight_file in ["best.pt", "last.pt"]:
        remote_path = f"/workspace/output/{weight_file}"
        local_path = os.path.join(output_dir, f"rfdetr_{weight_file}")
        try:
            subprocess.run(
                ["scp"] + scp_opts + [f"{ssh_target}:{remote_path}", local_path],
                check=True,
            )
            print(f"  {weight_file} → {local_path}")
        except subprocess.CalledProcessError:
            print(f"  {weight_file} not found on pod")

    print("Done!")


def terminate_pod(args=None):
    """Terminate the running pod."""
    rp = init_runpod()
    state = load_pod_state()

    if not state:
        print("No saved pod state found.")
        return

    pod_id = state["id"]
    print(f"Terminating pod {pod_id}...")
    rp.terminate_pod(pod_id)
    os.remove(POD_STATE_FILE)
    print("Pod terminated.")


def check_status(args=None):
    """Check status of the current pod."""
    rp = init_runpod()
    state = load_pod_state()

    if not state:
        print("No saved pod state found.")
        return

    pod_id = state["id"]
    info = rp.get_pod(pod_id)

    if not info:
        print(f"Pod {pod_id} not found (may have been terminated).")
        return

    status = info.get("desiredStatus", "unknown")
    runtime = info.get("runtime", {})
    uptime = runtime.get("uptimeInSeconds", 0) if runtime else 0
    gpu = info.get("machine", {}).get("gpuDisplayName", "unknown")
    gpu_count = info.get("gpuCount", 1)

    print(f"Pod: {pod_id}")
    print(f"  Status: {status}")
    print(f"  GPU: {gpu_count}x {gpu}")
    print(f"  Uptime: {uptime}s ({uptime/3600:.1f}h)")

    if "ssh_host" in state:
        host, port = state["ssh_host"], state["ssh_port"]
        reachable = ssh_is_reachable(host, port)
        print(f"  SSH: ssh {state['ssh_user']}@{host} -p {port}")
        print(f"  SSH reachable: {'yes' if reachable else 'no'}")


def run_full_pipeline(args):
    """Run the full training pipeline: create → upload → launch training (detached).

    Training runs detached on the pod. After launching, use:
      --check-training   to monitor progress
      --download-only    to grab weights when done
      --terminate        to shut down the pod
    """
    print("=" * 60)
    print("  RF-DETR Training Pipeline — RunPod")
    print("=" * 60)

    # Step 1: Create pod
    print("\n[1/3] Creating pod...")
    pod = create_pod(args)
    if not pod:
        return

    # Step 2: Upload
    print("\n[2/3] Uploading dataset and scripts...")
    upload_dataset(args)

    # Step 3: Launch training (detached — does not block)
    print("\n[3/3] Launching training...")
    success = run_training(args)

    if not success:
        print("\n⚠ Training failed to start. Pod left running for debugging.")
        print("  SSH in to check /workspace/train.log")
        return

    print("\n" + "=" * 60)
    print("  Training launched! It runs independently on the pod.")
    print()
    print("  Next steps:")
    print("    python scripts/launch_runpod.py --check-training    # monitor progress")
    print("    python scripts/launch_runpod.py --download-only     # grab weights when done")
    print("    python scripts/launch_runpod.py --terminate         # shut down pod")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Launch RF-DETR training on RunPod")

    # Actions
    parser.add_argument("--status", action="store_true", help="Check pod status")
    parser.add_argument("--terminate", action="store_true", help="Terminate running pod")
    parser.add_argument("--create-only", action="store_true", help="Only create pod, don't train")
    parser.add_argument("--upload-only", action="store_true", help="Only upload dataset")
    parser.add_argument("--train-only", action="store_true", help="Only launch training (pod must exist)")
    parser.add_argument("--download-only", action="store_true", help="Only download weights")
    parser.add_argument("--check-training", action="store_true", help="Check training progress from log file")

    # Pod config
    parser.add_argument("--gpu-type", default="NVIDIA GeForce RTX 5090",
                        help="GPU type (default: NVIDIA GeForce RTX 5090)")
    parser.add_argument("--gpu-count", type=int, default=1, help="Number of GPUs (default: 1)")
    parser.add_argument("--disk-size", type=int, default=50, help="Container disk size in GB (default: 50)")

    # Training config
    parser.add_argument("--dataset", default="dataset", help="Local path to COCO dataset")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs (default: 50)")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size per GPU (default: 4)")
    parser.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation (default: 4)")
    parser.add_argument("--resolution", type=int, default=1280, help="Input resolution (default: 1280)")

    args = parser.parse_args()

    # Handle individual actions
    if args.status:
        check_status()
        return
    if args.terminate:
        terminate_pod()
        return
    if args.create_only:
        create_pod(args)
        return
    if args.upload_only:
        upload_dataset(args)
        return
    if args.train_only:
        run_training(args)
        return
    if args.download_only:
        download_weights(args)
        return
    if args.check_training:
        check_training()
        return

    # Full pipeline
    run_full_pipeline(args)


if __name__ == "__main__":
    main()
