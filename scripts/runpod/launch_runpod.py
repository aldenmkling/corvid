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

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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


def wait_for_pod(rp, pod_id, gpu_type, timeout_sec=600, container_timeout_sec=120):
    """Wait for a pod to be ready with SSH accessible. Returns pod info or None.

    Two-phase wait:
      1. Wait for the API to report runtime ports (container started).
         If this takes longer than container_timeout_sec, the host is likely
         stuck pulling the Docker image — return None so caller can retry.
      2. Wait for SSH to accept connections (banner check).
    """
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
            # Detect stuck container: if ports stay null too long, host is bad
            if elapsed > container_timeout_sec:
                print(f"\n  Container not starting after {container_timeout_sec}s — host may be stuck.")
                return None

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
    """Create a new RunPod GPU pod.

    Auto-terminates the pod on boot failure so we don't leak billing on
    stuck Docker image pulls. Retries up to --create-retries times for
    stuck-host cases.
    """
    rp = init_runpod()

    cloud_type = getattr(args, 'cloud_type', 'ALL')
    retries = getattr(args, 'create_retries', 3)

    print(f"Creating RunPod pod...")
    print(f"  GPU: {args.gpu_type}")
    print(f"  GPU count: {args.gpu_count}")
    print(f"  Disk: {args.disk_size}GB")
    print(f"  Cloud type: {cloud_type}")
    print()

    for attempt in range(1, retries + 1):
        pod = rp.create_pod(
            name=f"all22-{args.training_type}-training",
            image_name="runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04",
            gpu_type_id=args.gpu_type,
            gpu_count=args.gpu_count,
            volume_in_gb=0,
            container_disk_in_gb=args.disk_size,
            ports="22/tcp",
            start_ssh=True,
            cloud_type=cloud_type,
        )

        pod_id = pod["id"]
        print(f"[attempt {attempt}/{retries}] Pod created: {pod_id}")
        save_pod_state({"id": pod_id, "gpu": args.gpu_type})

        info = wait_for_pod(rp, pod_id, args.gpu_type)
        if info is not None:
            return info

        # Boot failed. Terminate this pod before retrying.
        print(f"  [attempt {attempt}] boot failed — terminating pod {pod_id}")
        try:
            rp.terminate_pod(pod_id)
        except Exception as e:
            print(f"  WARNING: terminate failed: {e}")

    print(f"All {retries} attempts failed.")
    return None


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

    # Upload training script — pick based on training type
    tt = getattr(args, 'training_type', 'rfdetr')
    if tt == 'hrnet':
        train_script = os.path.join(PROJECT_ROOT, "scripts", "training", "train_hrnet_keypoints.py")
        requirements = os.path.join(PROJECT_ROOT, "scripts", "training", "requirements_hrnet_runpod.txt")
    elif tt == 'unet':
        train_script = os.path.join(PROJECT_ROOT, "scripts", "training", "train_unet_lines.py")
        requirements = os.path.join(PROJECT_ROOT, "scripts", "training", "requirements_unet_runpod.txt")
    else:
        train_script = os.path.join(PROJECT_ROOT, "scripts", "training", "train_rfdetr.py")
        requirements = os.path.join(PROJECT_ROOT, "scripts", "training", "requirements_runpod.txt")

    print("  Uploading training script...")
    subprocess.run(
        ["scp"] + scp_opts + [train_script, requirements, f"{ssh_target}:/workspace/"],
        check=True,
    )

    # Upload dataset as tarball (much faster than scp -r with many small files).
    # For HRNet fine-tuning, only pack the train/ and valid/ subdirectories so
    # we don't ship along unrelated sibling folders (annotation_images, etc.).
    # Use a temp staging dir so the archive contains just {dataset_name}/train,
    # {dataset_name}/valid — avoids needing GNU-specific tar --transform.
    import tempfile
    print(f"  Packing dataset...")
    tar_path = os.path.join(PROJECT_ROOT, ".dataset_upload.tar.gz")
    dataset_name = os.path.basename(dataset_dir)
    has_split = os.path.isdir(os.path.join(dataset_dir, "train")) and \
                os.path.isdir(os.path.join(dataset_dir, "valid"))
    # COPYFILE_DISABLE=1 tells macOS bsdtar not to emit `._filename`
    # AppleDouble sidecars, which otherwise pollute the extracted dataset on
    # Linux and trip up any glob that matches `*.jpg` or `*.png`.
    tar_env = {**os.environ, "COPYFILE_DISABLE": "1"}
    if getattr(args, 'training_type', 'rfdetr') in ('hrnet', 'unet') and has_split:
        with tempfile.TemporaryDirectory() as staging:
            stage_root = os.path.join(staging, dataset_name)
            os.makedirs(stage_root)
            # Symlinks are cheap and tar follows them by default
            os.symlink(os.path.join(dataset_dir, "train"),
                       os.path.join(stage_root, "train"))
            os.symlink(os.path.join(dataset_dir, "valid"),
                       os.path.join(stage_root, "valid"))
            subprocess.run(
                ["tar", "-czhf", tar_path,  # -h follows symlinks
                 "-C", staging,
                 dataset_name],
                env=tar_env,
                check=True,
            )
    else:
        subprocess.run(
            ["tar", "-czf", tar_path,
             "-C", os.path.dirname(dataset_dir),
             dataset_name],
            env=tar_env,
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

    # Upload resume checkpoint if provided (HRNet only)
    resume_path = getattr(args, 'resume', None)
    if resume_path:
        if not os.path.isabs(resume_path):
            resume_path = os.path.join(PROJECT_ROOT, resume_path)
        if not os.path.exists(resume_path):
            print(f"ERROR: Resume checkpoint not found: {resume_path}")
            sys.exit(1)
        size_mb = os.path.getsize(resume_path) / 1024 / 1024
        print(f"  Uploading resume checkpoint ({size_mb:.0f}MB)...")
        subprocess.run(
            ["scp"] + scp_opts + [resume_path, f"{ssh_target}:/workspace/resume.pth"],
            check=True,
        )

    # Install dependencies
    _tt = getattr(args, 'training_type', 'rfdetr')
    req_filename = {
        "hrnet": "requirements_hrnet_runpod.txt",
        "unet":  "requirements_unet_runpod.txt",
    }.get(_tt, "requirements_runpod.txt")
    print("  Installing dependencies (this can take a few minutes)...")
    proc = subprocess.Popen(
        ["ssh"] + ssh_opts + [ssh_target,
         "cd /workspace && python -m venv venv --system-site-packages"
         " && source venv/bin/activate"
         f" && pip install -r {req_filename}"],
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

    if getattr(args, 'training_type', 'rfdetr') == 'hrnet':
        # Detect whether the uploaded dataset has a train/valid split layout
        # (real-data fine-tuning) or a single-directory layout (synthetic).
        dataset_basename = os.path.basename(os.path.abspath(args.dataset).rstrip(os.sep))
        has_split = os.path.isdir(os.path.join(args.dataset, "train")) and \
                    os.path.isdir(os.path.join(args.dataset, "valid"))

        if has_split:
            train_args = (
                f" --dataset {dataset_basename}/train"
                f" --val-dataset {dataset_basename}/valid"
            )
        else:
            train_args = f" --dataset {dataset_basename}"

        train_args += (
            f" --epochs {args.epochs}"
            f" --batch-size {args.batch_size}"
            f" --lr {getattr(args, 'lr', 1e-3)}"
            f" --backbone-lr-mult {getattr(args, 'backbone_lr_mult', 0.1)}"
            f" --output /workspace/output"
        )
        if getattr(args, 'no_pretrained', False):
            train_args += " --no-pretrained"
        if getattr(args, 'resume', None):
            # Resume weights are uploaded to /workspace/resume.pth
            train_args += " --resume /workspace/resume.pth"
        if getattr(args, 'sigma_max', None) is not None:
            train_args += f" --sigma-max {args.sigma_max}"
        if getattr(args, 'sigma_min', None) is not None:
            train_args += f" --sigma-min {args.sigma_min}"
        if getattr(args, 'sigma_shrink_epochs', None) is not None:
            train_args += f" --sigma-shrink-epochs {args.sigma_shrink_epochs}"
        if getattr(args, 'channel_weights', None):
            train_args += f" --channel-weights {args.channel_weights}"
        if getattr(args, 'backbone', None):
            train_args += f" --backbone {args.backbone}"
        if getattr(args, 'num_channels', None) is not None:
            train_args += f" --num-channels {args.num_channels}"
        train_cmd = f"cd /workspace && source venv/bin/activate && python -u train_hrnet_keypoints.py{train_args}"
    elif getattr(args, 'training_type', 'rfdetr') == 'unet':
        dataset_basename = os.path.basename(os.path.abspath(args.dataset).rstrip(os.sep))
        train_args = (
            f" --dataset {dataset_basename}"
            f" --epochs {args.epochs}"
            f" --batch-size {args.batch_size}"
            f" --lr {getattr(args, 'lr', 1e-3)}"
            f" --encoder-lr-mult {getattr(args, 'backbone_lr_mult', 0.1)}"
            f" --output /workspace/output"
            f" --amp"
        )
        if getattr(args, 'encoder', None):
            train_args += f" --encoder {args.encoder}"
        if getattr(args, 'resume', None):
            train_args += " --resume /workspace/resume.pth"
        train_cmd = f"cd /workspace && source venv/bin/activate && python -u train_unet_lines.py{train_args}"
    else:
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
    _tt = getattr(args, 'training_type', 'rfdetr')
    script_name = {
        "hrnet": "train_hrnet_keypoints.py",
        "unet":  "train_unet_lines.py",
    }.get(_tt, "train_rfdetr.py")
    result = subprocess.run(
        ["ssh"] + ssh_opts + [ssh_target, f"pgrep -f {script_name}"],
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

    # Check if training is still running (match either training script)
    proc_check = subprocess.run(
        ["ssh"] + ssh_opts + [ssh_target, "pgrep -f 'train_(rfdetr|hrnet_keypoints|unet_lines)\\.py'"],
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

    training_type = getattr(args, 'training_type', 'rfdetr')
    if training_type == "hrnet":
        weight_files = ["best.pth", "last.pth"]
        prefix = "hrnet"
        ext = ".pth"
    else:
        weight_files = ["best.pt", "last.pt"]
        prefix = "rfdetr"
        ext = ".pt"

    print(f"Downloading {training_type} weights...")
    for weight_file in weight_files:
        remote_path = f"/workspace/output/{weight_file}"
        local_path = os.path.join(output_dir, f"{prefix}_{weight_file}")
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
    parser.add_argument("--cloud-type", default="SECURE",
                        choices=["SECURE", "COMMUNITY", "ALL"],
                        help="SECURE = dedicated GPU (no contention, slightly pricier, "
                             "but avoids OOMs and stuck-host issues). COMMUNITY = "
                             "consumer hosts, cheaper but unreliable. ALL = either.")
    parser.add_argument("--create-retries", type=int, default=3,
                        help="Retry pod creation this many times on boot failure.")
    parser.add_argument("--gpu-type", default="NVIDIA GeForce RTX 5090",
                        help="GPU type (default: NVIDIA GeForce RTX 5090)")
    parser.add_argument("--gpu-count", type=int, default=1, help="Number of GPUs (default: 1)")
    parser.add_argument("--disk-size", type=int, default=50, help="Container disk size in GB (default: 50)")

    # Training config
    parser.add_argument("--training-type", default="rfdetr", choices=["rfdetr", "hrnet", "unet"],
                        help="Model to train (default: rfdetr)")
    parser.add_argument("--dataset", default=None, help="Local path to dataset (auto-set per training type)")
    parser.add_argument("--epochs", type=int, default=None, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size per GPU")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (HRNet only)")
    parser.add_argument("--backbone-lr-mult", type=float, default=0.1, help="Backbone LR multiplier (HRNet only)")
    parser.add_argument("--no-pretrained", action="store_true", help="Don't use pretrained backbone (HRNet only)")
    parser.add_argument("--resume", default=None,
                        help="Local path to checkpoint to resume from (HRNet only). "
                             "Will be uploaded to /workspace/resume.pth on the pod.")
    parser.add_argument("--sigma-max", type=float, default=None,
                        help="Starting heatmap sigma (HRNet only). Set equal to --sigma-min to hold tight.")
    parser.add_argument("--sigma-min", type=float, default=None,
                        help="Final heatmap sigma (HRNet only).")
    parser.add_argument("--sigma-shrink-epochs", type=int, default=None,
                        help="Epochs over which sigma decays from max to min (HRNet only). "
                             "Default: 60%% of --epochs.")
    parser.add_argument("--channel-weights", default=None,
                        help="Comma-separated per-channel loss weights (HRNet only), e.g. '3,1'.")
    parser.add_argument("--backbone", default=None,
                        help="HRNet backbone name (HRNet only). Default hrnet_w48; "
                             "use hrnet_w18 for the downsized hash-only variant.")
    parser.add_argument("--num-channels", type=int, default=None,
                        help="Output heatmap channels (HRNet only). Default 2; "
                             "use 1 for hash-only.")
    # UNet specific
    parser.add_argument("--encoder", default=None,
                        help="UNet encoder (UNet only). Default efficientnet-b0.")
    # RF-DETR specific
    parser.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation (RF-DETR only)")
    parser.add_argument("--resolution", type=int, default=1280, help="Input resolution (RF-DETR only)")

    args = parser.parse_args()

    # Set defaults based on training type
    if args.training_type == "hrnet":
        if args.dataset is None:
            args.dataset = os.path.join(PROJECT_ROOT, "data", "field_keypoints")
        if args.epochs is None:
            args.epochs = 100
        if args.batch_size is None:
            args.batch_size = 16
    elif args.training_type == "unet":
        if args.dataset is None:
            args.dataset = os.path.join(PROJECT_ROOT, "data", "line_detection")
        if args.epochs is None:
            args.epochs = 100
        if args.batch_size is None:
            args.batch_size = 16
    else:
        if args.dataset is None:
            args.dataset = os.path.join(PROJECT_ROOT, "data", "player_detection")
        if args.epochs is None:
            args.epochs = 50
        if args.batch_size is None:
            args.batch_size = 4

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
