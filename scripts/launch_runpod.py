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


def get_ssh_command(pod):
    """Extract SSH connection info from pod details."""
    # RunPod exposes SSH via a proxied port
    runtime = pod.get("runtime", {})
    if not runtime:
        return None, None, None

    ports = runtime.get("ports", [])
    ssh_port = None
    ssh_host = None

    for port in ports:
        if port.get("privatePort") == 22:
            ssh_host = port.get("ip")
            ssh_port = port.get("publicPort")
            break

    return ssh_host, ssh_port, "root"


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
        image_name="runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04",
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

    # Wait for pod to be ready
    print("Waiting for pod to start...")
    for i in range(60):
        time.sleep(5)
        info = rp.get_pod(pod_id)
        status = info.get("desiredStatus", "unknown")
        runtime = info.get("runtime")

        if runtime and runtime.get("uptimeInSeconds"):
            ssh_host, ssh_port, ssh_user = get_ssh_command(info)
            print(f"\nPod is ready! (uptime: {runtime['uptimeInSeconds']}s)")
            if ssh_host and ssh_port:
                print(f"\nSSH: ssh {ssh_user}@{ssh_host} -p {ssh_port}")
                save_pod_state({
                    "id": pod_id,
                    "gpu": args.gpu_type,
                    "ssh_host": ssh_host,
                    "ssh_port": ssh_port,
                    "ssh_user": ssh_user,
                })
            return info

        sys.stdout.write(f"\r  Status: {status} ({(i+1)*5}s elapsed)...")
        sys.stdout.flush()

    print("\nTimeout waiting for pod. Check RunPod dashboard.")
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
        ["scp"] + ssh_opts + [train_script, requirements, f"{ssh_target}:/workspace/"],
        check=True,
    )

    # Upload dataset
    print(f"  Uploading dataset from {dataset_dir}...")
    subprocess.run(
        ["scp", "-r"] + ssh_opts + [dataset_dir, f"{ssh_target}:/workspace/dataset"],
        check=True,
    )

    # Install dependencies
    print("  Installing dependencies...")
    subprocess.run(
        ["ssh"] + ssh_opts + [ssh_target, "cd /workspace && pip install -r requirements_runpod.txt -q"],
        check=True,
    )

    print("Upload complete!")


def run_training(args):
    """Run training on the pod via SSH."""
    state = load_pod_state()
    if not state or "ssh_host" not in state:
        print("ERROR: No running pod found. Run with --create first.")
        sys.exit(1)

    ssh_host = state["ssh_host"]
    ssh_port = state["ssh_port"]
    ssh_user = state["ssh_user"]
    ssh_target = f"{ssh_user}@{ssh_host}"
    ssh_opts = ["-o", "StrictHostKeyChecking=no", "-p", str(ssh_port)]

    train_cmd = (
        f"cd /workspace && python train_rfdetr.py"
        f" --dataset dataset"
        f" --epochs {args.epochs}"
        f" --batch-size {args.batch_size}"
        f" --grad-accum {args.grad_accum}"
        f" --resolution {args.resolution}"
        f" --devices {args.gpu_count}"
    )

    if args.gpu_count > 1:
        train_cmd += " --strategy ddp"

    print(f"Starting training...")
    print(f"  Command: {train_cmd}")
    print()

    # Run training (streams output live)
    subprocess.run(
        ["ssh"] + ssh_opts + [ssh_target, train_cmd],
    )


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
    ssh_opts = ["-o", "StrictHostKeyChecking=no", "-p", str(ssh_port)]

    output_dir = os.path.join(PROJECT_ROOT, "models")
    os.makedirs(output_dir, exist_ok=True)

    print("Downloading trained weights...")
    for weight_file in ["best.pt", "last.pt"]:
        remote_path = f"/workspace/output/{weight_file}"
        local_path = os.path.join(output_dir, f"rfdetr_{weight_file}")
        try:
            subprocess.run(
                ["scp"] + ssh_opts + [f"{ssh_target}:{remote_path}", local_path],
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

    print(f"Pod: {pod_id}")
    print(f"  Status: {status}")
    print(f"  GPU: {gpu}")
    print(f"  Uptime: {uptime}s ({uptime/3600:.1f}h)")

    if "ssh_host" in state:
        print(f"  SSH: ssh {state['ssh_user']}@{state['ssh_host']} -p {state['ssh_port']}")


def run_full_pipeline(args):
    """Run the full training pipeline: create → upload → train → download → terminate."""
    print("=" * 60)
    print("  RF-DETR Training Pipeline — RunPod")
    print("=" * 60)

    # Step 1: Create pod
    print("\n[1/5] Creating pod...")
    pod = create_pod(args)
    if not pod:
        return

    # Step 2: Upload
    print("\n[2/5] Uploading dataset and scripts...")
    upload_dataset(args)

    # Step 3: Train
    print("\n[3/5] Running training...")
    run_training(args)

    # Step 4: Download weights
    print("\n[4/5] Downloading weights...")
    download_weights(args)

    # Step 5: Terminate
    print("\n[5/5] Terminating pod...")
    terminate_pod()

    print("\n" + "=" * 60)
    print("  Training complete!")
    print(f"  Weights saved to: models/rfdetr_best.pt")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Launch RF-DETR training on RunPod")

    # Actions
    parser.add_argument("--status", action="store_true", help="Check pod status")
    parser.add_argument("--terminate", action="store_true", help="Terminate running pod")
    parser.add_argument("--create-only", action="store_true", help="Only create pod, don't train")
    parser.add_argument("--upload-only", action="store_true", help="Only upload dataset")
    parser.add_argument("--train-only", action="store_true", help="Only run training (pod must exist)")
    parser.add_argument("--download-only", action="store_true", help="Only download weights")

    # Pod config
    parser.add_argument("--gpu-type", default="NVIDIA GeForce RTX 5090",
                        help="GPU type (default: NVIDIA GeForce RTX 5090)")
    parser.add_argument("--gpu-count", type=int, default=1, help="Number of GPUs (default: 1)")
    parser.add_argument("--disk-size", type=int, default=50, help="Container disk size in GB (default: 50)")

    # Training config
    parser.add_argument("--dataset", default="dataset", help="Local path to COCO dataset")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs (default: 50)")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size (default: 16, halve if OOM)")
    parser.add_argument("--grad-accum", type=int, default=1, help="Gradient accumulation (default: 1)")
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

    # Full pipeline
    run_full_pipeline(args)


if __name__ == "__main__":
    main()
