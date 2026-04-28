#!/usr/bin/env python3
"""Launch RTMO-s hash detection training on RunPod.

Reuses helpers from launch_runpod.py: pod creation, SSH polling, file
upload via tar+scp. Heavy lifting is in setup_and_train_rtmo.sh which
the pod runs.

Usage:
    python scripts/runpod/launch_rtmo.py
    python scripts/runpod/launch_rtmo.py --status
    python scripts/runpod/launch_rtmo.py --terminate
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

# Reuse the existing launcher's helpers.
from scripts.runpod.launch_runpod import (
    create_pod, get_ssh_info, init_runpod, load_pod_state, save_pod_state,
)

DATA_DIR = os.path.join(PROJECT_ROOT, "data/field_keypoints_rtmo")
CONFIG_PY = os.path.join(PROJECT_ROOT, "scripts/training/rtmo_hash_config.py")
SETUP_SH = os.path.join(PROJECT_ROOT, "scripts/training/setup_and_train_rtmo.sh")
SETUP_ONLY_SH = os.path.join(PROJECT_ROOT, "scripts/training/setup_only_rtmo.sh")
SSH_KEY = os.path.expanduser("~/.ssh/id_ed25519")


def ssh_run(host, port, user, cmd, check=True):
    full = (f"ssh -i {SSH_KEY} -o StrictHostKeyChecking=no "
            f"{user}@{host} -p {port} {cmd!r}")
    print(f"  $ {full}")
    return subprocess.run(full, shell=True, check=check)


def scp_file(local, host, port, user, remote):
    full = (f"scp -i {SSH_KEY} -o StrictHostKeyChecking=no -P {port} "
            f"{local!r} {user}@{host}:{remote!r}")
    print(f"  $ {full}")
    subprocess.run(full, shell=True, check=True)


def upload_config_and_setup(pod_info):
    """Stage 1: upload only config + setup_only.sh + train.sh. No data yet."""
    host, port, user = get_ssh_info(pod_info)
    if not host or not port:
        print("ERROR: cannot get SSH info from pod"); return False
    scp_file(CONFIG_PY, host, port, user, "/workspace/rtmo_hash_config.py")
    scp_file(SETUP_ONLY_SH, host, port, user, "/workspace/setup_only_rtmo.sh")
    scp_file(SETUP_SH, host, port, user, "/workspace/setup_and_train_rtmo.sh")
    ssh_run(host, port, user,
            "chmod +x /workspace/setup_only_rtmo.sh "
            "/workspace/setup_and_train_rtmo.sh")
    return True


def run_stage1_setup(pod_info):
    """Stage 1: install MMPose stack, parse config, build model. Aborts if
    anything fails — saves us from uploading data into a broken setup."""
    host, port, user = get_ssh_info(pod_info)
    print(f"\n  Stage 1: installing mmpose + verifying config (5-10 min)...")
    res = subprocess.run(
        f"ssh -i {SSH_KEY} -o StrictHostKeyChecking=no "
        f"{user}@{host} -p {port} "
        f"'bash /workspace/setup_only_rtmo.sh'",
        shell=True,
    )
    return res.returncode == 0


def upload_data(pod_info):
    """Stage 2: upload converted dataset."""
    host, port, user = get_ssh_info(pod_info)
    if not os.path.exists(os.path.join(DATA_DIR, "train/annotations.json")):
        print(f"ERROR: missing {DATA_DIR}/train/annotations.json — "
              f"run scripts/data_prep/convert_to_mmpose_rtmo.py first")
        return False
    with tempfile.TemporaryDirectory() as td:
        tar_path = os.path.join(td, "data_rtmo.tar")
        print(f"\n  packing {DATA_DIR} → {tar_path}")
        subprocess.run(
            ["tar", "-cLf", tar_path, "-C",
             os.path.dirname(DATA_DIR), os.path.basename(DATA_DIR)],
            check=True,
        )
        sz = os.path.getsize(tar_path) / 1e6
        print(f"  tar size: {sz:.1f} MB")
        scp_file(tar_path, host, port, user, "/workspace/data_rtmo.tar")
        ssh_run(host, port, user,
                "cd /workspace && tar -xf data_rtmo.tar && "
                "mv field_keypoints_rtmo data_rtmo && rm data_rtmo.tar")
    return True


def kick_off_training(pod_info):
    host, port, user = get_ssh_info(pod_info)
    print(f"\n  starting training in background (nohup) on pod...")
    ssh_run(
        host, port, user,
        "cd /workspace && source venv/bin/activate && "
        "nohup python /workspace/mmpose/tools/train.py /workspace/rtmo_hash_config.py "
        "--work-dir /workspace/output_rtmo_hash > train.log 2>&1 & disown",
    )
    print(f"\n  Training kicked off. Monitor:")
    print(f"    ssh -i {SSH_KEY} {user}@{host} -p {port} 'tail -f /workspace/train.log'")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu-type", default="NVIDIA GeForce RTX 5090")
    ap.add_argument("--gpu-count", type=int, default=1)
    ap.add_argument("--disk-size", type=int, default=80)
    ap.add_argument("--cloud-type", default="SECURE")
    ap.add_argument("--create-retries", type=int, default=3)
    ap.add_argument("--training-type", default="rtmo")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--terminate", action="store_true")
    args = ap.parse_args()

    if args.status:
        state = load_pod_state()
        if not state:
            print("no pod state file")
            return
        rp = init_runpod()
        info = rp.get_pod(state["id"])
        print(info)
        return

    if args.terminate:
        state = load_pod_state()
        if not state:
            print("no pod state to terminate")
            return
        rp = init_runpod()
        rp.terminate_pod(state["id"])
        print(f"terminated {state['id']}")
        return

    pod_info = create_pod(args)
    if pod_info is None:
        print("FAIL: pod creation"); sys.exit(1)
    if not upload_config_and_setup(pod_info):
        print("FAIL: stage1 upload"); sys.exit(1)
    if not run_stage1_setup(pod_info):
        host, port, user = get_ssh_info(pod_info)
        print(f"\n  Stage 1 FAILED. SSH in to debug:")
        print(f"    ssh -i {SSH_KEY} {user}@{host} -p {port}")
        print(f"  Pod kept alive for inspection. Terminate with:")
        print(f"    python scripts/runpod/launch_rtmo.py --terminate")
        sys.exit(1)
    print(f"\n  Stage 1 OK. Uploading dataset (~tens of MB)...")
    if not upload_data(pod_info):
        print("FAIL: stage2 upload"); sys.exit(1)
    kick_off_training(pod_info)


if __name__ == "__main__":
    main()
