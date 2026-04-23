#!/usr/bin/env python3
"""Launch clip rectification on a RunPod RTX 5090 pod.

Rendering a clip locally on CPU takes 5-10 min (HRNet inference is the
bottleneck). On a dedicated 5090 it's <1 min. Worth the upload overhead for
any batch rendering.

Usage:
  python scripts/runpod/launch_rectify.py \\
      --clip videos/clips/2019102712/play_011/sideline.mp4 \\
      --anchor 40.0 \\
      --smooth-window 31 --smooth-poly 3

Workflow:
  1. Secure-cloud RTX 5090 pod (inherits launch_runpod.py's safe defaults).
  2. Upload: HRNet weights, code (src/homography/* + test script), clip.
  3. Install deps in venv.
  4. Run rectify with --device cuda.
  5. Download: output MP4 (+ logs).
  6. Terminate pod.
"""

import argparse
import os
import subprocess
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts", "runpod"))

from launch_runpod import (  # noqa: E402
    load_api_key, wait_for_pod, save_pod_state, terminate_pod,
)

LOCAL_WEIGHTS = os.path.join(PROJECT_ROOT, "models", "hrnet_finetuned_last.pth")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output", "tracker_rectify")

# Python source files the rectify script needs (paths relative to project root)
CODE_FILES = [
    "scripts/testing/test_tracker_rectify_clip.py",
    "src/__init__.py",
    "src/homography/__init__.py",
    "src/homography/tracker.py",
    "src/homography/grid_solver.py",
    "src/homography/distortion.py",
    "src/homography/field_model.py",
    "src/homography/keypoint_detector.py",
    "src/homography/keypoint_schema.py",
    "src/homography/keypoint_track_bank.py",
    "src/homography/apply_homography.py",
]

# Minimal requirements for rectify on pod (system image has torch+cuda)
REQUIREMENTS = """timm
opencv-python-headless
scipy
"""


def create_pod(gpu_type="NVIDIA GeForce RTX 5090", disk_gb=40,
               country_code="US"):
    import runpod
    runpod.api_key = load_api_key()
    print(f"Creating pod: {gpu_type}, {disk_gb}GB disk, country={country_code}, SECURE cloud")
    pod = runpod.create_pod(
        name="all22-rectify",
        image_name="runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04",
        gpu_type_id=gpu_type,
        gpu_count=1,
        volume_in_gb=0,
        container_disk_in_gb=disk_gb,
        ports="22/tcp",
        start_ssh=True,
        country_code=country_code,
        cloud_type="SECURE",
    )
    pod_id = pod["id"]
    print(f"Pod created: {pod_id}")
    save_pod_state({"id": pod_id, "gpu": gpu_type})
    info = wait_for_pod(runpod, pod_id, gpu_type)
    if info is None:
        print("Pod failed to come up.")
        sys.exit(1)
    return info


def run_cmd(cmd, check=True, **kwargs):
    return subprocess.run(cmd, check=check, **kwargs)


def setup_pod(ssh_target, ssh_opts, scp_opts, skip_pip):
    """One-time setup on a fresh pod: dirs, code, requirements, weights, pip."""
    # 0. Dirs
    run_cmd(["ssh"] + ssh_opts + [ssh_target,
        "mkdir -p /workspace/scripts/testing /workspace/src/homography "
        "/workspace/models /workspace/clips /workspace/out"])

    # 1. Code files
    print("Uploading code...")
    for rel in CODE_FILES:
        local = os.path.join(PROJECT_ROOT, rel)
        if not os.path.exists(local):
            print(f"  MISSING: {local}")
            sys.exit(1)
        run_cmd(["scp"] + scp_opts + [local, f"{ssh_target}:/workspace/{rel}"])

    # 2. Requirements file
    req_path = os.path.join(PROJECT_ROOT, ".pod_rectify_requirements.txt")
    with open(req_path, "w") as f:
        f.write(REQUIREMENTS)
    run_cmd(["scp"] + scp_opts + [req_path,
                                   f"{ssh_target}:/workspace/requirements.txt"])
    os.remove(req_path)

    # 3. Weights
    print(f"Uploading weights ({os.path.getsize(LOCAL_WEIGHTS)/1024/1024:.0f} MB)...")
    t0 = time.time()
    run_cmd(["scp"] + scp_opts + [
        LOCAL_WEIGHTS,
        f"{ssh_target}:/workspace/models/hrnet_finetuned_last.pth",
    ])
    print(f"  weights uploaded in {time.time()-t0:.0f}s")

    # 4. Install deps
    if not skip_pip:
        print("Installing deps...")
        try:
            run_cmd(["ssh"] + ssh_opts + [ssh_target,
                "cd /workspace && python -m venv venv --system-site-packages "
                "&& source venv/bin/activate "
                "&& pip install -q -r requirements.txt"], check=True)
            return False  # pip succeeded
        except subprocess.CalledProcessError:
            print("  pip install failed — falling back to system python...")
            return True  # skip_pip
    return skip_pip


def rectify_one(ssh_target, ssh_opts, scp_opts, clip, anchor,
                smooth_window, smooth_poly, bank_coast, no_track_bank,
                python_cmd, output_mp4_name):
    """Upload a single clip, run rectify, download result + log."""
    clip_size_mb = os.path.getsize(clip) / 1024 / 1024
    print(f"\n─── {os.path.basename(clip)} ({clip_size_mb:.1f} MB) ───")
    clip_remote = f"/workspace/clips/{os.path.basename(clip)}"
    run_cmd(["scp"] + scp_opts + [clip, f"{ssh_target}:{clip_remote}"])

    remote_out = f"/workspace/out/{output_mp4_name}"
    remote_log = (f"/workspace/out/"
                  f"{os.path.splitext(output_mp4_name)[0]}_pod.log")

    cmd_args = [
        f"--clip {clip_remote}",
        f"--anchor {anchor}",
        f"--device cuda",
        f"--output {remote_out}",
    ]
    if smooth_window:
        cmd_args.append(f"--smooth-window {smooth_window}")
        cmd_args.append(f"--smooth-poly {smooth_poly}")
    if bank_coast:
        cmd_args.append("--bank-coast")
    if no_track_bank:
        cmd_args.append("--no-track-bank")

    print("  running rectify on pod...")
    run_cmd(["ssh"] + ssh_opts + [ssh_target,
        f"cd /workspace && {python_cmd} -u scripts/testing/test_tracker_rectify_clip.py "
        + " ".join(cmd_args) + f" 2>&1 | tee {remote_log}"])

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    local_out = os.path.join(OUTPUT_DIR, output_mp4_name)
    local_log = os.path.join(OUTPUT_DIR,
                              os.path.splitext(output_mp4_name)[0] + "_pod.log")
    run_cmd(["scp"] + scp_opts + [
        f"{ssh_target}:{remote_out}", local_out,
    ])
    run_cmd(["scp"] + scp_opts + [
        f"{ssh_target}:{remote_log}", local_log,
    ])
    print(f"  saved: {local_out}")


def main():
    parser = argparse.ArgumentParser(
        description="Batch rectify clips on a RunPod RTX 5090.",
        epilog="Example: --clips a.mp4 b.mp4 --anchors 40.0 25.0",
    )
    parser.add_argument("--clips", nargs="+", required=True,
                        help="One or more local .mp4 clip paths.")
    parser.add_argument("--anchors", nargs="+", type=float, required=True,
                        help="NGS x for grid_pos 0, one per --clip (same order).")
    parser.add_argument("--output-names", nargs="*", default=None,
                        help="Output MP4 filenames (default: derived from clip).")
    parser.add_argument("--smooth-window", type=int, default=31)
    parser.add_argument("--smooth-poly", type=int, default=3)
    parser.add_argument("--bank-coast", action="store_true")
    parser.add_argument("--no-track-bank", action="store_true")
    parser.add_argument("--gpu-type", default="NVIDIA GeForce RTX 5090")
    parser.add_argument("--country-code", default="US")
    parser.add_argument("--disk-gb", type=int, default=40)
    parser.add_argument("--skip-pip", action="store_true")
    parser.add_argument("--no-terminate", action="store_true")
    args = parser.parse_args()

    if len(args.clips) != len(args.anchors):
        print(f"ERROR: got {len(args.clips)} clips but "
              f"{len(args.anchors)} anchors; they must match 1:1")
        sys.exit(1)
    for c in args.clips:
        if not os.path.exists(c):
            print(f"Clip not found: {c}")
            sys.exit(1)
    if not os.path.exists(LOCAL_WEIGHTS):
        print(f"Weights missing: {LOCAL_WEIGHTS}")
        sys.exit(1)

    # Resolve output names
    out_names = []
    for i, c in enumerate(args.clips):
        if args.output_names and i < len(args.output_names):
            out_names.append(args.output_names[i])
        else:
            out_names.append(
                os.path.basename(os.path.dirname(c)) + "_"
                + os.path.splitext(os.path.basename(c))[0] + "_pod.mp4"
            )

    info = create_pod(gpu_type=args.gpu_type, disk_gb=args.disk_gb,
                      country_code=args.country_code)

    from launch_runpod import load_pod_state
    state = load_pod_state()
    ssh_host = state["ssh_host"]
    ssh_port = state["ssh_port"]
    ssh_user = state["ssh_user"]
    ssh_target = f"{ssh_user}@{ssh_host}"
    ssh_opts = ["-o", "StrictHostKeyChecking=no", "-p", str(ssh_port)]
    scp_opts = ["-o", "StrictHostKeyChecking=no", "-P", str(ssh_port)]

    try:
        skip_pip = setup_pod(ssh_target, ssh_opts, scp_opts, args.skip_pip)
        python_cmd = ("source venv/bin/activate && python" if not skip_pip
                      else "python")

        print(f"\n=== Batch: {len(args.clips)} clip(s) ===")
        for clip, anchor, out_name in zip(args.clips, args.anchors, out_names):
            rectify_one(
                ssh_target, ssh_opts, scp_opts,
                clip=clip, anchor=anchor,
                smooth_window=args.smooth_window,
                smooth_poly=args.smooth_poly,
                bank_coast=args.bank_coast,
                no_track_bank=args.no_track_bank,
                python_cmd=python_cmd,
                output_mp4_name=out_name,
            )

        print("\n=== All clips rendered ===")
    except Exception as e:
        print(f"\n=== Failed: {e} ===")
        raise
    finally:
        if not args.no_terminate:
            print(f"\nTerminating pod {state['id']}...")
            terminate_pod(argparse.Namespace())
        else:
            print(f"Pod left alive: ssh {ssh_user}@{ssh_host} -p {ssh_port}")


if __name__ == "__main__":
    main()
