#!/usr/bin/env python3
"""Launch AL Round 2 scoring + preannotation on RunPod RTX 5090.

Workflow:
  1. Create RTX 5090 pod.
  2. Upload candidates tarball, weights, scripts, requirements.
  3. Install deps via venv.
  4. Run scoring (select top N from candidates).
  5. Run preannotation on the selected frames using the grid solver.
  6. Download only the three small result files: manifest.json, scores.csv,
     ls_import.json.
  7. Leave pod alive on failure (for debugging), terminate on success.

Not a general-purpose launcher — hardcoded for the al_round2 layout.
"""

import argparse
import os
import subprocess
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts", "runpod"))

from launch_runpod import (  # noqa: E402
    load_api_key,
    wait_for_pod,
    save_pod_state,
    terminate_pod,
)

LOCAL_CANDIDATES_DIR = os.path.join(
    PROJECT_ROOT, "data", "field_keypoints", "al_round2", "candidates",
)
LOCAL_CANDIDATES_MANIFEST = os.path.join(
    PROJECT_ROOT, "data", "field_keypoints", "al_round2", "candidates_manifest.json",
)
LOCAL_WEIGHTS = os.path.join(PROJECT_ROOT, "models", "hrnet_finetuned_last.pth")
LOCAL_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "field_keypoints", "al_round2")

FILES_TO_UPLOAD = [
    # (local_path_rel_project, remote_path_rel_workspace)
    ("scripts/data_prep/select_active_learning_frames.py",
     "scripts/data_prep/select_active_learning_frames.py"),
    ("scripts/data_prep/preannotate_keypoints.py",
     "scripts/data_prep/preannotate_keypoints.py"),
    ("scripts/testing/test_yard_line_grouping.py",
     "scripts/testing/test_yard_line_grouping.py"),
    ("src/homography/keypoint_detector.py",
     "src/homography/keypoint_detector.py"),
    ("src/homography/keypoint_schema.py",
     "src/homography/keypoint_schema.py"),
    ("src/homography/field_model.py",
     "src/homography/field_model.py"),
    ("src/homography/__init__.py",
     "src/homography/__init__.py"),
    ("src/__init__.py",
     "src/__init__.py"),
    ("scripts/training/requirements_hrnet_runpod.txt",
     "requirements.txt"),
]


def create_pod(gpu_type="NVIDIA GeForce RTX 5090", disk_gb=80, country_code="US"):
    import runpod
    runpod.api_key = load_api_key()
    print(f"Creating pod: {gpu_type}, {disk_gb}GB disk, country={country_code}...")
    pod = runpod.create_pod(
        name="all22-al-r2-scoring",
        image_name="runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04",
        gpu_type_id=gpu_type,
        gpu_count=1,
        volume_in_gb=0,
        container_disk_in_gb=disk_gb,
        ports="22/tcp",
        start_ssh=True,
        country_code=country_code,
    )
    pod_id = pod["id"]
    print(f"Pod created: {pod_id}")
    save_pod_state({"id": pod_id, "gpu": gpu_type})
    info = wait_for_pod(runpod, pod_id, gpu_type)
    if info is None:
        print("Pod failed to come up. Terminating and retrying would be needed.")
        sys.exit(1)
    return info


def run_cmd(cmd, check=True, capture=False, input_text=None):
    if capture:
        return subprocess.run(
            cmd, capture_output=True, text=True, check=check, input=input_text,
        )
    return subprocess.run(cmd, check=check, input=input_text, text=True if input_text else None)


def bench_scp(ssh_target, ssh_opts, base_scp_opts, min_ok_mb_s=5.0):
    """Benchmark scp upload with different ciphers. Returns the best cipher opts
    (list of extra scp args), or None if nothing meets min_ok_mb_s."""
    import tempfile
    print("Benchmarking scp throughput (50 MB test)...", flush=True)
    with tempfile.NamedTemporaryFile(delete=False) as tf:
        test_path = tf.name
    try:
        subprocess.run(["dd", "if=/dev/urandom", f"of={test_path}",
                        "bs=1m", "count=50"],
                       capture_output=True, check=True)
        candidates = [
            ("default", []),
            ("aes128-gcm", ["-c", "aes128-gcm@openssh.com"]),
            ("chacha20", ["-c", "chacha20-poly1305@openssh.com"]),
        ]
        best = None
        best_rate = 0.0
        for label, extra in candidates:
            t0 = time.time()
            try:
                r = subprocess.run(
                    ["scp"] + base_scp_opts + extra +
                    [test_path, f"{ssh_target}:/tmp/al_scp_bench"],
                    capture_output=True, text=True, timeout=60,
                )
            except subprocess.TimeoutExpired:
                dt = time.time() - t0
                print(f"  {label}: timed out after {dt:.0f}s (<0.85 MB/s)",
                      flush=True)
                subprocess.run(
                    ["pkill", "-f", "al_scp_bench"], capture_output=True,
                )
                continue
            dt = time.time() - t0
            if r.returncode != 0:
                print(f"  {label}: FAILED ({r.stderr[:100]})", flush=True)
                continue
            rate = 50.0 / dt
            print(f"  {label}: {rate:.2f} MB/s ({dt:.1f}s)", flush=True)
            if rate > best_rate:
                best_rate = rate
                best = extra
            subprocess.run(
                ["ssh"] + ssh_opts + [ssh_target, "rm -f /tmp/al_scp_bench"],
                capture_output=True,
            )
            # If default is already great, no need to try more
            if best_rate >= 20.0:
                break
        return best, best_rate
    finally:
        os.remove(test_path)


def upload_and_run(ssh_target, ssh_opts, scp_opts, skip_pip=False, extra_scp_opts=None):
    if extra_scp_opts is None:
        extra_scp_opts = []
    full_scp = scp_opts + extra_scp_opts

    # 0. Ensure workspace dirs exist
    print("Preparing workspace dirs...")
    run_cmd(["ssh"] + ssh_opts + [ssh_target,
        "mkdir -p /workspace/scripts/data_prep /workspace/scripts/testing "
        "/workspace/src/homography /workspace/models /workspace/al_round2"])

    # 1. Upload scripts
    print("Uploading scripts...")
    for local_rel, remote_rel in FILES_TO_UPLOAD:
        local_path = os.path.join(PROJECT_ROOT, local_rel)
        if not os.path.exists(local_path):
            print(f"  ERROR: missing local file {local_path}")
            sys.exit(1)
        run_cmd(["scp"] + full_scp + [
            local_path, f"{ssh_target}:/workspace/{remote_rel}",
        ])

    # 2. Upload weights
    print("Uploading HRNet weights...")
    w_size_mb = os.path.getsize(LOCAL_WEIGHTS) / 1024 / 1024
    print(f"  {w_size_mb:.0f} MB")
    run_cmd(["scp"] + full_scp + [
        LOCAL_WEIGHTS, f"{ssh_target}:/workspace/models/hrnet_finetuned_last.pth",
    ])

    # 3. Upload candidates manifest
    print("Uploading candidates manifest...")
    run_cmd(["scp"] + full_scp + [
        LOCAL_CANDIDATES_MANIFEST,
        f"{ssh_target}:/workspace/al_round2/candidates_manifest.json",
    ])

    # 4. Pack + upload candidates (971MB of JPEGs, poorly-compressible)
    print("Packing candidates tarball...")
    tar_path = os.path.join(PROJECT_ROOT, ".al_candidates_upload.tar")
    run_cmd([
        "tar", "-cf", tar_path,
        "-C", os.path.dirname(LOCAL_CANDIDATES_DIR),
        "candidates",
    ])
    tar_size_mb = os.path.getsize(tar_path) / 1024 / 1024
    print(f"  {tar_size_mb:.0f} MB — uploading...")
    t0 = time.time()
    run_cmd(["scp"] + full_scp + [
        tar_path, f"{ssh_target}:/workspace/al_round2/candidates.tar",
    ])
    print(f"  uploaded in {time.time() - t0:.0f}s")
    os.remove(tar_path)

    print("Extracting on pod...")
    run_cmd(["ssh"] + ssh_opts + [ssh_target,
        "cd /workspace/al_round2 && tar -xf candidates.tar && rm candidates.tar"])

    # 5. Install deps (may fail per memory note; will give up and continue if
    # system torch is present)
    if not skip_pip:
        print("Installing deps (may take a few minutes)...")
        try:
            run_cmd(["ssh"] + ssh_opts + [ssh_target,
                "cd /workspace && python -m venv venv --system-site-packages "
                "&& source venv/bin/activate "
                "&& pip install -q -r requirements.txt"], check=True)
        except subprocess.CalledProcessError:
            print("  pip install failed — falling back to system python.")
            skip_pip = True

    python_cmd = ("source venv/bin/activate && python" if not skip_pip
                  else "python")

    # 6. Run scoring
    print("\n=== Running scoring phase on pod ===")
    score_cmd = (
        f"cd /workspace && {python_cmd} scripts/data_prep/select_active_learning_frames.py "
        "--phase score "
        "--weights models/hrnet_finetuned_last.pth "
        "--output-dir al_round2 "
        "--device cuda "
        "--n-select 300 "
        "2>&1 | tee al_round2/score.log"
    )
    run_cmd(["ssh"] + ssh_opts + [ssh_target, score_cmd])

    # 7. Run preannotation (uses new grid solver)
    print("\n=== Running preannotation on pod ===")
    # ls_path_prefix lives relative to LS's LOCAL_FILES_DOCUMENT_ROOT
    # (data/field_keypoints). We're serving directly from candidates/ to avoid
    # a separate images/ folder download, so prefix = al_round2/candidates.
    preannotate_cmd = (
        f"cd /workspace && {python_cmd} scripts/data_prep/preannotate_keypoints.py "
        "--image-dir al_round2/images "
        "--weights models/hrnet_finetuned_last.pth "
        "--output al_round2/ls_import.json "
        "--ls-path-prefix al_round2/candidates "
        "--device cuda "
        "2>&1 | tee al_round2/preannotate.log"
    )
    run_cmd(["ssh"] + ssh_opts + [ssh_target, preannotate_cmd])

    # 8. Download results (tiny)
    print("\n=== Downloading result files ===")
    os.makedirs(LOCAL_OUTPUT_DIR, exist_ok=True)
    for fname in ["manifest.json", "scores.csv", "ls_import.json",
                  "score.log", "preannotate.log"]:
        run_cmd(["scp"] + full_scp + [
            f"{ssh_target}:/workspace/al_round2/{fname}",
            os.path.join(LOCAL_OUTPUT_DIR, fname),
        ])
        print(f"  downloaded {fname}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-type", default="NVIDIA GeForce RTX 5090")
    parser.add_argument("--disk-gb", type=int, default=80)
    parser.add_argument("--country-code", default="US",
                        help="Force a specific country for the pod (default: US).")
    parser.add_argument("--skip-bench", action="store_true",
                        help="Skip the scp speed-test gate and just upload.")
    parser.add_argument("--skip-pip", action="store_true",
                        help="Skip pip install, use system python.")
    parser.add_argument("--no-terminate", action="store_true",
                        help="Leave pod alive after completion (for debugging).")
    args = parser.parse_args()

    # Sanity check local files
    if not os.path.isdir(LOCAL_CANDIDATES_DIR):
        print(f"ERROR: {LOCAL_CANDIDATES_DIR} does not exist. Run extract first.")
        sys.exit(1)
    if not os.path.exists(LOCAL_WEIGHTS):
        print(f"ERROR: {LOCAL_WEIGHTS} missing.")
        sys.exit(1)

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
        if args.skip_bench:
            print("Skipping bench gate (--skip-bench).")
            best_extra = []
        else:
            best_extra, best_rate = bench_scp(ssh_target, ssh_opts, scp_opts)
            if best_extra is None or best_rate < 5.0:
                print(f"\nBest scp rate: {best_rate:.2f} MB/s — too slow to proceed.")
                print(f"Terminating pod {state['id']}.")
                terminate_pod(argparse.Namespace())
                sys.exit(2)
            label = ' '.join(best_extra) if best_extra else 'default'
            print(f"Using scp options: {label} (~{best_rate:.1f} MB/s)\n")
        upload_and_run(ssh_target, ssh_opts, scp_opts,
                       skip_pip=args.skip_pip, extra_scp_opts=best_extra)
        print("\n=== Success ===")
        if not args.no_terminate:
            print("Terminating pod...")
            terminate_pod(argparse.Namespace())
        else:
            print(f"Pod left alive. Terminate manually or re-run --terminate.")
    except Exception as e:
        print(f"\n=== Run failed: {e} ===")
        print(f"Terminating pod {state['id']} to avoid billing...")
        try:
            terminate_pod(argparse.Namespace())
        except Exception as term_e:
            print(f"WARNING: terminate also failed: {term_e}")
            print(f"  Manually terminate via RunPod dashboard: pod {state['id']}")
        raise


if __name__ == "__main__":
    main()
