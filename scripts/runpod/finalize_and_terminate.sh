#!/usr/bin/env bash
# Autonomous orchestration for unattended training finish + pod cleanup.
#
# Reads pod info from .runpod_pod.json (written by launch_runpod.py).
# Polls the pod until the training process exits (or a timeout elapses),
# SCPs weights + log out, installs weights into models/, and ALWAYS
# terminates the pod at the end (even on upstream error).
#
# Usage:
#   scripts/runpod/finalize_and_terminate.sh <name-prefix> [timeout-hours]
#
# Example:
#   scripts/runpod/finalize_and_terminate.sh hrnet_w18_hash_round1 3
#
# Output files (relative to repo root):
#   output/<name-prefix>/weights/{best.pth,last.pth,train.log}
#   output/<name-prefix>/finalize.log
#   models/<name-prefix>_best.pth
#   models/<name-prefix>_last.pth

set -u

if [[ $# -lt 1 ]]; then
    echo "usage: $0 <name-prefix> [timeout-hours]" >&2
    exit 2
fi
NAME_PREFIX="$1"
TIMEOUT_HOURS="${2:-3}"

ROOT="/Users/aldenkling/Desktop/Personal Research/cv-player-tracking-all22"
cd "$ROOT"

POD_STATE="$ROOT/.runpod_pod.json"
if [[ ! -f "$POD_STATE" ]]; then
    echo "no pod state at $POD_STATE — is a pod actually running?" >&2
    exit 2
fi

POD_ID=$("$ROOT/.venv/bin/python" -c "import json; print(json.load(open('$POD_STATE'))['id'])")
SSH_HOST=$("$ROOT/.venv/bin/python" -c "import json; print(json.load(open('$POD_STATE'))['ssh_host'])")
SSH_PORT=$("$ROOT/.venv/bin/python" -c "import json; print(json.load(open('$POD_STATE'))['ssh_port'])")

WORK_DIR="$ROOT/output/$NAME_PREFIX"
WEIGHTS_DIR="$WORK_DIR/weights"
LOG="$WORK_DIR/finalize.log"
mkdir -p "$WEIGHTS_DIR"

SSH_OPTS=(-o StrictHostKeyChecking=no -o BatchMode=yes -p "$SSH_PORT" "root@$SSH_HOST")
SCP_OPTS=(-o StrictHostKeyChecking=no -o BatchMode=yes -P "$SSH_PORT")

echo "=== finalize start $(date) ===" >> "$LOG"
echo "pod=$POD_ID  host=$SSH_HOST:$SSH_PORT  name=$NAME_PREFIX  timeout=${TIMEOUT_HOURS}h" >> "$LOG"

# --- Always-terminate guarantee -----------------------------------------------
# Use a trap so that ANY exit (normal, signal, error) still terminates the pod.
terminate_pod() {
    echo "$(date): terminating pod $POD_ID..." >> "$LOG"
    # Use the repo's .env explicitly — `load_dotenv()` without a path calls
    # `find_dotenv()` which walks stack frames and fails under `python -`.
    "$ROOT/.venv/bin/python" -c "
import os
from dotenv import load_dotenv
load_dotenv('$ROOT/.env')
import runpod
runpod.api_key = os.environ['RUNPOD_API_KEY']
try:
    runpod.terminate_pod('$POD_ID')
    print('terminated $POD_ID')
except Exception as e:
    print(f'terminate error: {e}')
" >> "$LOG" 2>&1
    echo "=== finalize done $(date) ===" >> "$LOG"
}
trap terminate_pod EXIT

# --- Wait for training to finish ----------------------------------------------
# Initial grace so Python + DataLoader workers have time to spin up. Without
# this, a transient "no process yet" pgrep read on the very first iteration
# would make us conclude training crashed before it actually started.
echo "$(date): initial grace sleep (120s)..." >> "$LOG"
sleep 120

deadline=$(( $(date +%s) + TIMEOUT_HOURS * 3600 ))
while :; do
    if [[ $(date +%s) -ge $deadline ]]; then
        echo "$(date): TIMEOUT — forcing finish" >> "$LOG"
        break
    fi
    # Two completion signals:
    #   (a) train.log ends with "Done. Best recall:" — definitive success marker
    #   (b) pgrep finds no matching process — bracket trick avoids matching
    #       the remote shell's own argv. Without `[t]`, pgrep always finds
    #       itself since our query string appears in its own cmdline.
    # Both HRNet ("Done. Best recall:") and UNet ("Done. Best mean_f1:")
    # print a "Done. Best " line at natural exit. Match the prefix so the
    # same finalizer works for both training types.
    if ssh "${SSH_OPTS[@]}" "tail -5 /workspace/train.log 2>/dev/null | grep -q '^Done. Best'" 2>> "$LOG"; then
        echo "$(date): training completion marker found" >> "$LOG"
        break
    fi
    # Two separate bracket-trick pgrep calls, ORed. Avoids ERE ambiguity in
    # pgrep's default pattern mode (parentheses / alternation aren't reliable
    # across procps versions when sent through ssh quoting layers).
    running=$(ssh "${SSH_OPTS[@]}" "pgrep -f '[t]rain_hrnet_keypoints.py'; pgrep -f '[t]rain_unet_lines.py'; true" 2>> "$LOG" | grep -v '^$')
    if [[ -z "$running" ]]; then
        echo "$(date): no training process running (and no completion marker — may have crashed)" >> "$LOG"
        break
    fi
    echo "$(date): training still running (pid=$running)" >> "$LOG"
    sleep 60
done

# Small buffer for the final save + log flush on the pod
sleep 20

# --- Download weights + log ---------------------------------------------------
for FILE in best.pth last.pth; do
    echo "$(date): downloading $FILE..." >> "$LOG"
    scp "${SCP_OPTS[@]}" "root@$SSH_HOST:/workspace/output/$FILE" "$WEIGHTS_DIR/$FILE" 2>> "$LOG" || \
        echo "  download failed: $FILE" >> "$LOG"
done
scp "${SCP_OPTS[@]}" "root@$SSH_HOST:/workspace/train.log" "$WEIGHTS_DIR/train.log" 2>> "$LOG" || \
    echo "  download failed: train.log" >> "$LOG"

# --- Install into models/ -----------------------------------------------------
mkdir -p "$ROOT/models"
if [[ -f "$WEIGHTS_DIR/best.pth" ]]; then
    cp "$WEIGHTS_DIR/best.pth" "$ROOT/models/${NAME_PREFIX}_best.pth"
    echo "$(date): installed models/${NAME_PREFIX}_best.pth" >> "$LOG"
fi
if [[ -f "$WEIGHTS_DIR/last.pth" ]]; then
    cp "$WEIGHTS_DIR/last.pth" "$ROOT/models/${NAME_PREFIX}_last.pth"
    echo "$(date): installed models/${NAME_PREFIX}_last.pth" >> "$LOG"
fi

# EXIT trap will now terminate the pod
