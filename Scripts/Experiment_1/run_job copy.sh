#!/bin/bash
#SBATCH --job-name=lss-onejob
#SBATCH --partition=gpu_a100_short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --output=%x-%j.out
#SBATCH -A ka

set -euo pipefail

# --- Conda ---
source /opt/bwhpc/common/devel/miniforge/24.11.0-py3.12/etc/profile.d/conda.sh
eval "$(conda shell.bash hook)"
conda activate lss

# --- Cache roots (prefer UC3 workspace) ---
export HF_HOME="$(ws_find myhf)/hf_cache"
mkdir -p "$HF_HOME"
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_HUB_OFFLINE=1

# (optional) stage to node-local SSD for speed
if [[ -n "${TMPDIR:-}" ]]; then
  mkdir -p "$TMPDIR/hf"
  rsync -a --delete "$HF_HOME/" "$TMPDIR/hf/" || true
  export TRANSFORMERS_CACHE="$TMPDIR/hf"
fi

# --- Locate a local snapshot of Llama 3.1 8B (workspace first, then common fallbacks) ---
MODEL_SNAPSHOT_DIR="$(
python3 - <<'PY'
import os, glob
cands = []
env_hf = os.environ.get("HF_HOME","")
if env_hf:
    cands.append(os.path.join(env_hf, "hub", "models--meta-llama--Llama-3.1-8B-Instruct", "snapshots", "*"))
# fallbacks: user caches
home = os.path.expanduser("~")
cands.append(os.path.join(home, ".cache", "huggingface", "hub", "models--meta-llama--Llama-3.1-8B-Instruct", "snapshots", "*"))
for pat in cands:
    snaps = sorted(glob.glob(pat), key=os.path.getmtime, reverse=True)
    if snaps:
        print(snaps[0])
        break
PY
)"

if [[ -z "${MODEL_SNAPSHOT_DIR}" || ! -d "${MODEL_SNAPSHOT_DIR}" ]]; then
  echo "ERROR: No local snapshot for meta-llama/Llama-3.1-8B-Instruct found."
  echo "Searched:"
  echo "  $HF_HOME/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/*"
  echo "  $HOME/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/*"
  echo "List of $HF_HOME/hub (for debugging):"
  ls -al "$HF_HOME/hub" || true
  exit 1
fi

echo "Using local model snapshot: $MODEL_SNAPSHOT_DIR"

# --- vLLM ---
PORT=8000
srun --gpu-bind=none --exclusive -n 1 bash -lc "
  vllm serve '${MODEL_SNAPSHOT_DIR}' \
    --tokenizer '${MODEL_SNAPSHOT_DIR}' \
    --host 127.0.0.1 --port ${PORT} \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 8192 \
    --download-dir '${HF_HOME}' --trust-remote-code \
    > vllm.log 2>&1
" &

VLLM_PID=$!
trap 'kill $VLLM_PID >/dev/null 2>&1 || true' EXIT

echo "Waiting for vLLM on :$PORT ..."
for i in {1..500}; do
  if curl -s "http://127.0.0.1:${PORT}/v1/models" >/dev/null; then
    echo "vLLM is ready."
    break
  fi
  sleep 2
  if ! kill -0 $VLLM_PID 2>/dev/null; then
    echo "vLLM process died early. See vllm.log"; exit 1
  fi
  if [ $i -eq 500 ]; then
    echo "Timed out waiting for vLLM. See vllm.log"; exit 1
  fi
done

export API_URL="http://127.0.0.1:${PORT}/v1/chat/completions"
export MODEL="${MODEL_SNAPSHOT_DIR}"
export HF_TOKEN=

export RUN_ROOT="$PWD/runs"
mkdir -p "$RUN_ROOT"

python RUNME.py