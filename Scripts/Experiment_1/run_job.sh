#!/bin/bash
#SBATCH --job-name=lss-onejob
#SBATCH --partition=dev_gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out

set -euo pipefail

# --- Paths & env (adjust to your workspace) ---
module purge
module load Anaconda3

# your conda env with vllm + deps installed
source activate $WORK/conda_envs/lss

# keep HF cache in workspace (pre-populate on a login node once)
export HF_HOME=$WORK/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export HF_HUB_OFFLINE=1

# pick a free localhost port on the node
PORT=8000

# --- Start vLLM server on the allocated GPU, in the background ---
# (First time only, ensure vllm is installed in the env: `pip install vllm` on login node)
MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"

# run the server as a Slurm step so it’s tracked & gets the GPU
srun --gpu-bind=none --exclusive -n 1 bash -lc "
  vllm serve ${MODEL_NAME} \
    --host 127.0.0.1 --port ${PORT} \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 8192 \
    --download-dir '${HF_HOME}' \
    > vllm.log 2>&1
" &

VLLM_PID=$!

# ensure we clean up the background server on exit
cleanup() { kill $VLLM_PID >/dev/null 2>&1 || true; }
trap cleanup EXIT

# --- Wait for server to be ready (health check loop) ---
echo "Waiting for vLLM to come up on :$PORT ..."
for i in {1..30}; do
  if curl -s "http://127.0.0.1:${PORT}/v1/models" >/dev/null; then
    echo "vLLM is ready."
    break
  fi
  sleep 2
  if ! kill -0 $VLLM_PID 2>/dev/null; then
    echo "vLLM process died early. See vllm.log"; exit 1
  fi
  if [ $i -eq 30 ]; then
    echo "Timed out waiting for vLLM. See vllm.log"; exit 1
  fi
done

# --- Point your app to the local server (override .env) ---
export API_URL="http://127.0.0.1:${PORT}/v1/chat/completions"
export MODEL="${MODEL_NAME}"
export HF_TOKEN=   # not needed offline

# Optional: keep outputs off $HOME
export RUN_ROOT="${WORK}/lss_runs"
mkdir -p "$RUN_ROOT"

# --- Run your simulation (same GPU node) ---
srun -n 1 python scripts/experiment_1/RUNME.py