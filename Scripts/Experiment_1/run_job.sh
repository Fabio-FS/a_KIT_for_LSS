#!/bin/bash
#SBATCH --job-name=lss-onejob
#SBATCH --partition=dev_gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out
## If your site requires an account, uncomment and set:
## #SBATCH -A <your_project_account>

set -euo pipefail

# --- Conda environment ---
module purge
# On UC3, Anaconda3 is available, but Miniforge is usually recommended:
# source /opt/bwhpc/common/devel/miniforge/24.11.0-py3.12/etc/profile.d/conda.sh
module load Anaconda3
eval "$(conda shell.bash hook)"
conda activate "$WORK/conda_envs/lss"

# --- Hugging Face cache (workspace, not $HOME) ---
export HF_HOME="$WORK/hf_cache"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_HUB_OFFLINE=1
mkdir -p "$HF_HOME"

# --- Port for vLLM ---
PORT=8000
MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"

# --- Start vLLM in the background ---
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
trap 'kill $VLLM_PID >/dev/null 2>&1 || true' EXIT

# --- Wait for server ---
echo "Waiting for vLLM on port ${PORT} ..."
for i in {1..30}; do
  if curl -s "http://127.0.0.1:${PORT}/v1/models" >/dev/null; then
    echo "vLLM is ready."
    break
  fi
  sleep 2
  if ! kill -0 $VLLM_PID 2>/dev/null; then
    echo "vLLM crashed, see vllm.log"
    exit 1
  fi
  if [ $i -eq 30 ]; then
    echo "Timed out waiting for vLLM"
    exit 1
  fi
done

# --- Point simulation to local vLLM ---
export API_URL="http://127.0.0.1:${PORT}/v1/chat/completions"
export MODEL="${MODEL_NAME}"
export HF_TOKEN=   # not needed offline

# --- Run directory ---
export RUN_ROOT="$WORK/lss_runs"
mkdir -p "$RUN_ROOT"

# --- Run the simulation ---
srun -n 1 python scripts/experiment_1/RUNME.py
