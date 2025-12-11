#!/bin/bash
#SBATCH --job-name=lss_experiments
#SBATCH --partition=gpu_a100_il
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=0-3
#SBATCH --output=logs/exp_%A_%a.out

set -euo pipefail

# Setup environment
export HF_HOME=$(ws_find llm_models)/hf_cache
export TRANSFORMERS_CACHE="$HF_HOME"
cd $(ws_find llm_models)

# Array of parameter files
PARAMS_FILES=(
    "PARAMS_control.json"
    "PARAMS_agent_success.json"
    "PARAMS_post_success.json"
    "PARAMS_personal_pref.json"
)

# Get parameter file for this array task
PARAMS_FILE=${PARAMS_FILES[$SLURM_ARRAY_TASK_ID]}

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Experiment: ${PARAMS_FILE}"
echo "Start time: $(date)"
echo "=========================================="

# Start timer
START_TIME=$(date +%s)

# Run experiment inside singularity
singularity exec --nv \
    --bind $(ws_find llm_models):$(ws_find llm_models) \
    vllm.sif \
    python3 Scripts/Experiment_Comparison/run_experiment.py Scripts/Experiment_Comparison/${PARAMS_FILE}

# End timer
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo "=========================================="
echo "End time: $(date)"
echo "Elapsed time: ${ELAPSED} seconds ($((ELAPSED / 60)) minutes)"
echo "=========================================="