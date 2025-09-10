# LSS-KIT: Large-Scale Social Simulation Toolkit

A toolkit for simulating social media dynamics using Large Language Models (LLMs) as agent opinion models. The framework supports multiple opinion models and can run distributed simulations on HPC clusters.

## Overview

LSS-KIT simulates social networks where agents:
- Generate posts based on their political views and social context
- Read and evaluate posts from their neighbors
- Update their preferences based on social interactions
- Evolve their behavior over time through reinforcement learning

The system supports both traditional opinion models (BCM) and modern LLM-based agents that generate realistic social media content.

## Project Structure

```
LSS-KIT/
├── src/
│   ├── simulation.py          # Main simulation engine
│   ├── interfaces.py          # Model dispatch interface
│   ├── save.py               # Data persistence (with secret filtering)
│   ├── executors/
│   │   └── hf.py             # Async LLM execution
│   └── models/
│       └── llm/              # LLM opinion model
│           ├── agent.py      # Agent behaviors
│           └── utlities.py   # Prompt composition
├── Scripts/
│   └── Experiment_1/         # Example experiment
│       ├── RUNME.py          # Main experiment script
│       ├── PARAMS.json       # Experiment parameters
│       ├── prompts.yaml      # LLM prompts
│       ├── run_job.sh        # SLURM batch script
│       └── .env              # API credentials (create this)
└── notebooks/                # Analysis notebooks
```

## Installation

### Local Development

1. Create conda environment with Python 3.11:
```bash
conda create -n lss python=3.11 pip
conda activate lss
```

2. Install dependencies:
```bash
pip install torch torchvision torchaudio
pip install igraph numpy aiohttp pyyaml python-dotenv requests transformers huggingface_hub vllm
```

### HPC Cluster Setup

1. Clone the repository:
```bash
git clone https://github.com/Fabio-FS/a_KIT_for_LSS.git
cd a_KIT_for_LSS
```

2. Create conda environment in your workspace:
```bash
conda create -p $WORK/conda_envs/lss python=3.11 pip
conda activate $WORK/conda_envs/lss
pip install vllm transformers huggingface_hub igraph numpy aiohttp pyyaml python-dotenv requests

pip install accelerate
```

3. Cache the model (run once on login node):
```bash
export HF_HOME=$WORK/hf_cache
python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('meta-llama/Llama-3.1-8B-Instruct')"
```

## Configuration

### Environment Variables

Create `Scripts/Experiment_1/.env`:
```bash
HF_TOKEN=your_huggingface_token_here
API_URL=http://127.0.0.1:8000/v1/chat/completions
MODEL=meta-llama/Llama-3.1-8B-Instruct
```

### Experiment Parameters

Edit `Scripts/Experiment_1/PARAMS.json`:
```json
{
  "replicas": 2,
  "num_agents": 16,
  "neighbors": 4,
  "timesteps": 10,
  "fill_history": 3,
  "opinion_model": "LLM",
  "rewiring_p": 0.05,
  "post_read_per_round": 2
}
```

## Usage

### Local Execution

```bash
cd Scripts/Experiment_1
python RUNME.py
```

### HPC Cluster Execution

Submit to SLURM scheduler:
```bash
cd Scripts/Experiment_1
sbatch run_job.sh
```

The cluster script will:
1. Allocate a GPU node
2. Start vLLM server locally
3. Run the simulation using the local LLM
4. Save results to `$WORK/lss_runs`

## How It Works

### Simulation Flow

1. **Initialization**: Create network topology (Watts-Strogatz) and initialize agents
2. **Thermalization**: Pre-generate initial posts for system warm-up
3. **Main Loop**: For each timestep:
   - Calculate post weights based on success, recency, and personal preferences
   - Agents read top-k posts from neighbors
   - Agents evaluate posts (like/dislike decisions)
   - Update agent success and neighbor preferences
   - Generate new posts based on reading history and agent identity

### LLM Agent Behavior

Agents have political identities (far left, left, center, right, far right) and:
- Generate posts reflecting their political views
- Evaluate posts from neighbors based on alignment with their beliefs
- Maintain memory of recent interactions
- Adapt their neighbor preferences based on successful interactions

### Data Persistence

Results are saved with automatic secret filtering:
- **Arrays**: Agent weights, reading matrices, like counts (`.npz` format)
- **Posts**: All generated text content (`.csv` format)
- **Metadata**: Experiment parameters and run information (`.json` format)
- **Manifest**: Overview of all files and experiment metadata

## Key Features

- **Modular Architecture**: Easy to add new opinion models
- **Async LLM Execution**: Efficient parallel processing of language model calls
- **HPC Ready**: SLURM integration with local GPU inference
- **Secret Protection**: Automatic filtering of API tokens from saved data
- **Reproducible**: Full parameter tracking and random seed control

## Output Format

Results are organized as:
```
runs/
└── experiment_id/
    ├── manifest.json         # Experiment overview
    ├── r000/                 # Replica 0
    │   ├── arrays.npz        # Numerical data
    │   ├── posts.csv         # Generated content
    │   └── meta.json         # Replica metadata
    └── r001/                 # Replica 1
        └── ...
```

## Analysis

Use the provided loader functions to analyze results:
```python
from src.save import load_manifest, load_replica

# Load experiment metadata
manifest = load_manifest("runs/experiment_id")

# Load specific replica
data = load_replica("runs/experiment_id/r000")
weights = data["WEIGHTS"]
posts = data["POSTS_long"]  # [(agent_id, timestep, text), ...]
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all experiments reproduce correctly
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this toolkit in your research, please cite:

```bibtex
@software{lss_kit,
  title={LSS-KIT: Large-Scale Social Simulation Toolkit},
  author={Your Name},
  year={2025},
  url={https://github.com/Fabio-FS/a_KIT_for_LSS}
}
```