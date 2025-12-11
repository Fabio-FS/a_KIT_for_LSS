```
# Project Structure

├── .env
├── README.md
├── Scripts
│   └── Experiment_1
│       ├── .env
│       ├── 1024_Realistic_Usernames.csv
│       ├── PARAMS.json
│       ├── RUNME.py
│       ├── prompts.yaml
│       ├── run_job copy.sh
│       └── run_job.sh
├── notebooks
│   ├── .env
│   ├── 1024_Realistic_Usernames.csv
│   ├── PARAMS.json
│   ├── debug_Dez.ipynb
│   ├── metrics.py
│   ├── prompt_play.ipynb
│   ├── prompting101.ipynb
│   ├── prompts.yaml
│   ├── restart_25.ipynb
│   └── runs
│       ├── Exp_W_personal
│       │   ├── manifest.json
│       │   └── r000
│       │       ├── arrays.npz
│       │       ├── graph_data.json
│       │       ├── meta.json
│       │       └── posts.csv
│       ├── Exp_W_post
│       │   ├── manifest.json
│       │   └── r000
│       │       ├── arrays.npz
│       │       ├── graph_data.json
│       │       ├── meta.json
│       │       └── posts.csv
│       ├── Exp_W_success
│       │   ├── manifest.json
│       │   └── r000
│       │       ├── arrays.npz
│       │       ├── graph_data.json
│       │       ├── meta.json
│       │       └── posts.csv
│       ├── Exp_W_success_1
│       │   ├── manifest.json
│       │   └── r000
│       │       ├── arrays.npz
│       │       ├── graph_data.json
│       │       ├── meta.json
│       │       └── posts.csv
│       ├── Exp_W_success_GEMMA
│       │   ├── manifest.json
│       │   └── r000
│       │       ├── arrays.npz
│       │       ├── graph_data.json
│       │       ├── meta.json
│       │       └── posts.csv
│       └── Exp_W_success_OPENGPT
│           ├── manifest.json
│           └── r000
│               ├── arrays.npz
│               ├── graph_data.json
│               ├── meta.json
│               └── posts.csv
├── print_tree.ipynb
└── src
    ├── .env
    ├── __init__.py
    ├── executors
    │   └── hf.py
    ├── interfaces.py
    ├── models
    │   ├── __init__.py
    │   └── llm
    │       ├── __init__.py
    │       ├── agent.py
    │       └── utlities.py
    ├── save.py
    ├── simulation.py
    └── visualization.py
```