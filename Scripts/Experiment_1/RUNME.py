#!/usr/bin/env python3
import sys, os, json, numpy as np, asyncio
import time
from dotenv import dotenv_values

# ensure src/ is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.simulation import run_simulation
from src.save import save_replicas_raw

def _get(k, env, default=None):
    # prefer process env (sbatch can set these), else .env, else default
    return os.environ.get(k, env.get(k, default))

async def main():
    start_time = time.time()
    print(f"Simulation started at: {time.strftime('%H:%M:%S')}")
    # load PARAMS.json from the current dir
    with open("PARAMS.json", "r") as f:
        PARAMS = json.load(f)

    # knobs
    PARAMS["time_decay_rate"] = np.log(2) / 2
    PARAMS["W_agent_success"] = 1
    PARAMS["W_personal_weights"] = 1
    PARAMS["W_post_success"] = 1
    PARAMS["noise_level"] = 0.01

    # .env (local dev) + allow sbatch to override via environment
    ENV = dotenv_values("./.env")
    PARAMS["API_URL"] = _get("API_URL", ENV, "")
    PARAMS["MODEL"]   = _get("MODEL",   ENV, "meta-llama/Llama-3.1-8B-Instruct")
    PARAMS["HF_TOKEN"]= _get("HF_TOKEN",ENV, "")

    print("API_URL:", PARAMS["API_URL"] or "(empty)")
    print("MODEL:  ", PARAMS["MODEL"])

    # small quick-test defaults – change in PARAMS.json for real runs
    PARAMS["num_agents"] = int(PARAMS.get("num_agents", 6))
    PARAMS["timesteps"]  = int(PARAMS.get("timesteps", 2))
    PARAMS["fill_history"] = int(PARAMS.get("fill_history", 1))

    # run
    sim_start = time.time()
    print(f"Starting LLM simulation at: {time.strftime('%H:%M:%S')}")
    List_of_WEIGHTS, List_of_READ_MATRIX, List_of_LIKES, List_of_POSTS = await run_simulation(PARAMS)
    sim_end = time.time()
    print(f"Simulation completed at: {time.strftime('%H:%M:%S')}")
    print(f"Pure simulation time: {sim_end - sim_start:.1f} seconds")
    
    

    # save under RUN_ROOT if provided (cluster), else local "runs"
    out_root = os.environ.get("RUN_ROOT", "runs")
    os.makedirs(out_root, exist_ok=True)
    save_replicas_raw(
        out_root,
        List_of_WEIGHTS, List_of_READ_MATRIX, List_of_LIKES, List_of_POSTS,
        PARAMS,
        extra_meta={"note": "raw generation only, no metrics"},
    )
    total_time = time.time() - start_time
    print(f"Total script time: {total_time:.1f} seconds")

if __name__ == "__main__":
    asyncio.run(main())