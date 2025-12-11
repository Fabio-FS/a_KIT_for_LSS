import json
import os
import sys
import asyncio
from src.simulation import run_simulation
from src.save import save_simulation_results

def main():
    if len(sys.argv) != 2:
        print("Usage: python run_experiment.py <params_file>")
        sys.exit(1)
    
    params_file = sys.argv[1]
    
    # Load PARAMS
    with open(params_file, "r") as f:
        PARAMS = json.load(f)
    
    # Set API URL for local vLLM server (no token needed)
    PARAMS["API_URL"] = "http://localhost:8000/v1/chat/completions"
    PARAMS["HF_TOKEN"] = ""  # Empty string, won't be used
    
    # Extract experiment name from params file
    exp_name = os.path.basename(params_file).replace("PARAMS_", "").replace(".json", "")
    out_root = f"../runs/{exp_name}"
    
    print(f"Running experiment: {exp_name}")
    print(f"Output directory: {out_root}")
    
    # Run simulation
    results = asyncio.run(run_simulation(PARAMS))
    
    # Unpack results
    List_of_WEIGHTS, List_of_READ_MATRIX, List_of_LIKES, List_of_POSTS, List_of_GRAPHS, List_of_INDIVIDUAL_LIKES = results
    
    # Save results
    save_simulation_results(
        out_root,
        List_of_WEIGHTS,
        List_of_READ_MATRIX,
        List_of_LIKES,
        List_of_POSTS,
        List_of_GRAPHS,
        List_of_INDIVIDUAL_LIKES,
        PARAMS
    )
    
    print(f"Experiment {exp_name} completed. Results saved to {out_root}")

if __name__ == "__main__":
    main()