import numpy as np
import json
import os


def load_run(run_path):
    """Load all data from a saved simulation run."""
    manifest_path = os.path.join(run_path, "manifest.json")
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
    
    replicas = []
    for r in range(manifest["replicas"]):
        rdir = os.path.join(run_path, f"r{r:03d}")
        
        arrays = np.load(os.path.join(rdir, "arrays.npz"), allow_pickle=True)
        
        with open(os.path.join(rdir, "graph_data.json"), "r") as f:
            graph_data = json.load(f)
        
        replicas.append({
            "INDIVIDUAL_LIKES": arrays["INDIVIDUAL_LIKES"],
            "graph_data": graph_data,
        })
    
    return replicas, manifest


def calculate_exposure_diversity(replica):
    """
    Exposure diversity: stance distribution in reads.
    
    For each agent at each timestep, compute the entropy of stance distribution
    in what they read. Higher entropy = more diverse exposure.
    
    Returns:
        diversity_per_agent: shape (num_agents, T_total)
        diversity_mean: scalar, mean across all agents and timesteps
    """
    graph_data = replica["graph_data"]
    num_agents = len(graph_data["agents"])
    T_total = graph_data["T_total"]
    
    diversity_per_agent = np.zeros((num_agents, T_total))
    
    for agent_id in range(num_agents):
        read_history = graph_data["read_history"][agent_id]
        
        for t in range(T_total):
            reads = read_history[t]
            
            stances = []
            for author_id, post_t in reads:
                if author_id != -1 and post_t != -1:
                    stance = graph_data["agents"][author_id]["stance"]
                    stances.append(stance)
            
            if len(stances) == 0:
                diversity_per_agent[agent_id, t] = 0
                continue
            
            unique, counts = np.unique(stances, return_counts=True)
            probs = counts / counts.sum()
            entropy = -np.sum(probs * np.log(probs + 1e-10))
            diversity_per_agent[agent_id, t] = entropy
    
    diversity_mean = diversity_per_agent.mean()
    
    return diversity_per_agent, diversity_mean


def calculate_echo_chamber_score(replica):
    """
    Echo chamber effects: homophily in reading.
    
    For each agent, compute the fraction of reads that come from same-stance agents.
    Higher score = stronger echo chamber.
    
    Returns:
        echo_per_agent: shape (num_agents, T_total)
        echo_mean: scalar, mean across all agents and timesteps
    """
    graph_data = replica["graph_data"]
    num_agents = len(graph_data["agents"])
    T_total = graph_data["T_total"]
    
    echo_per_agent = np.zeros((num_agents, T_total))
    
    for agent_id in range(num_agents):
        my_stance = graph_data["agents"][agent_id]["stance"]
        read_history = graph_data["read_history"][agent_id]
        
        for t in range(T_total):
            reads = read_history[t]
            
            same_stance = 0
            total_reads = 0
            
            for author_id, post_t in reads:
                if author_id != -1 and post_t != -1:
                    author_stance = graph_data["agents"][author_id]["stance"]
                    total_reads += 1
                    if author_stance == my_stance:
                        same_stance += 1
            
            if total_reads == 0:
                echo_per_agent[agent_id, t] = 0
            else:
                echo_per_agent[agent_id, t] = same_stance / total_reads
    
    echo_mean = echo_per_agent.mean()
    
    return echo_per_agent, echo_mean


def calculate_engagement_patterns(replica):
    """
    Engagement patterns: like rates by stance.
    
    Computes:
    1. Like rate when reading same-stance posts
    2. Like rate when reading different-stance posts
    3. Overall like rate
    
    Returns:
        engagement_stats: dict with various engagement metrics
    """
    graph_data = replica["graph_data"]
    INDIVIDUAL_LIKES = replica["INDIVIDUAL_LIKES"]
    num_agents = len(graph_data["agents"])
    T_total = graph_data["T_total"]
    
    same_stance_likes = 0
    same_stance_reads = 0
    
    diff_stance_likes = 0
    diff_stance_reads = 0
    
    total_likes = 0
    total_reads = 0
    
    for agent_id in range(num_agents):
        my_stance = graph_data["agents"][agent_id]["stance"]
        read_history = graph_data["read_history"][agent_id]
        
        for t in range(T_total):
            reads = read_history[t]
            
            for author_id, post_t in reads:
                if author_id != -1 and post_t != -1:
                    author_stance = graph_data["agents"][author_id]["stance"]
                    
                    liked = INDIVIDUAL_LIKES[agent_id, author_id, post_t]
                    
                    total_reads += 1
                    if liked:
                        total_likes += 1
                    
                    if author_stance == my_stance:
                        same_stance_reads += 1
                        if liked:
                            same_stance_likes += 1
                    else:
                        diff_stance_reads += 1
                        if liked:
                            diff_stance_likes += 1
    
    engagement_stats = {
        "overall_like_rate": total_likes / total_reads if total_reads > 0 else 0,
        "same_stance_like_rate": same_stance_likes / same_stance_reads if same_stance_reads > 0 else 0,
        "diff_stance_like_rate": diff_stance_likes / diff_stance_reads if diff_stance_reads > 0 else 0,
        "like_rate_ratio": (same_stance_likes / same_stance_reads) / (diff_stance_likes / diff_stance_reads) if diff_stance_reads > 0 and diff_stance_likes > 0 else np.inf,
        "total_likes": total_likes,
        "total_reads": total_reads,
    }
    
    return engagement_stats


def calculate_all_metrics(run_path):
    """
    Calculate all three metrics for a simulation run.
    
    Args:
        run_path: Path to the saved simulation run (e.g., "runs/Exp_W_post")
    
    Returns:
        results: dict with metrics for each replica
    """
    replicas, manifest = load_run(run_path)
    
    results = {
        "run_path": run_path,
        "params": manifest["params"],
        "replicas": []
    }
    
    for r, replica in enumerate(replicas):
        diversity_per_agent, diversity_mean = calculate_exposure_diversity(replica)
        echo_per_agent, echo_mean = calculate_echo_chamber_score(replica)
        engagement_stats = calculate_engagement_patterns(replica)
        
        results["replicas"].append({
            "replica_id": r,
            "exposure_diversity": {
                "per_agent": diversity_per_agent,
                "mean": diversity_mean,
            },
            "echo_chamber": {
                "per_agent": echo_per_agent,
                "mean": echo_mean,
            },
            "engagement": engagement_stats,
        })
    
    return results


def print_summary(results):
    """Print a summary of the metrics across all replicas."""
    print(f"\n{'='*60}")
    print(f"METRICS SUMMARY: {results['run_path']}")
    print(f"{'='*60}\n")
    
    num_replicas = len(results["replicas"])
    
    diversity_means = [r["exposure_diversity"]["mean"] for r in results["replicas"]]
    echo_means = [r["echo_chamber"]["mean"] for r in results["replicas"]]
    overall_like_rates = [r["engagement"]["overall_like_rate"] for r in results["replicas"]]
    same_stance_rates = [r["engagement"]["same_stance_like_rate"] for r in results["replicas"]]
    diff_stance_rates = [r["engagement"]["diff_stance_like_rate"] for r in results["replicas"]]
    
    print(f"EXPOSURE DIVERSITY (entropy of stance distribution)")
    print(f"  Mean across replicas: {np.mean(diversity_means):.3f} ± {np.std(diversity_means):.3f}")
    print()
    
    print(f"ECHO CHAMBER SCORE (fraction same-stance reads)")
    print(f"  Mean across replicas: {np.mean(echo_means):.3f} ± {np.std(echo_means):.3f}")
    print()
    
    print(f"ENGAGEMENT PATTERNS")
    print(f"  Overall like rate: {np.mean(overall_like_rates):.3f} ± {np.std(overall_like_rates):.3f}")
    print(f"  Same-stance like rate: {np.mean(same_stance_rates):.3f} ± {np.std(same_stance_rates):.3f}")
    print(f"  Diff-stance like rate: {np.mean(diff_stance_rates):.3f} ± {np.std(diff_stance_rates):.3f}")
    print()
    
    print(f"{'='*60}\n")






