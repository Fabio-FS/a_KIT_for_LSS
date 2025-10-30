import os, json, csv, time, uuid
import numpy as np

def _flatten_posts(POSTS):
    # POSTS: list[list[str|None]] or np.ndarray(dtype=object)
    R = len(POSTS)
    for agent_id in range(R):
        row = POSTS[agent_id]
        T = len(row)
        for t in range(T):
            txt = row[t]
            if txt is None:
                continue
            s = str(txt).strip()
            if s:
                yield (agent_id, t, s)

def _filter_sensitive_params(params):
    """Remove sensitive keys from params dict before saving."""
    sensitive_keys = {
        'hf_token', 'huggingface_token', 'api_key', 'token', 
        'password', 'secret', 'key', 'auth'
    }
    
    filtered = {}
    for k, v in params.items():
        key_lower = k.lower()
        if any(sensitive in key_lower for sensitive in sensitive_keys):
            filtered[k] = "[REDACTED]"
        else:
            filtered[k] = v
    return filtered


def save_graphs_data(List_of_GRAPHS, out_root):
    """Save essential graph data as JSON files."""
    
    for r, G in enumerate(List_of_GRAPHS):
        rdir = os.path.join(out_root, f"r{r:03d}")
        os.makedirs(rdir, exist_ok=True)
        
        # Extract read history for each agent
        read_history = []
        for agent in G.vs:
            agent_reads = agent["read_history"].tolist()
            read_history.append(agent_reads)
        
        # Extract other useful agent data
        agent_data = []
        for agent in G.vs:
            agent_info = {
                "name": agent["name"],
                "neighbors": agent["neighbors"].tolist(),
                "success": agent["success"],
                "stance" : agent["stance"]
            }
            agent_data.append(agent_info)
        
        # Graph metadata
        graph_data = {
            "T_total": G["T_total"],
            "T_current": G["T_current"],
            "warmup_length": G["warmup_length"],
            "read_history": read_history,
            "agents": agent_data
        }
        
        with open(os.path.join(rdir, "graph_data.json"), "w", encoding="utf-8") as f:
            json.dump(graph_data, f, ensure_ascii=False, indent=2)

def save_simulation_results(out_root, List_of_WEIGHTS, List_of_READ_MATRIX, 
                           List_of_LIKES, List_of_POSTS, List_of_GRAPHS, 
                           List_of_INDIVIDUAL_LIKES, PARAMS):
    
    save_replicas_raw(out_root, List_of_WEIGHTS, List_of_READ_MATRIX, 
                     List_of_LIKES, List_of_POSTS, List_of_INDIVIDUAL_LIKES, PARAMS)
    
    save_graphs_data(List_of_GRAPHS, out_root)

def save_replicas_raw(out_root, List_of_WEIGHTS, List_of_READ_MATRIX, 
                     List_of_LIKES, List_of_POSTS, List_of_INDIVIDUAL_LIKES, 
                     PARAMS, extra_meta=None):
    
    os.makedirs(out_root, exist_ok=True)
    run_id = str(uuid.uuid4())
    saved_at = time.strftime("%Y-%m-%d %H:%M:%S")
    safe_params = _filter_sensitive_params(PARAMS)

    R = len(List_of_POSTS)
    manifest = {
        "run_id": run_id,
        "saved_at": saved_at,
        "replicas": R,
        "out_root": out_root,
        "files": [],
        "params": safe_params,
    }
    if extra_meta:
        manifest["extra_meta"] = extra_meta

    for r in range(R):
        rdir = os.path.join(out_root, f"r{r:03d}")
        os.makedirs(rdir, exist_ok=True)

        # arrays - now includes INDIVIDUAL_LIKES
        np.savez_compressed(
            os.path.join(rdir, "arrays.npz"),
            WEIGHTS=List_of_WEIGHTS[r],
            READ_MATRIX=List_of_READ_MATRIX[r],
            LIKES=List_of_LIKES[r],  # Keep old format
            INDIVIDUAL_LIKES=List_of_INDIVIDUAL_LIKES[r],  # Add new format
        )

        # posts
        with open(os.path.join(rdir, "posts.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["agent", "t", "text"])
            w.writerows(_flatten_posts(List_of_POSTS[r]))

        # robust shape reporting for POSTS even if it's a numpy array
        posts_r = List_of_POSTS[r]
        posts_num_agents = len(posts_r)
        posts_T = len(posts_r[0]) if posts_num_agents > 0 else 0

        # per-replica meta
        with open(os.path.join(rdir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "run_id": run_id,
                    "replica": r,
                    "saved_at": saved_at,
                    "params": safe_params,
                    "shapes": {
                        "WEIGHTS": list(np.shape(List_of_WEIGHTS[r])),
                        "READ_MATRIX": list(np.shape(List_of_READ_MATRIX[r])),
                        "LIKES": list(np.shape(List_of_LIKES[r])),
                        "INDIVIDUAL_LIKES": list(np.shape(List_of_INDIVIDUAL_LIKES[r])),  # ← add this
                        "POSTS": [posts_num_agents, posts_T],
                    },
                    "files": {"arrays": "arrays.npz", "posts": "posts.csv"},
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        manifest["files"].append(
            {"replica": r, "arrays": f"r{r:03d}/arrays.npz", "posts": f"r{r:03d}/posts.csv", "meta": f"r{r:03d}/meta.json"}
        )

    with open(os.path.join(out_root, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

def load_manifest(out_root: str):
    """Load the top-level manifest.json."""
    with open(os.path.join(out_root, "manifest.json"), "r", encoding="utf-8") as f:
        return json.load(f)

def load_replica(replica_dir):
    arrays = np.load(os.path.join(replica_dir, "arrays.npz"), allow_pickle=True)
    
    with open(os.path.join(replica_dir, "meta.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)

    posts = []
    posts_csv = os.path.join(replica_dir, "posts.csv")
    if os.path.exists(posts_csv):
        with open(posts_csv, "r", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                posts.append((int(row["agent"]), int(row["t"]), row["text"]))

    return {
        "WEIGHTS": arrays["WEIGHTS"],
        "READ_MATRIX": arrays["READ_MATRIX"],
        "LIKES": arrays["LIKES"],  # Old format still available
        "INDIVIDUAL_LIKES": arrays["INDIVIDUAL_LIKES"],  # New format available
        "POSTS_long": posts,
        "meta": meta,
    }


def show_discussion_from_saved(posts_data, graph_data, agent_id, max_timesteps=5):
    print(f"=== DISCUSSION FOR AGENT {agent_id} ===\n")
    read_history = graph_data["read_history"][agent_id]
    agent_name = graph_data["agents"][agent_id]["name"]

    # Build a fast lookup: (author_id, t) -> text
    posts_lookup = {(a, t): txt for a, t, txt in posts_data}

    for t in range(min(max_timesteps, len(read_history))):
        print(f"--- TIMESTEP {t} ({agent_name}) ---")
        print("READ:")
        for author_id, post_t in read_history[t]:
            if author_id != -1 and post_t != -1:
                txt = posts_lookup.get((author_id, post_t))
                if txt is not None:
                    author_name = graph_data["agents"][author_id]["name"]
                    print(f"  {author_name} (t={post_t}): {txt}")

        # wrote
        my_txt = posts_lookup.get((agent_id, t))
        if my_txt is not None:
            print(f"WROTE: {my_txt}")
        print()