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

def save_replicas_raw(
    out_root: str,
    List_of_WEIGHTS,
    List_of_READ_MATRIX,
    List_of_LIKES,
    List_of_POSTS,
    PARAMS: dict,
    extra_meta: dict | None = None,
):
    os.makedirs(out_root, exist_ok=True)
    run_id = str(uuid.uuid4())
    saved_at = time.strftime("%Y-%m-%d %H:%M:%S")

    R = len(List_of_POSTS)
    manifest = {
        "run_id": run_id,
        "saved_at": saved_at,
        "replicas": R,
        "out_root": out_root,
        "files": [],
        "params": PARAMS,
    }
    if extra_meta:
        manifest["extra_meta"] = extra_meta

    for r in range(R):
        rdir = os.path.join(out_root, f"r{r:03d}")
        os.makedirs(rdir, exist_ok=True)

        # arrays
        np.savez_compressed(
            os.path.join(rdir, "arrays.npz"),
            WEIGHTS=List_of_WEIGHTS[r],
            READ_MATRIX=List_of_READ_MATRIX[r],
            LIKES=List_of_LIKES[r],
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
                    "params": PARAMS,
                    "shapes": {
                        "WEIGHTS": list(np.shape(List_of_WEIGHTS[r])),
                        "READ_MATRIX": list(np.shape(List_of_READ_MATRIX[r])),
                        "LIKES": list(np.shape(List_of_LIKES[r])),
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

def load_replica(replica_dir: str):
    """
    Load one replica folder (e.g. runs/2025-09-08_exp1/r000).
    Returns a dict with arrays, posts, and meta.
    """
    # arrays
    arrays = np.load(os.path.join(replica_dir, "arrays.npz"), allow_pickle=True)
    
    # meta
    with open(os.path.join(replica_dir, "meta.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)

    # posts
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
        "LIKES": arrays["LIKES"],
        "POSTS_long": posts,
        "meta": meta,
    }