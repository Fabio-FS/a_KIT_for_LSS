import json
import numpy as np
import requests
import re
import os


def _initialize_agents_LLM(G, PARAMS):
    identity = _load_prompt("identity")
    likes_with_history = _load_prompt("likes_with_history")
    post_generation = _load_prompt("post_generation")
    for agent in G.vs:
        agent["identity"] = identity
        agent["likes_with_history"] = likes_with_history
        agent["post_generation"] = post_generation

def _generate_post_warmup_LLM(agent, current_timestep_index, POSTS, PARAMS, self_memory=10, debug=False):
    """
    Generate a post that conditions ONLY on the agent's own past posts.
    Used to 'thermalize' the history for t < 0.
    """
    if debug:
        return f"(warmup) agent {agent.index} at pre-timestep {current_timestep_index}"

    # build self-history using array index, not 'simulation time'
    start = max(0, current_timestep_index - self_memory)
    self_history = []
    for i in range(start, current_timestep_index):
        post = POSTS[agent.index][i]
        if post is not None:
            self_history.append(post)
    self_history = "\n".join(self_history)

    # prefer a dedicated warmup prompt; fall back to post_generation with empty context
    warmup_prompt = _load_prompt("warmup_post_generation")
    if warmup_prompt.startswith("Error loading") or warmup_prompt.startswith("Fallback"):
        template = _load_prompt("post_generation")
        prompt = template.format(identity=agent["identity"], history=self_history, context="")
    else:
        prompt = warmup_prompt.format(identity=agent["identity"], history=self_history)

    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "model": PARAMS["MODEL"]
    }
    headers = {"Authorization": f"Bearer {PARAMS['HF_TOKEN']}"}
    try:
        response = requests.post(PARAMS["API_URL"], headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
        #return "I like cats."
    except Exception:
        return f"ERROR. But I like cats."


def _generate_post_LLM(agent, current_timestep, POSTS, PARAMS, debug = False, self_memory = 10, read_memory = 5):
    if debug:
        return "just a test post"
    
    self_history = _build_history_of_written_posts(agent, current_timestep, POSTS, memory = self_memory)
    read_history = _build_history_of_recent_reads(agent, current_timestep, POSTS, memory = read_memory)
    prompt = agent["post_generation"].format(
        identity = agent["identity"], 
        history = self_history, 
        context = read_history
    )
    
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "model": PARAMS["MODEL"]
    }
    headers = {"Authorization": f"Bearer {PARAMS['HF_TOKEN']}"}
    
    try:
        r = requests.post(PARAMS["API_URL"], headers=headers, json=payload, timeout=60)
        r.raise_for_status()
        content = r.json()["choices"][0]["message"]["content"]
        #content = "I like cats."
    except Exception:
        return "ERROR. But I like cats."  # for _generate_post
    
    return content


def _like_decision_LLM(agent, current_timestep, POSTS, post, PARAMS, debug=False):
    if debug:
        return np.random.rand() < 0.5
    
    prompt = agent["likes_with_history"].format(
        identity=agent["identity"], 
        # (agent, current_timestep, POSTS, memory = self_memory)
        history=_build_history_of_written_posts(agent, current_timestep, POSTS, memory=10), 
        message=post
    )
    
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "model": PARAMS["MODEL"]
    }
    headers = {"Authorization": f"Bearer {PARAMS['HF_TOKEN']}"}
    
    try:
        r = requests.post(PARAMS["API_URL"], headers=headers, json=payload, timeout=60)
        r.raise_for_status()
        content = r.json()["choices"][0]["message"]["content"]
        #content = "5"
    except Exception:
        return np.random.rand() < 0.5  # for _like_decision
    
    return np.random.rand() < _extract_number(content) / 9



def _load_prompt(field_name) -> str:
    import json
    with open("prompts.json", "r") as f:
        data = json.load(f)
    if field_name not in data or not str(data[field_name]).strip():
        raise KeyError(f"Missing prompt: {field_name}")
    return data[field_name]




def _build_history_of_recent_reads(agent, current_timestep, POSTS, memory=5, max_chars=4000):
    t_min = max(0, current_timestep - memory + 1)
    pieces, used = [], 0
    for t in range(current_timestep, t_min - 1, -1):
        for author_id, post_timestep in agent["read_history"][t]:
            if author_id == -1 or post_timestep == -1:
                continue
            post = POSTS[author_id][post_timestep]
            if not post:
                continue
            add = len(post) + (1 if pieces else 0)
            if used + add > max_chars:
                return "\n".join(pieces)
            pieces.append(post)
            used += add
    return "\n".join(pieces)
def _build_history_of_written_posts(agent, current_timestep, POSTS, memory=10, max_chars=4000):
    i_min = max(0, current_timestep - memory + 1)
    pieces, used = [], 0
    for i in range(current_timestep, i_min - 1, -1):
        post = POSTS[agent.index][i]
        if not post:
            continue
        add = len(post) + (1 if pieces else 0)
        if used + add > max_chars:
            return "\n".join(pieces)
        pieces.append(post)
        used += add
    return "\n".join(pieces)






def _extract_number(response):
    s = response.strip()
    if s and s[0].isdigit():
        return min(max(int(s[0]), 0), 9)
    return 5

