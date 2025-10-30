import json
import numpy as np
import requests
import re
import os

import aiohttp

import numpy as np
from typing import List, Tuple

from src.models.llm.utlities import (
    load_usernames,
    _compose_prompt_for_warmup,
    _execute_prompts_with_coords,
    _compose_prompt_for_likes,
    _compose_prompt_for_posts,
    _load_prompt_yaml,
)

from src.executors.hf import execute_prompts_parallel


DEFAULT_MEMORY = 10


async def initialize_agents(G, PARAMS) -> None:
    """
    Assign stance + persona YAMLs and preformatted identities.
    NOTE: Topical personas are currently hard-coded (Ukraine-related).
    If you later generalize, choose by PARAMS["topic"] or model name.
    """
    names = load_usernames()

    for agent in G.vs:
        # Random stance in {pro-Russia, neutral, pro-Ukraine}
        r = np.random.rand()
        if r < 1/3:
            agent["stance"] = 0
            agent["persona"] = _load_prompt_yaml("persona_pro_Russia")
        elif r < 2/3:
            agent["stance"] = 1
            agent["persona"] = _load_prompt_yaml("persona_neutral")
        else:
            agent["stance"] = 2
            agent["persona"] = _load_prompt_yaml("persona_pro_Ukraine")

        agent["name"] = names[agent.index]

        # Pre-baked identities from YAMLs
        persona = agent["persona"]
        agent["warmup"] = _load_prompt_yaml("identity_for_warmup").format(persona=persona)
        agent["likes"]  = _load_prompt_yaml("identity_for_likes").format(persona=persona)
        agent["posts"]  = _load_prompt_yaml("identity_for_posts").format(persona=persona)

### ----------------------------------------------------------
### thermalize_system
### ----------------------------------------------------------

# =============================
# Warmup / Thermalization
# =============================
async def thermalize_system(G, PARAMS):
    """
    Pre-generate posts for t in [0, warmup_length - 1].
    Returns:
        POSTS: ndarray shape (num_agents, T_total), dtype=object
    """
    POSTS = np.empty((len(G.vs), G["T_total"]), dtype=object)
    warmup_length = int(PARAMS.get("warmup_length", PARAMS.get("warmup_length", 0)) or 0)
    if warmup_length <= 0:
        return POSTS

    prompts, coords = _generate_list_prompts_for_warmup(G, POSTS, warmup_length)
    coords, texts = await _execute_prompts_with_coords(prompts, coords, PARAMS)

    for i, text in enumerate(texts):
        a, t = coords[i]
        POSTS[a][t] = text

    return POSTS

def _generate_list_prompts_for_warmup(G, POSTS, warmup_length: int) -> Tuple[List, List[Tuple[int, int]]]:
    prompts, coords = [], []
    for t in range(warmup_length):
        for agent in G.vs:
            a = agent.index
            prompts.append(_compose_prompt_for_warmup(agent, t, POSTS))
            coords.append((a, t))
    return prompts, coords


# =============================
# Likes and Posts
# =============================
async def evaluate_likes(G, TOP_POSTS, POSTS, PARAMS):
    """
    Build like-evaluation prompts for each agent over their top-k reads at G['T_current'].

    Returns:
        decisions: list[bool]  # like or not
        coords3:   list[[reader_id, sender_id, timestep]]
    """
    memory = int(PARAMS.get("memory", DEFAULT_MEMORY))
    prompts, coords3 = [], []

    for agent in G.vs:
        top_for_agent = TOP_POSTS[agent.index]
        for post_index in range(len(top_for_agent)):
            sender_id = top_for_agent[post_index, 0]
            timestep  = top_for_agent[post_index, 1]

            # Guard against unfilled slots (e.g., -1 placeholders)
            if sender_id is None or timestep is None or sender_id < 0 or timestep < 0:
                continue

            # Ensure the referenced post exists
            post = POSTS[sender_id][timestep]
            if post is None:
                raise ValueError(
                    f"Missing post while evaluating likes: "
                    f"reader={agent.index}, sender={sender_id}, post_t={timestep}, "
                    f"T_current={G.attributes().get('T_current')}"
                )

            coords3.append([agent.index, sender_id, timestep])
            prompt = _compose_prompt_for_likes(agent, G["T_current"], POSTS, coords3[-1], memory=memory)
            prompts.append(prompt)

    if not prompts:
        return [], coords3

    coords3, texts = await _execute_prompts_with_coords(prompts, coords3, PARAMS, max_tokens=10, temperature=0.5)  # Down from default 128
    decisions = [np.random.rand() < (_extract_number(text) / 9) for text in texts]
    return decisions, coords3


async def generate_posts(G, POSTS, PARAMS) -> None:
    """
    Generate one post per agent at t = G['T_current'] using read history + own past posts.
    """
    memory = int(PARAMS.get("memory", DEFAULT_MEMORY))
    prompts, coords = [], []

    for agent in G.vs:
        prompt, coord = _compose_prompt_for_posts(G, agent, G["T_current"], POSTS, memory=memory)
        prompts.append(prompt)
        coords.append(coord)

    if not prompts:
        return

    coords, texts = await _execute_prompts_with_coords(prompts, coords, PARAMS, max_tokens=128, temperature=1.2)
    for i, text in enumerate(texts):
        a, t = coords[i]
        POSTS[a][t] = text



# =============================
# Helpers
# =============================
def _extract_number(response: str) -> int:
    """
    Extract the first digit in {0..9} from response, including super/subscripts.
    Fallback 0 if none found.
    """
    s = (response or "").strip()

    special_digits_map = {
        # Superscripts
        "⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4",
        "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9",
        # Subscripts
        "₀": "0", "₁": "1", "₂": "2", "₃": "3", "₄": "4",
        "₅": "5", "₆": "6", "₇": "7", "₈": "8", "₉": "9",
    }

    for ch in s:
        if ch.isdigit():
            return min(max(int(ch), 0), 9)
        mapped = special_digits_map.get(ch)
        if mapped is not None:
            return min(max(int(mapped), 0), 9)
    return 0