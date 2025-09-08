import json
import numpy as np
import requests
import re
import os

import aiohttp

from src.models.llm.utlities import load_usernames, _compose_prompt_for_warmup, _execute_prompts_with_coords
from src.models.llm.utlities import _compose_prompt_for_likes, _compose_prompt_for_posts
from src.models.llm.utlities import _load_prompt_yaml

def initialize_agents(G, PARAMS):
    identity_for_warmup     = _load_prompt_yaml("identity_for_warmup")
    identity_for_likes      = _load_prompt_yaml("identity_for_likes")
    identity_for_posts      = _load_prompt_yaml("identity_for_posts")

    names = load_usernames()
    for agent in G.vs:
        agent["warmup"]     = identity_for_warmup
        agent["likes"]      = identity_for_likes
        agent["posts"]      = identity_for_posts
        agent["name"]       = names[agent.index]
        # randomly select a value from ["far left", "left", "center", "right", "far right"]
        agent["left_right"] = np.random.choice(["far left", "left", "center", "right", "far right"])


### ----------------------------------------------------------
### thermalize_system
### ----------------------------------------------------------

async def thermalize_system(G, PARAMS):
    """
    LLM thermalization: pre-generate posts for the first `fill_history` timesteps.

    Steps:
      1) For each t in [0, fill_history-1] and each agent, build a post prompt.
      2) Send prompts in parallel batches to HF.
      3) Write returned texts into POSTS[agent_id][t].

    Notes:
      - Does NOT mutate RNG.
      - Uses PARAMS['max_post_tokens'] if present; defaults to 128.
      - Uses PARAMS['batch_size'] if present; defaults to 128 prompts per batch.
    """

     #POSTS = [[None] * G["T_total"] for _ in range(len(G.vs))]
    POSTS = np.empty((len(G.vs), G["T_total"]), dtype=object)
    fill_history = int(PARAMS.get("fill_history", 0) or 0)
    if fill_history <= 0:
        return POSTS  # nothing to do

    # Build all prompts once; keep explicit (agent_id, t) coords
    
    prompts, coords = _generate_list_prompts_for_warmup(G, POSTS)

    # Fire in batches to avoid giant single requests
    coords, texts = await _execute_prompts_with_coords(prompts, coords, PARAMS)
    for i, text in enumerate(texts):
        a, t = coords[i]
        POSTS[a][t] = text

    return POSTS
def _generate_list_prompts_for_warmup(G, POSTS):
    prompts, coords = [], []
    for t in range(G["fill_history"]):
        # Do NOT rely on G["T_current"] here; pass t explicitly to the builder.
        for agent in G.vs:
            a = agent.index
            prompts.append(_compose_prompt_for_warmup(agent, t, POSTS))
            coords.append((a, t))
    return prompts, coords




async def evaluate_likes(G, TOP_POSTS, POSTS, PARAMS):
    prompts, coords3 = [], []
    for agent in G.vs:
        for post_index in range(len(TOP_POSTS[agent.index])):
            sender_id = TOP_POSTS[agent.index, post_index, 0]
            timestep = TOP_POSTS[agent.index, post_index, 1]
            coords3.append([agent.index, sender_id, timestep])
            post = POSTS[sender_id][timestep]
            if post is None:
                continue
            prompt =_compose_prompt_for_likes(agent, G["T_current"], POSTS, coords3[-1], memory = 10)
            prompts.append(prompt)
    
    coords3, texts = await _execute_prompts_with_coords(prompts, coords3, PARAMS)
    decisions = [np.random.rand() < _extract_number(text) / 9 for text in texts]
    return decisions, coords3


async def generate_posts(G, POSTS, PARAMS):
    """
    generate posts for the next timestep
    Each agent will generate 1 post.
    The decision of which post is based BOTH on:
    - the posts the agent itself wrote in the past,
    - the posts the agent has read in the past
    """
    prompts, coords = [], []
    for agent in G.vs:
        prompt, coord =_compose_prompt_for_posts(G, agent, G["T_current"], POSTS, memory = 10)
        prompts.append(prompt)
        coords.append(coord)
    
    coords, texts = await _execute_prompts_with_coords(prompts, coords, PARAMS)
    for i, text in enumerate(texts):
        a, t = coords[i]
        POSTS[a][t] = text



























def _extract_number(response):
    s = response.strip()
    if s and s[0].isdigit():
        return min(max(int(s[0]), 0), 9)
    return 5

