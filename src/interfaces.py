from agent_LLM import _generate_post_LLM, _generate_like_LLM, _generate_post_warmup_LLM, _initialize_agents_LLM
from agent_LLM import _generate_prompt_post_LLM, _generate_prompt_warmup_LLM, _generate_prompt_like_LLM
# if more opinion models are added, add them here and have agent_BCM, agent_LLM, ... each of them needs to implement the same interface

import aiohttp

def _generate_post(agent, timestep, POSTS, PARAMS):
    if PARAMS["opinion_model"] == "LLM":
        return _generate_prompt_post_LLM(agent, timestep, POSTS, PARAMS)
    else:
        print("unknown opinion model")
        exit()

def _like_decision(agent, timestep, POSTS, post, PARAMS):
    if PARAMS["opinion_model"] == "LLM":
        return _generate_prompt_like_LLM(agent, timestep, POSTS, post, PARAMS)
    else:
        print("unknown opinion model")
        exit()

def _generate_post_warmup(agent, timestep, POSTS, PARAMS):
    if PARAMS["opinion_model"] == "LLM":
        return _generate_prompt_warmup_LLM(agent, timestep, POSTS, PARAMS)
    else:
        print("unknown opinion model")
        exit()

def _initialize_agents(G, PARAMS):
    if PARAMS["opinion_model"] == "LLM":
        _initialize_agents_LLM(G, PARAMS)
    else:
        print("unknown opinion model")
        exit()