import numpy as np
import igraph as ig
import asyncio
import aiohttp


# from agent_LLM import _generate_prompt_post_LLM, _generate_prompt_like_LLM, _generate_prompt_history_LLM
# from agent_BCM import compute_post_BCM, compute_history_BCM, compute_like_decision_BCM
# if more opinion models are added, add them here and have agent_BCM, agent_LLM, ... each of them needs to implement the same interface




from src.models import get_model
import inspect



async def initialize_agents(G, PARAMS):
    model = get_model(PARAMS["opinion_model"])
    func = model.initialize_agents
    if inspect.iscoroutinefunction(func):
        return await func(G, PARAMS)
    else:
        return func(G, PARAMS)

async def thermalize_system(G, PARAMS):
    model = get_model(PARAMS["opinion_model"])
    func = model.thermalize_system
    if inspect.iscoroutinefunction(func):
        return await func(G, PARAMS)
    else:
        return func(G, PARAMS)

async def evaluate_likes(G, TOP_POSTS, POSTS, PARAMS):
    model = get_model(PARAMS["opinion_model"])
    func = model.evaluate_likes
    if inspect.iscoroutinefunction(func):
        return await func(G, TOP_POSTS, POSTS, PARAMS)
    else:
        return func(G, TOP_POSTS, POSTS, PARAMS)

async def generate_posts(G, POSTS, PARAMS):
    model = get_model(PARAMS["opinion_model"])
    func = model.generate_posts
    if inspect.iscoroutinefunction(func):
        return await func(G, POSTS, PARAMS)
    else:
        return func(G, POSTS, PARAMS)

















