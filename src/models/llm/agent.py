import json
import numpy as np
import requests
import re
import os

import aiohttp

from src.models.llm.utlities import load_usernames, _compose_prompt_for_warmup, _execute_prompts_with_coords
from src.models.llm.utlities import _compose_prompt_for_likes, _compose_prompt_for_posts
from src.models.llm.utlities import _load_prompt_yaml, print_prompt
from src.executors.hf import execute_prompts_parallel


def generate_agent_personas(G, PARAMS):
    """
    Generate personas optimized for heated, less civil discussions
    """
    names = load_usernames()
    
    for agent in G.vs:
        # LOW Agreeableness = more confrontational, argumentative
        agent["agreeableness"] = int(np.clip(np.random.normal(25, 12), 0, 100))
        
        # HIGH Neuroticism = more emotional instability, reactive
        agent["neuroticism"] = int(np.clip(np.random.normal(75, 15), 0, 100))
        
        # LOW Openness = more rigid thinking, less tolerant of other views
        agent["openness"] = int(np.clip(np.random.normal(30, 15), 0, 100))
        
        # Normal distributions for these (they can vary)
        agent["conscientiousness"] = int(np.clip(np.random.normal(50, 15), 0, 100))
        agent["extraversion"] = int(np.clip(np.random.normal(50, 15), 0, 100))
        
        # Correlated political views - one random draw determines both
        if agent.index == 0:
            is_left_wing = False
        else:
            is_left_wing = True#np.random.rand() < 0.5
        
        if is_left_wing:
            # Left wing: low economic (left) + high social (liberal)
            agent["economic_left_right"] = 0  # Fixed far left
            agent["social_conservative_liberal"] = 100  # Fixed very liberal
            agent["social_label"] = "extreme liberal"
            agent["economic_label"] = "extreme left"
        else:
            # Right wing: high economic (right) + low social (conservative)  
            agent["economic_left_right"] = 100  # Fixed far right
            agent["social_conservative_liberal"] = 0  # Fixed very conservative
            agent["social_label"] = "extreme conservative"
            agent["economic_label"] = "extreme right"
        
        agent["name"] = names[agent.index]
        agent["topic"] = PARAMS["topic"]

async def generate_persona_statements(G, PARAMS):
    """
    Generate characteristic statements for each agent based on their traits
    """
    topic = PARAMS["topic"]
    prompts = []
    coords = []
    
    for agent in G.vs:
        # Create personality description
        # Create personality description with labels instead of numbers
        personality_desc = f"""
            You are someone with these traits:
            - Openness: {agent['openness']}/100 (creativity, curiosity)
            - Conscientiousness: {agent['conscientiousness']}/100 (organization, discipline) 
            - Extraversion: {agent['extraversion']}/100 (social energy, assertiveness)
            - Agreeableness: {agent['agreeableness']}/100 (cooperation, trust)
            - Neuroticism: {agent['neuroticism']}/100 (emotional instability, anxiety)
            - Economic views: {agent['economic_label']}
            - Social views: {agent['social_label']}

            Write an authentic statement about {topic}. 
            Keep it under 400 words.
            """
        
        prompt = [
            {"role": "system", "content": "You are generating authentic social media posts based on personality traits."},
            {"role": "user", "content": personality_desc}
        ]
        
        prompts.append(prompt)
        coords.append(agent.index)
        #if agent.index == 0:
            #print("agent number -- generate_persona_statements:", agent.index)
            #print("prompt:", prompt)
        
    # Generate statements
    statements = await execute_prompts_parallel(prompts, PARAMS, max_tokens=400, parse=None)
    
    #print   ("statement -- generate_persona_statements:", statements[0])
        



    # Store statements for each agent
    for i, statement in enumerate(statements):
        agent_id = coords[i]
        if not hasattr(G.vs[agent_id], "sample_statements"):
            G.vs[agent_id]["sample_statements"] = []
        G.vs[agent_id]["sample_statements"].append(statement.strip())

def build_final_personas(G, PARAMS):
    """
    Create the final persona prompts using generated statements
    """
    for agent in G.vs:
        statements = agent["sample_statements"]
        #print("statements -- build_final_personas:", statements)
        if statements:
            persona_prompt = f"""You are a person who thinks and writes like this:
            {chr(10).join(f'- "{stmt}"' for stmt in statements)}
            Your communication style and viewpoints should be consistent with these examples."""
        else:
            # Fallback if statement generation failed
            raise ValueError("building persona failed")
        
        agent["persona"] = persona_prompt
        agent["warmup"] = _load_prompt_yaml("identity_for_warmup") 
        agent["likes"] = _load_prompt_yaml("identity_for_likes")
        agent["posts"] = _load_prompt_yaml("identity_for_posts")

        #if agent.index == 0:
        #    print("persona_prompt -- build_final_personas:", persona_prompt)
        #    print("-" * 30)
async def initialize_agents(G, PARAMS):
    """
    Generate enhanced personas using LLM calls
    """
    # First generate personality traits
    generate_agent_personas(G, PARAMS)
    
    # generate persona statements
    await generate_persona_statements(G, PARAMS)
    
    # Build final persona prompts
    build_final_personas(G, PARAMS)


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
    #print("LIST OF PROMPTS FOR WARMUP:")
    #for prompt in prompts:
    #    print_prompt(prompt)
    # Fire in batches to avoid giant single requests
    coords, texts = await _execute_prompts_with_coords(prompts, coords, PARAMS)
    for i, text in enumerate(texts):
        a, t = coords[i]
        POSTS[a][t] = text
        #if a == 0:
        #    print("POSTS[a][t]:", POSTS[a][t])

    return POSTS


def _generate_list_prompts_for_warmup(G, POSTS):
    prompts, coords = [], []
    for t in range(G["fill_history"]):
        # Do NOT rely on G["T_current"] here; pass t explicitly to the builder.
        for agent in G.vs:
            a = agent.index
            prompts.append(_compose_prompt_for_warmup(agent, t, POSTS))
            coords.append((a, t))
            #if a == 0:
            #    print("prompt:", prompts[-1])
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
                raise ValueError(f"Missing post: agent {sender_id} at timestep {timestep}")
            prompt =_compose_prompt_for_likes(agent, G["T_current"], POSTS, coords3[-1], memory = 10)
            prompts.append(prompt)
    
    coords3, texts = await _execute_prompts_with_coords(prompts, coords3, PARAMS)

    decisions = [np.random.rand() < _extract_number(text) / 9 for text in texts]
    # for each text print: [reader_id, sender_id, prompt, text, decision]
    for i, text in enumerate(texts):
        if coords3[i][0] == 0: # agent 0
            print(f"[reader_id, sender_id] = [{coords3[i][0]}, {coords3[i][1]}]")
            print(f"prompt: {prompts[i]}")
            print(f"text: {text}")
            print(f"decision: {decisions[i]}")
            print("-" * 30)

    #print("LIST OF PROMPTS FOR LIKES:")
    #for prompt in prompts:
    #    print_prompt(prompt)
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
    
    #print("LIST OF PROMPTS FOR POSTS:")
    #for prompt in prompts:
    #    print_prompt(prompt)


def _extract_number(response):
    """Extract first digit found anywhere in the text"""
    s = response.strip()
    
    # Map superscript and subscript digits to regular digits
    special_digits_map = {
        # Superscripts
        '⁰': '0', '¹': '1', '²': '2', '³': '3', '⁴': '4', 
        '⁵': '5', '⁶': '6', '⁷': '7', '⁸': '8', '⁹': '9',
        # Subscripts
        '₀': '0', '₁': '1', '₂': '2', '₃': '3', '₄': '4', 
        '₅': '5', '₆': '6', '₇': '7', '₈': '8', '₉': '9'
    }
    
    for ch in s:
        if ch.isdigit():
            return min(max(int(ch), 0), 9)
        if ch in special_digits_map:
            return min(max(int(special_digits_map[ch]), 0), 9)
    
    return 0