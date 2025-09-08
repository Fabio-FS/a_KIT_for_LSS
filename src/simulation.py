import numpy as np
import igraph as ig
import asyncio
import aiohttp

from src.executors.hf import close_session_if_any

from src.interfaces import initialize_agents, thermalize_system, evaluate_likes, generate_posts



async def run_simulation(PARAMS):

    G, WEIGHTS, READ_MATRIX, LIKES = _initialize_everything(PARAMS)
    initialize_agents(G, PARAMS)                                            # dispatch to interfaces.py
    POSTS = await thermalize_system(G, PARAMS)                       # dispatch to interfaces.py

    for i in range(PARAMS["fill_history"], PARAMS["fill_history"] + PARAMS["timesteps"]):
        G["T_current"] = i
        print(f"T_current: {G['T_current']}")
        
        WEIGHTS, TOP_POSTS, READ_MATRIX = _matrix_operations(G, READ_MATRIX, LIKES, PARAMS, WEIGHTS, k = 2)

        # here there is the part where agents reads posts and decide what to like.
        decision, coords = await evaluate_likes(G, TOP_POSTS, POSTS, PARAMS) # dispatch to interfaces.py
        _update_likes_and_consequences(G, LIKES, decision, coords)
        
        await generate_posts(G, POSTS, PARAMS) # dispatch to interfaces.py


    # Clean up session
    await close_session_if_any()


    return G, POSTS, WEIGHTS, READ_MATRIX, LIKES



def _initialize_everything(PARAMS):
    np.random.seed(42 if PARAMS.get("seed") is None else PARAMS["seed"])
    G = generate_network(PARAMS)
    WEIGHTS, READ_MATRIX, LIKES = _initialize_matrices(G, PARAMS)
    return G, WEIGHTS, READ_MATRIX, LIKES
def generate_network(PARAMS):
    N = PARAMS["num_agents"]
    neighbors = PARAMS["neighbors"]
    timesteps = PARAMS["timesteps"]
    fill_history = PARAMS["fill_history"]
    post_read_per_round = PARAMS["post_read_per_round"]
    rewiring_p = PARAMS["rewiring_p"]
    G = ig.Graph.Watts_Strogatz(dim=1, size=N, nei=neighbors, p=rewiring_p)

    G["T_total"] = timesteps + fill_history
    G["T_current"] = 0
    G["fill_history"] = fill_history

    for agent in G.vs:
        agent["neighbors"] = np.array([neighbor.index for neighbor in agent.neighbors()])
        agent["success"] = 0
        agent["preferred_neighbors"] = np.zeros(len(agent["neighbors"]))
        agent["read_history"] = np.full((G["T_total"], post_read_per_round, 2), -1, dtype=int)

    _build_neighbor_lookups(G)
    
    return G
def _build_neighbor_lookups(G):
    for agent in G.vs:
        lookout = np.zeros(len(agent["neighbors"]), dtype=int)
        for i, neighbor_id in enumerate(agent["neighbors"]):
            neighbor_neighbors = G.vs[neighbor_id]["neighbors"]
            lookout[i] = np.where(neighbor_neighbors == agent.index)[0][0]
        agent["lookout"] = lookout

    num_agents = len(G.vs)
    max_degree = max(G.degree())
    neighbor_lookup = np.full((num_agents, max_degree), -1, dtype=int)
    lookout_lookup  = np.full((num_agents, max_degree), -1, dtype=int)

    for agent_id, agent in enumerate(G.vs):
        neighbors = agent["neighbors"]
        lookouts  = agent["lookout"]
        neighbor_lookup[agent_id, :len(neighbors)] = neighbors
        lookout_lookup[agent_id, :len(neighbors)]  = lookouts

    G["neighbor_lookup"] = neighbor_lookup
    G["lookout_lookup"]  = lookout_lookup
def _initialize_matrices(G, PARAMS):
    max_degree = np.max(np.array(G.degree()))
    num_agents = len(G.vs)

    #POSTS = initialize_posts(G, PARAMS)             # dispatched to interfaces.py
    
    READ_MATRIX = np.zeros((num_agents, G["T_total"], max_degree), dtype=float)
    WEIGHTS = np.zeros((num_agents, G["T_total"], max_degree), dtype=float)
    LIKES = np.zeros((num_agents, G["T_total"]), dtype=float)

    
    return WEIGHTS, READ_MATRIX, LIKES



def _matrix_operations(G, read_MATRIX, LIKES, PARAMS, WEIGHTS, k = 2):
    WEIGHTS = _calculate_weights(G, read_MATRIX, LIKES, PARAMS)
    TOP_POSTS = _find_top_k_posts(G, WEIGHTS, k=k)
    READ_MATRIX = _mark_posts_as_read(G, read_MATRIX, TOP_POSTS)
    _update_read_list(G, TOP_POSTS)

    return WEIGHTS, TOP_POSTS, READ_MATRIX
def _calculate_weights(G, read_MATRIX, LIKES, PARAMS):
    max_degree = np.max(np.array(G.degree()))
    num_agents = len(G.vs)
    fill_history = G["fill_history"]
    WEIGHTS = np.ones((num_agents, G["T_total"], max_degree))
    
    if PARAMS["time_decay_rate"] > 0:
        time_diffs = G["T_current"] - np.arange(G["T_total"])  
        decay_factors = np.exp(-PARAMS["time_decay_rate"] * time_diffs)
        WEIGHTS *= decay_factors[np.newaxis, :, np.newaxis]
    
    if PARAMS.get("W_agent_success", 0) > 0:
        success = np.array(G.vs["success"])
        agent_multiplier =  1.0 + success * PARAMS["W_agent_success"]
        WEIGHTS *= agent_multiplier[:, np.newaxis, np.newaxis]
    
    if PARAMS.get("W_post_success", 0) > 0:
        post_multiplier =  1.0 + LIKES * PARAMS["W_post_success"]
        WEIGHTS *= post_multiplier[:, :, np.newaxis]
    
    if PARAMS.get("W_personal_weights", 0) > 0:
        _update_personal_weights(G, WEIGHTS, PARAMS["W_personal_weights"])

    WEIGHTS += 100.0

    if PARAMS.get("noise_level", 0) > 0:
        WEIGHTS += np.random.normal(0, PARAMS["noise_level"], WEIGHTS.shape)
    
    WEIGHTS[read_MATRIX == 1] = 0

    future_mask = np.arange(G["T_total"]) >= G["T_current"]
    WEIGHTS[:, future_mask, :] = 0

    return WEIGHTS
def _update_personal_weights(G, WEIGHTS, W_personal_weights):
    num_agents = len(G.vs)
    max_degree = WEIGHTS.shape[2]
    
    neighbors_data = np.full((num_agents, max_degree), -1, dtype=int)
    preferred_data = np.zeros((num_agents, max_degree))
    lookout_data = np.zeros((num_agents, max_degree), dtype=int)
    neighbor_counts = np.zeros(num_agents, dtype=int)
    
    for i, agent in enumerate(G.vs):
        n_neighbors = len(agent["neighbors"])
        neighbors_data[i, :n_neighbors] = agent["neighbors"]
        preferred_data[i, :n_neighbors] = agent["preferred_neighbors"]
        lookout_data[i, :n_neighbors] = agent["lookout"]
        neighbor_counts[i] = n_neighbors
    
    for i in range(num_agents):
        n_neighbors = neighbor_counts[i]
        if n_neighbors == 0:
            continue
            
        neighbor_ids = neighbors_data[i, :n_neighbors]
        preferences = preferred_data[i, :n_neighbors]
        lookout_positions = lookout_data[i, :n_neighbors]
        
        multipliers = 1 + preferences * W_personal_weights
        WEIGHTS[neighbor_ids, :, lookout_positions] *= multipliers[:, np.newaxis]
def _update_likes_and_consequences(G, LIKES, decisions, coordinates):
    for i, decision in enumerate(decisions):
        if decision:
            reader_id, sender_id, timestep = coordinates[i]
            LIKES[sender_id, timestep] += 1
            G.vs[sender_id]["success"] += 1
            
            reader = G.vs[reader_id]
            position = np.where(reader["neighbors"] == sender_id)[0]
            reader["preferred_neighbors"][position] += 1
def _find_top_k_posts(G, WEIGHTS, k=5):
    num_agents = len(G.vs)
    top_posts = np.full((num_agents, k, 2), -1, dtype=int)
    
    for agent in G.vs:
        agent_id = agent.index
        neighbor_ids = agent["neighbors"]
        lookout_positions = agent["lookout"]
        
        if len(neighbor_ids) == 0:
            continue
            
        accessible_weights = WEIGHTS[neighbor_ids, :, lookout_positions]
        flat_weights = accessible_weights.flatten()
        top_k_flat_indices = np.argpartition(flat_weights, -k)[-k:]
        top_k_flat_indices = top_k_flat_indices[np.argsort(flat_weights[top_k_flat_indices])[::-1]]
        
        neighbor_positions, timesteps = np.unravel_index(top_k_flat_indices, accessible_weights.shape)
        
        for rank in range(min(k, len(top_k_flat_indices))):
            author_id = neighbor_ids[neighbor_positions[rank]]
            timestep = timesteps[rank]
            top_posts[agent_id, rank] = [author_id, timestep]
    
    return top_posts
def _mark_posts_as_read(G, read_MATRIX, top_posts):
    num_agents, k, _ = top_posts.shape
    
    valid_mask = (top_posts[:, :, 0] != -1) & (top_posts[:, :, 1] != -1)
    valid_agents, valid_ranks = np.where(valid_mask)
    
    if len(valid_agents) == 0:
        return read_MATRIX
    
    author_ids = top_posts[valid_agents, valid_ranks, 0]
    timesteps = top_posts[valid_agents, valid_ranks, 1]
    
    neighbor_lookup = G["neighbor_lookup"]
    lookout_lookup = G["lookout_lookup"]

    for i in range(len(valid_agents)):
        agent_id = valid_agents[i]
        author_id = author_ids[i]
        timestep = timesteps[i]
        
        neighbor_positions = np.where(neighbor_lookup[agent_id] == author_id)[0]
        if len(neighbor_positions) > 0:
            lookout_pos = lookout_lookup[agent_id, neighbor_positions[0]]
            read_MATRIX[author_id, timestep, lookout_pos] = 1
    
    return read_MATRIX
def _update_read_list(G, top_posts):
    current_timestep = G["T_current"]
    
    for agent_id in range(len(G.vs)):
        G.vs[agent_id]["read_history"][current_timestep, :, :] = top_posts[agent_id]


def printout_posts(G, POSTS, n):
    print("__________")
    for i in range(n):
        print(f"timestep: {i}")
        for i, _ in enumerate(G.vs):
            print(f"agent {i}: {POSTS[i][i]}")
    print("__________")