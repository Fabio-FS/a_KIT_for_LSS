import numpy as np
import igraph as ig
from agent import _load_prompt, _generate_post, _like_decision

        
def generate_network(N = 5, neighbors = 4, p = 0.05, timesteps = 10, fill_history = 3, k = 2):
   G = ig.Graph.Watts_Strogatz(dim=1, size=N, nei=neighbors, p=p)
   identity = _load_prompt("identity")
   likes_no_history = _load_prompt("likes_no_history")
   likes_with_history = _load_prompt("likes_with_history")
   post_generation = _load_prompt("post_generation")
   
   G["T_total"] = timesteps + fill_history
   G["T_current"] = 0  # Simulation starts at timestep 0, history is at negative indices
   G["fill_history"] = fill_history  # Store fill_history in graph
   
   # First pass: create neighbors arrays
   for agent in G.vs:
        agent["identity"] = identity
        agent["likes_with_history"] = likes_with_history
        agent["post_generation"] = post_generation
        agent["neighbors"] = np.array([neighbor.index for neighbor in agent.neighbors()])
        agent["success"] = 0     # total number of likes received
        # number of likes given to each neighbor
        agent["preferred_neighbors"] = np.zeros(len(agent["neighbors"]))
        # k is the number of posts read per round.
        agent["read_history"] = np.full((timesteps, k, 2), -1, dtype=int)  # -1 for empty slots

   # Second pass: create lookout arrays (where am I in each neighbor's list?)
   for agent in G.vs:
       lookout = np.zeros(len(agent["neighbors"]), dtype=int)
       for i, neighbor_id in enumerate(agent["neighbors"]):
           neighbor_neighbors = G.vs[neighbor_id]["neighbors"]
           lookout[i] = np.where(neighbor_neighbors == agent.index)[0][0]
       agent["lookout"] = lookout
   
   return G
def _initialize_posts_and_weights(G, fill_history = 3, debug = False):
    max_degree = np.max(np.array(G.degree()))
    num_agents = len(G.vs)

    posts           = [[None] * G["T_total"] for _ in range(num_agents)]
    READ_MATRIX     = np.zeros((num_agents, G["T_total"], max_degree))
    WEIGHTS         = np.zeros((num_agents, G["T_total"], max_degree))
    LIKES           = np.zeros((num_agents, G["T_total"]))

    if debug:
        for i in range(num_agents):
            for j in range(0,fill_history):
                posts[i][j] = f"agent {i}:  at timestep {j-fill_history}"
    return posts, READ_MATRIX, WEIGHTS, LIKES
def _update_personal_weights(G, WEIGHTS, W_personal_weights):
    """ Fully vectorized update of personalized weights """
    num_agents = len(G.vs)
    max_degree = WEIGHTS.shape[2]
    
    # Pre-extract all data as numpy arrays
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
    
    # Vectorized weight updates
    for i in range(num_agents):
        n_neighbors = neighbor_counts[i]
        if n_neighbors == 0:
            continue
            
        # Get this agent's neighbors and preferences
        neighbor_ids = neighbors_data[i, :n_neighbors]
        preferences = preferred_data[i, :n_neighbors]
        lookout_positions = lookout_data[i, :n_neighbors]
        
        # Vectorized update: WEIGHTS[j, :, lookout[k]] *= (1 + pref[k] * W)
        multipliers = 1 + preferences * W_personal_weights
        WEIGHTS[neighbor_ids, :, lookout_positions] *= multipliers[:, np.newaxis]
def _calculate_weights(G, READ_MATRIX, LIKES, PARAMS):
    max_degree = np.max(np.array(G.degree()))
    num_agents = len(G.vs)
    fill_history = G["fill_history"]
    WEIGHTS = np.ones((num_agents, G["T_total"], max_degree))
    
    # Vectorized exponential time decay - handle history properly
    if PARAMS["time_decay_rate"] > 0:
        # Actual timesteps: [-fill_history, ..., -1, 0, 1, 2, ...]
        actual_timesteps = np.arange(G["T_total"]) - fill_history
        current_actual_time = G["T_current"]  # Current simulation time
        time_diffs = current_actual_time - actual_timesteps  # How far back in time
        decay_factors = np.exp(-PARAMS["time_decay_rate"] * time_diffs)
        WEIGHTS *= decay_factors[np.newaxis, :, np.newaxis]
    
    # Vectorized agent success
    if PARAMS.get("W_agent_success", 0) > 0:
        success = np.array(G.vs["success"])
        agent_multiplier = success * PARAMS["W_agent_success"]
        WEIGHTS *= agent_multiplier[:, np.newaxis, np.newaxis]
    
    # Vectorized post success
    if PARAMS.get("W_post_success", 0) > 0:
        post_multiplier = LIKES * PARAMS["W_post_success"]
        WEIGHTS *= post_multiplier[:, :, np.newaxis]
    
    if PARAMS.get("W_personal_weights", 0) > 0:
        _update_personal_weights(G, WEIGHTS, PARAMS["W_personal_weights"])

    # Add noise
    if PARAMS.get("noise_level", 0) > 0:
        WEIGHTS += np.random.normal(0, PARAMS["noise_level"], WEIGHTS.shape)
    
    # set weights to 0 if the post has been read
    WEIGHTS[READ_MATRIX == 1] = 0

    # set weights to 0 for future timesteps (after current simulation time)
    current_timestep_index = G["T_current"] + fill_history
    future_mask = np.arange(G["T_total"]) > current_timestep_index
    WEIGHTS[:, future_mask, :] = 0

    return WEIGHTS
def _find_top_k_posts(G, WEIGHTS, k=5):
    """ Find top k highest weighted posts for each agent from their neighbors """
    num_agents = len(G.vs)
    top_posts = np.full((num_agents, k, 2), -1, dtype=int)  # [agent_id, rank, (author_id, timestep)]
    
    for agent in G.vs:
        agent_id = agent.index
        neighbor_ids = agent["neighbors"]
        lookout_positions = agent["lookout"]
        
        if len(neighbor_ids) == 0:
            continue
            
        # Get all accessible weights: WEIGHTS[neighbor_ids, :, lookout_positions]
        accessible_weights = WEIGHTS[neighbor_ids, :, lookout_positions]
        
        # Flatten and get indices of top k
        flat_weights = accessible_weights.flatten()
        top_k_flat_indices = np.argpartition(flat_weights, -k)[-k:]
        top_k_flat_indices = top_k_flat_indices[np.argsort(flat_weights[top_k_flat_indices])[::-1]]  # Sort descending
        
        # Convert flat indices back to (neighbor_position, timestep)
        neighbor_positions, timesteps = np.unravel_index(top_k_flat_indices, accessible_weights.shape)
        
        # Convert to actual agent IDs and store
        for rank in range(min(k, len(top_k_flat_indices))):
            author_id = neighbor_ids[neighbor_positions[rank]]
            timestep = timesteps[rank]
            top_posts[agent_id, rank] = [author_id, timestep]
    
    return top_posts
def _mark_posts_as_read(G, READ_MATRIX, top_posts):
    """ Mark selected top posts as read in the READ_MATRIX - vectorized """
    num_agents, k, _ = top_posts.shape
    
    # Get all valid posts (not -1 padding)
    valid_mask = (top_posts[:, :, 0] != -1) & (top_posts[:, :, 1] != -1)
    valid_agents, valid_ranks = np.where(valid_mask)
    
    if len(valid_agents) == 0:
        return
    
    # Extract author_ids and timesteps for valid posts
    author_ids = top_posts[valid_agents, valid_ranks, 0]
    timesteps = top_posts[valid_agents, valid_ranks, 1]
    
    # Pre-build lookup arrays
    max_degree = READ_MATRIX.shape[2]
    neighbor_lookup = np.full((num_agents, max_degree), -1, dtype=int)
    lookout_lookup = np.full((num_agents, max_degree), -1, dtype=int)
    
    for agent_id, agent in enumerate(G.vs):
        neighbors = agent["neighbors"]
        lookouts = agent["lookout"]
        neighbor_lookup[agent_id, :len(neighbors)] = neighbors
        lookout_lookup[agent_id, :len(neighbors)] = lookouts
    
    # Vectorized marking
    for i in range(len(valid_agents)):
        agent_id = valid_agents[i]
        author_id = author_ids[i]
        timestep = timesteps[i]
        
        # Find author in agent's neighbors
        neighbor_positions = np.where(neighbor_lookup[agent_id] == author_id)[0]
        if len(neighbor_positions) > 0:
            lookout_pos = lookout_lookup[agent_id, neighbor_positions[0]]
            READ_MATRIX[author_id, timestep, lookout_pos] = 1
    return READ_MATRIX
def _update_read_list(G, top_posts):
    """Update each agent's read_history array with what they read this timestep"""
    num_agents, _, _ = top_posts.shape
    current_timestep = G["T_current"]
    
    for agent_id in range(num_agents):
        # Copy the top_posts for this agent directly into their read_history
        # top_posts[agent_id] is shape (k, 2) - exactly what we need
        G.vs[agent_id]["read_history"][current_timestep, :, :] = top_posts[agent_id]
def run_simulation(G, WEIGHTS, READ_MATRIX, LIKES, PARAMS, POSTS, fill_history, timesteps):
    for i in range(fill_history, fill_history + timesteps):
        G["T_current"] = i - fill_history  # Update simulation time first
        current_timestep_index = i  # This is the array index for new posts
        
        WEIGHTS = _calculate_weights(G, READ_MATRIX, LIKES, PARAMS)
        TOP_POSTS = _find_top_k_posts(G, WEIGHTS, k=2)
        READ_MATRIX = _mark_posts_as_read(G, READ_MATRIX, TOP_POSTS)
        _update_read_list(G, TOP_POSTS)
        
        for agent in G.vs:
            agent_id = agent.index
            for p in range(len(TOP_POSTS[agent_id])):
                if _like_decision(agent, TOP_POSTS[agent_id, p]):
                    author_id = TOP_POSTS[agent_id, p, 0]
                    timestep = TOP_POSTS[agent_id, p, 1]
                    
                    if author_id == -1 or timestep == -1:
                        continue
                    
                    LIKES[author_id, timestep] += 1
                    G.vs[author_id]["success"] += 1
                    
                    author_neighbors = G.vs[author_id]["neighbors"]
                    neighbor_position = np.where(author_neighbors == agent_id)[0]
                    if len(neighbor_position) > 0:
                        G.vs[author_id]["preferred_neighbors"][neighbor_position[0]] += 1
            
            # Generate and store new post at current timestep
            post = _generate_post(agent)
            POSTS[agent_id][current_timestep_index] = post
    return G, POSTS, WEIGHTS, READ_MATRIX, LIKES