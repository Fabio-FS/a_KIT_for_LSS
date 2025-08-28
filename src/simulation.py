import numpy as np
import igraph as ig
from src.agent_LLM import _load_prompt

from interfaces import _generate_post, _like_decision, _generate_post_warmup, _initialize_agents




def run_simulation(PARAMS):
    num_agents, neighbors, timesteps, fill_history, seed = PARAMS["num_agents"], PARAMS["neighbors"], PARAMS["timesteps"], PARAMS["fill_history"], PARAMS["seed"]


    if seed is not None:
        np.random.seed(seed)
    else:
        np.random.seed(42)

    G = generate_network(N = num_agents, neighbors = neighbors, p = 0.2, timesteps = timesteps, fill_history = fill_history)
    POSTS, WEIGHTS, READ_MATRIX, LIKES = _initialize_posts_and_weights(G, debug = True)

    # Changing from here on out. We start the simulation at time steps "fill_history". Otherwise, the negative indices are a pain.
    for i in range(fill_history, fill_history + timesteps):
        G["T_current"] = i  # Update simulation time first
        current_timestep_index = i  # This is the array index for new posts
        print(f"T_current: {G['T_current']}")
        
        # Calculate weights, find top k posts, and mark posts as read
        WEIGHTS, TOP_POSTS, READ_MATRIX = _matrix_operations(G, READ_MATRIX, LIKES, PARAMS, WEIGHTS, TOP_POSTS, k = 2)

        # WEIGHTS [sender, timestep, receiver_lookout_position], TOP_POSTS [receiver, rank, (author_id, timestep)], READ_MATRIX [author_id, timestep, receiver_lookout_position]
        # !! always access receiver_lookout_position from the receiver's perspective using the lookup table

        for agent in G.vs:
            agent_id = agent.index
            for p in range(len(TOP_POSTS[agent_id])):
                sender_id = TOP_POSTS[agent_id, p, 0]
                timestep  = TOP_POSTS[agent_id, p, 1]

                # guard against padding
                if sender_id == -1 or timestep == -1:
                    print(f"Padding at timestep {G['T_current']}")
                    # I don't expect this to ever happen.
                    continue

                post = POSTS[sender_id][timestep]
                if post is None:
                    print(f"Post of agent {sender_id} is None at timestep {G['T_current']}")
                    # I don't expect this to ever happen.
                    continue
                like_decision = _like_decision(agent, G["T_current"], POSTS, post, PARAMS)
                if like_decision:
                    LIKES[sender_id, timestep] += 1
                    G.vs[sender_id]["success"] += 1

                    # now I check which is the position of the sender in the receiver's neighbor list

                    position = np.where(agent["neighbors"] == sender_id)[0]
                    agent["preferred_neighbors"][position] += 1
            
            # Generate and store new post at current timestep
            post = _generate_post(agent, G["T_current"], POSTS, PARAMS)
            POSTS[agent_id][G["T_current"]] = post
    return G, POSTS, WEIGHTS, READ_MATRIX, LIKES





def _matrix_operations(G, READ_MATRIX, LIKES, PARAMS, WEIGHTS, TOP_POSTS, k = 2):
    WEIGHTS = _calculate_weights(G, READ_MATRIX, LIKES, PARAMS)

    # Find top k posts (k is the number of posts read per round)
    TOP_POSTS = _find_top_k_posts(G, WEIGHTS, k=k)
        
    # Mark posts as read
    READ_MATRIX = _mark_posts_as_read(G, READ_MATRIX, TOP_POSTS)

    if PARAMS["opinion_model"] == "LLM":
        # Update read list (this is useful for the LLM generation)
        _update_read_list(G, TOP_POSTS)
    else:
        pass
    return WEIGHTS, TOP_POSTS, READ_MATRIX







def build_neighbor_lookups(G):
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
def generate_network(PARAMS,N = 5, neighbors = 4, p = 0.05, timesteps = 10, fill_history = 3, k = 2):
    N, neighbors, timesteps, fill_history, k = PARAMS["num_agents"], PARAMS["neighbors"], PARAMS["timesteps"], PARAMS["fill_history"], PARAMS["k"]
    G = ig.Graph.Watts_Strogatz(dim=1, size=N, nei=neighbors, p=p)





    G["T_total"] = timesteps + fill_history
    G["T_current"] = 0  # Simulation starts at timestep 0, history is at negative indices
    G["fill_history"] = fill_history  # Store fill_history in graph

    # First pass: create neighbors arrays
    for agent in G.vs:

        agent["neighbors"] = np.array([neighbor.index for neighbor in agent.neighbors()])
        agent["success"] = 0     # total number of likes received
        # number of likes given to each neighbor
        agent["preferred_neighbors"] = np.zeros(len(agent["neighbors"]))
        # k is the number of posts read per round.
        agent["read_history"] = np.full((timesteps, k, 2), -1, dtype=int)  # -1 for empty slots

    build_neighbor_lookups(G)


    # initialize agent_LLM stuff
    _initialize_agents(G, PARAMS)
    return G
def _initialize_posts_and_weights(G, PARAMS):
    """
    Allocate POSTS/READ_MATRIX/WEIGHTS/LIKES and 'thermalize'
    the first `fill_history` slots in POSTS with self-only warmup posts.

    When debug=True, warmup uses deterministic strings (no API calls).
    """
    max_degree = np.max(np.array(G.degree()))
    num_agents = len(G.vs)

    if PARAMS["opinion_model"] == "LLM":
        POSTS       = [[None] * G["T_total"] for _ in range(num_agents)]
    else:
        print("unknown opinion model")
        exit()
    READ_MATRIX = np.zeros((num_agents, G["T_total"], max_degree), dtype=float)
    WEIGHTS     = np.zeros((num_agents, G["T_total"], max_degree), dtype=float)
    LIKES       = np.zeros((num_agents, G["T_total"]), dtype=float)

    # Always fill the warmup slots if fill_history > 0
    if PARAMS["fill_history"] > 0:
        _thermalize_system(G, POSTS, PARAMS)
    return POSTS, READ_MATRIX, WEIGHTS, LIKES
def _thermalize_system(G, POSTS, PARAMS):
    
    for t_idx in range(PARAMS["fill_history"]):  # array indices 0..fill_history-1 (actual time -fill_history..-1)
        for agent in G.vs:
            if PARAMS["opinion_model"] == "LLM":
                post = _generate_post_warmup(agent, 
                                            current_timestep_index=t_idx, 
                                            POSTS=POSTS, 
                                            self_memory=PARAMS["self_memory"], 
                                            debug=PARAMS["debug"],
                                            opinion_model=PARAMS["opinion_model"])
                POSTS[agent.index][t_idx] = post
            else:
                print("unknown opinion model")
                exit()


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
        time_diffs = G["T_current"] - np.arange(G["T_total"])  
        decay_factors = np.exp(-PARAMS["time_decay_rate"] * time_diffs)
        WEIGHTS *= decay_factors[np.newaxis, :, np.newaxis]
    
    # Vectorized agent success
    if PARAMS.get("W_agent_success", 0) > 0:
        success = np.array(G.vs["success"])
        agent_multiplier =  1.0 + success * PARAMS["W_agent_success"]
        WEIGHTS *= agent_multiplier[:, np.newaxis, np.newaxis]
    
    # Vectorized post success
    if PARAMS.get("W_post_success", 0) > 0:
        post_multiplier =  1.0 + LIKES * PARAMS["W_post_success"]
        WEIGHTS *= post_multiplier[:, :, np.newaxis]
    
    if PARAMS.get("W_personal_weights", 0) > 0:
        _update_personal_weights(G, WEIGHTS, PARAMS["W_personal_weights"])


    # ensure that the weights are not negative
    WEIGHTS += 100.0

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
    

    neighbor_lookup = G["neighbor_lookup"]
    lookout_lookup  = G["lookout_lookup"]

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

