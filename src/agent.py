import json
import numpy as np
import requests
import re



def _load_prompt(field_name) -> str:
    """Load a prompt template from prompts.json."""
    try:
        with open("prompts.json", 'r') as f:
            data = json.load(f)
        result = data.get(field_name, f"Fallback prompt for {field_name}")
        return result
    except Exception as e:  # Changed from bare except
        return f"Error loading {field_name}"
    




def _build_history_of_written_posts(agent, current_timestep, POSTS, memory = 10):
    history = []
    for i in range(max(0, current_timestep - memory + 1), current_timestep + 1):
        if i < len(POSTS[agent.index]) and POSTS[agent.index][i] is not None:
            history.append(POSTS[agent.index][i])
    history = "\n".join(history)
    return history

def _build_history_of_recent_reads(agent, current_timestep, POSTS, memory = 5):
    history = []
    for t in range(max(0, current_timestep - memory + 1), current_timestep + 1):
        for k in range(agent["read_history"].shape[1]):  # k posts per timestep
            author_id, post_timestep = agent["read_history"][t, k]
            if author_id != -1 and post_timestep != -1:
                post = POSTS[author_id][post_timestep]
                if post is not None:
                    history.append(post)
    return "\n".join(history)


# ----------------------------------- #
#  LLM functions for post generation  #
# ----------------------------------- #








    # debug true implies testing mode, and no LLM stuff.
def _generate_post(G, POSTS, debug = False):
    if debug:
        post = "just a test post"
        return post
    else:
        self_history = _build_history_of_written_posts(G, G["current_timestep"], POSTS, memory = 10)
        read_history = _build_history_of_recent_reads(G, G["current_timestep"], POSTS, memory = 5)
        prompt = G["post_generation"].format(identity = G["identity"], 
                                             self_history = self_history, 
                                             read_history = read_history)








# ----------------------------------- #
#   LLM functions for like decision   #
# ----------------------------------- #




def _extract_number(response):
    numbers = re.findall(r'\d+', response)
    if numbers:
        num = int(numbers[0])
        return min(max(num, 0), 9)
    return 5
def _like_decision(agent, post, api_key, api_url, model, debug=False):
    if debug:
        return np.random.rand() < 0.5
    
    prompt = agent["likes_with_history"].format(
        identity=agent["identity"], 
        history=_build_history_of_written_posts(agent, memory=10), 
        message=post
    )
    
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "model": model
    }
    headers = {"Authorization": f"Bearer {api_key}"}
    
    response = requests.post(api_url, headers=headers, json=payload)
    content = response.json()["choices"][0]["message"]["content"]
    
    return np.random.rand() < _extract_number(content) / 9
