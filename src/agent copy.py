import json
import numpy as np



def _load_prompt(field_name: str) -> str:
    """Load a prompt template from prompts.json."""
    try:
        with open("prompts.json", 'r') as f:
            data = json.load(f)
        result = data.get(field_name, f"Fallback prompt for {field_name}")
        return result
    except Exception as e:  # Changed from bare except
        return f"Error loading {field_name}"
    
def _generate_post(G):
    post = "just a test post"
    return post

def _like_decision(agent, post):
    return np.random.rand() < 0.5
