
from src.executors.hf import execute_prompts_parallel  # your async batch HF caller+


def load_usernames(filename="1024_Realistic_Usernames.csv") -> list[str]:
    import csv
    with open(filename, "r") as f:
        reader = csv.reader(f)
        return [row[0] for row in reader]



def _load_prompt_yaml(field_name: str) -> str:
    import yaml
    with open("prompts.yaml", "r") as f:
        data = yaml.safe_load(f)
    if field_name not in data or not str(data[field_name]).strip():
        raise KeyError(f"Missing prompt: {field_name}")
    return data[field_name]




def _compose_prompt_for_warmup(agent, t, POSTS):
    try:
        lr = agent["left_right"]
    except KeyError:
        lr = "center"  # default

    prompt = [{"role": "system",
               "content": _load_prompt_yaml("identity_for_warmup").format(left_right=lr)}]
    prompt.append({"role": "user", "content": "Please write your post"})
    return prompt
    
def _compose_prompt_for_likes(agent, t, POSTS, coord, memory=10):
    """
    compose the prompt for the likes decision
    The only factor to decide the like decision is the list of messages that the agent has wrote.

    coord is the coordinate of the post to be evaluated
    memory is the number of messages in the past to be considered for the like decision
    t is the current timestep
    """
    try:
        lr = agent["left_right"]
    except KeyError:
        lr = "center"

    system_msg = _load_prompt_yaml("identity_for_likes").format(left_right=lr)
    PROMPT = [{"role": "system", "content": system_msg}]

    # collect up to `memory` past posts before timestep t
    start = max(0, t - memory)
    for i in range(start, t):
        post = POSTS[agent.index][i]
        if post:
            PROMPT.append({"role": "assistant", "content": str(post).strip()})

    _, author_id, post_t = coord
    target_post = POSTS[author_id][post_t]
    PROMPT.append({"role": "user", "content": f"now evaluate this message: {target_post}"})

    return PROMPT

def _compose_prompt_for_posts(G, agent, t, POSTS, memory=10):
    try:
        lr = agent["left_right"]
    except KeyError:
        lr = "center"
    sys_text = _load_prompt_yaml("identity_for_posts").format(left_right=lr)  # your identity block
    PROMPT = [{"role": "system", "content": sys_text}]

    names = G.vs["name"] if "name" in G.vs.attributes() else None
    me = agent.index
    start = max(0, t - memory)
    for i in range(start, t):
        # READs
        for author_id, post_t in agent["read_history"][i]:
            if author_id == -1 or post_t == -1: 
                continue
            txt = POSTS[author_id][post_t]
            if not txt: 
                continue
            who = names[author_id] if names else f"agent_{author_id}"
            PROMPT.append({"role": "user", "content": f"t-{t-i} READ @{who}: \"{txt.strip()}\""})

        # WROTE (me)
        my_txt = POSTS[me][i]
        if my_txt:
            PROMPT.append({"role": "assistant", "content": f"t-{t-i} I WROTE: \"{my_txt.strip()}\""})
        coord = [me, t]
    # final instruction
    PROMPT.append({"role": "user", "content": "Write your next post (max 3 statements). Return only the post text."})
    return PROMPT, coord

async def _execute_prompts_with_coords(prompts, coords3, PARAMS, max_tokens=128, batch_size=128):
   """
   Execute prompts in batches and return coordinates and texts as separate lists.
   coords3 is the coordinate of the post to be evaluated
   [0] is reader index, [1] is sender index, [2] is timestep
   """
   def _chunks(seq, size):
       for i in range(0, len(seq), size):
           yield i, seq[i:i+size]

   result_coords = []
   result_texts = []
   write_idx = 0
   for start, chunk in _chunks(prompts, batch_size):
       texts = await execute_prompts_parallel(
           chunk,
           PARAMS,
           max_tokens=max_tokens,
           parse=None
       )
       for text in texts:
           
           result_coords.append(coords3[write_idx])
           result_texts.append(text)
           write_idx += 1
   
   return result_coords, result_texts