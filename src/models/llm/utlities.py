
from src.executors.hf import execute_prompts_parallel  # your async batch HF caller+


def load_usernames(filename="1024_Realistic_Usernames.csv") -> list[str]:
    import csv
    with open(filename, "r") as f:
        reader = csv.reader(f)
        # like return [row[0] for row in reader], but skip the first row
        return [row[0] for row in reader][1:]



def _load_prompt_yaml(field_name: str) -> str:
    import yaml
    with open("prompts.yaml", "r") as f:
        data = yaml.safe_load(f)
    if field_name not in data or not str(data[field_name]).strip():
        raise KeyError(f"Missing prompt: {field_name}")
    return data[field_name]

def _compose_prompt_for_warmup(agent, t, POSTS):
    """
    First-post prompt:
      - System: agent["warmup"] (persona + stable rules)
      - User: minimal TASK + strict output contract
    """
    PROMPT = [{"role": "system", "content": agent["warmup"]}]

    user_msg = (
        "CONTEXT START\n"
        "This is your first post. You have not read any messages yet.\n\n"
        "TASK:\nWrite your first post about the war in Ukraine. "
        "Do NOT reply to anyone. Keep it ≤ 3 sentences.\n\n"
        "OUTPUT FORMAT (STRICT):\nReturn ONLY the post text (no quotes, no preface).\n"
        "CONTEXT END"
    )

    PROMPT.append({"role": "user", "content": user_msg})
    return PROMPT
    
def _compose_prompt_for_likes(agent, t, POSTS, coord, memory=10):
    

    PROMPT = [{"role": "system", "content": agent["likes"]}]

    # Collect the agent's recent posts (memory window)
    start = max(0, t - memory)
    recent_posts = []
    for i in range(start, t):
        post = POSTS[agent.index][i]
        if post:
            recent_posts.append(post.strip())

    # Target message to evaluate
    _, author_id, post_t = coord
    target_post = POSTS[author_id][post_t]
    target_text = str(target_post).strip() if target_post else "[EMPTY POST]"

    # Build the structured user message
    history_block = "\n".join(f"you: {p}" for p in recent_posts) or "(no prior posts)"
    user_msg = (
        "CONTEXT START\n"
        f"YOUR RECENT POSTS:\n{history_block}\n\n"
        f"MESSAGE TO RATE:\n@{author_id}: {target_text}\n\n"
        "TASK:\nEvaluate how much you like this message given your personality and past posts.\n\n"
        "OUTPUT FORMAT (STRICT):\nRATING=<0-9>\n"
        "CONTEXT END"
    )

    PROMPT.append({"role": "user", "content": user_msg})
    return PROMPT

def _compose_prompt_for_posts(G, agent, t, POSTS, memory=10):
    """
    Build a structured posting prompt.

    - System message: agent["posts"]  (already formatted from prompts.yaml)
    - User message: READ_HISTORY + YOUR_PREVIOUS_POSTS + TASK + strict output contract
    - Returns (PROMPT, coord) where coord = [agent_id, t]
    """
    # System: persona + stable rules (already in agent["posts"])
    PROMPT = [{"role": "system", "content": agent["posts"]}]

    names = G.vs["name"]
    me = agent.index
    start = max(0, t - memory)

    # READ_HISTORY (what this agent read from others)
    read_lines = []
    for i in range(start, t):
        for author_id, post_t in agent["read_history"][i]:
            if author_id == -1 or post_t == -1:
                continue
            txt = POSTS[author_id][post_t]
            if not txt:
                continue
            who = names[author_id]
            read_lines.append(f"@{who}: {str(txt).strip()}")

    # YOUR_PREVIOUS_POSTS (this agent's own past posts)
    my_lines = []
    for i in range(start, t):
        my_txt = POSTS[me][i]
        if my_txt:
            my_lines.append(f"you: {str(my_txt).strip()}")

    read_block = "\n".join(read_lines) if read_lines else "(none)"
    my_block = "\n".join(my_lines) if my_lines else "(none)"

    # User message with explicit context and tight output contract
    user_msg = (
        "CONTEXT START\n"
        f"READ_HISTORY:\n{read_block}\n\n"
        f"YOUR_PREVIOUS_POSTS:\n{my_block}\n\n"
        "TASK:\nWrite your next post NOW. Either reply to one of the above using @username, "
        "or write a new post if you prefer. Keep it ≤ 3 sentences.\n\n"
        "OUTPUT FORMAT (STRICT):\n"
        "Return ONLY the post text (no quotes, no preface).\n"
        "CONTEXT END"
    )

    PROMPT.append({"role": "user", "content": user_msg})
    coord = [me, t]
    return PROMPT, coord

async def _execute_prompts_with_coords(prompts, coords3, PARAMS, max_tokens=128, batch_size=512, temperature=1.2):
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
           temperature=temperature,  # Pass it through
           parse=None
       )
       for text in texts:
           result_coords.append(coords3[write_idx])
           result_texts.append(text)
           write_idx += 1
   
   return result_coords, result_texts


def print_prompt(prompt):
    print("=" * 50)
    for _, msg in enumerate(prompt):
        role = msg['role'].upper()
        content = msg['content']
        print(f"{role}:")
        print(content)
        print("-" * 30)
    print()