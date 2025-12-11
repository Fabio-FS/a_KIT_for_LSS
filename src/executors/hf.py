# executors/hf.py
import asyncio
import aiohttp
import numpy as np
from typing import Any, List

_global_session: aiohttp.ClientSession | None = None


async def get_session() -> aiohttp.ClientSession:
    """Return a global aiohttp session (created on first use)."""
    global _global_session
    if _global_session is None or _global_session.closed:
        _global_session = aiohttp.ClientSession()
    return _global_session


async def close_session_if_any():
    """Close the global session if it exists."""
    global _global_session
    if _global_session and not _global_session.closed:
        await _global_session.close()
    _global_session = None


def _digit_to_prob(s: str) -> float:
    """
    Map the first digit found in string s to a probability in [0,1] via d/9.
    Fallback is 0.5 if no digit is present.
    """
    s = (s or "").strip()
    for ch in s:
        if ch.isdigit():
            d = int(ch)
            if d < 0:
                d = 0
            if d > 9:
                d = 9
            return d / 9.0
    raise ValueError(f"No digit found in string: '{s}'")

async def execute_prompts_parallel(
    prompts: List[Any],
    PARAMS: dict,
    *,
    max_tokens: int = 1,
    temperature: float = 1.2,
    parse: str | None = "digit_to_prob",
    timeout: float = 60.0,
    retries: int = 2,
    concurrency: int | None = None,
) -> np.ndarray | list[str]:
    """
    Execute LLM calls in parallel for a list of prompts.

    Args
    ----
    prompts:
        List of prompts. Each item can be:
          - a string (will be wrapped as a single user message), or
          - a list of OpenAI-style chat messages:
              [{"role": "system"|"user"|"assistant", "content": "..."}]
    PARAMS: dict with keys:
        - "API_URL": str   (OpenAI-compatible /v1/chat/completions endpoint)
        - "MODEL": str
        - "HF_TOKEN": str (optional, can be empty for local vLLM)
    max_tokens: tokens to request from the model.
    temperature: sampling temperature.
    parse:
        - "digit_to_prob" -> returns np.ndarray[float] in [0,1]
        - None or "raw"   -> returns list[str] (raw text)
    timeout: per-request timeout, seconds.
    retries: number of retry attempts on transient errors.
    concurrency: max in-flight requests; if None, defaults to min(64, len(prompts)).

    Returns
    -------
    np.ndarray[float] if parse == "digit_to_prob", else list[str].
    """
    if not prompts:
        raise ValueError("no prompts")
        
    session = await get_session()
    api_url = PARAMS["API_URL"]
    model = PARAMS["MODEL"]
    token = PARAMS.get("HF_TOKEN", "")
    headers = {"Authorization": f"Bearer {token}"} if token else {}

    if concurrency is None:
        concurrency = min(64, max(1, len(prompts)))

    sem = asyncio.Semaphore(concurrency)

    def _as_messages(p: Any) -> list[dict]:
        if isinstance(p, list):
            return p
        return [{"role": "user", "content": str(p)}]

    async def one_call(idx: int, prompt: Any):
        payload = {
            "messages": _as_messages(prompt),
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        attempt = 0
        while True:
            attempt += 1
            try:
                async with sem:
                    async with session.post(
                        api_url, headers=headers, json=payload, timeout=timeout
                    ) as resp:
                        if resp.status >= 500 or resp.status == 429:
                            text = await resp.text()
                            raise aiohttp.ClientResponseError(
                                request_info=resp.request_info,
                                history=resp.history,
                                status=resp.status,
                                message=text,
                                headers=resp.headers,
                            )
                        data = await resp.json()
                        
                        if "choices" not in data:
                            print(f"ERROR - API response: {data}")
                            raise KeyError(f"'choices' not in response")

                content = data["choices"][0]["message"]["content"]
                if parse == "digit_to_prob":
                    return idx, _digit_to_prob(content)
                else:
                    return idx, str(content)

            except Exception as e:
                if attempt > retries:
                    raise Exception(f"API call failed after {retries} retries for prompt {idx}: {e}")
                
                await asyncio.sleep(0.05 * attempt)

    tasks = [one_call(i, p) for i, p in enumerate(prompts)]
    results = await asyncio.gather(*tasks)

    results.sort(key=lambda pair: pair[0])
    values = [v for _, v in results]

    if parse == "digit_to_prob":
        return np.asarray(values, dtype=float)
    else:
        return values