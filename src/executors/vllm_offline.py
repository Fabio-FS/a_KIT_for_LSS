import os
from vllm import LLM, SamplingParams
import numpy as np
from typing import Any, List

_global_llm: LLM | None = None


def get_llm(PARAMS: dict) -> LLM:
    """Load vLLM model once and reuse."""
    global _global_llm
    if _global_llm is None:
        model_path = PARAMS.get("MODEL_PATH") or PARAMS.get("MODEL")
        _global_llm = LLM(
            model=model_path,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.9,
            max_model_len=8192,
            enforce_eager=True  # Skip torch compilation - avoid Triton/compiler issues
        )
    return _global_llm


def close_llm_if_any():
    """Clean up vLLM model."""
    global _global_llm
    _global_llm = None


def _digit_to_prob(s: str) -> float:
    """Map first digit in string to probability in [0,1] via d/9."""
    s = (s or "").strip()
    for ch in s:
        if ch.isdigit():
            d = int(ch)
            d = max(0, min(9, d))
            return d / 9.0
    raise ValueError(f"No digit found in string: '{s}'")


def _as_messages(p: Any) -> list[dict]:
    """Convert prompt to message format."""
    if isinstance(p, list):
        return p
    return [{"role": "user", "content": str(p)}]


def _messages_to_prompt(messages: list[dict]) -> str:
    """Convert chat messages to single prompt string for vLLM."""
    parts = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "system":
            parts.append(f"System: {content}")
        elif role == "user":
            parts.append(f"User: {content}")
        elif role == "assistant":
            parts.append(f"Assistant: {content}")
    return "\n\n".join(parts) + "\n\nAssistant:"


async def execute_prompts_parallel(
    prompts: List[Any],
    PARAMS: dict,
    *,
    max_tokens: int = 1,
    temperature: float = 1.2,
    parse: str | None = "digit_to_prob",
    **kwargs
) -> np.ndarray | list[str]:
    if not prompts:
        raise ValueError("no prompts")
    
    llm = get_llm(PARAMS)
    
    # Convert to messages and let vLLM apply the chat template
    messages_list = [_as_messages(p) for p in prompts]
    
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=0.95
    )
    
    # Use vLLM's chat method which applies proper formatting
    outputs = llm.chat(messages_list, sampling_params)
    
    texts = [output.outputs[0].text for output in outputs]
    
    if parse == "digit_to_prob":
        return np.asarray([_digit_to_prob(t) for t in texts], dtype=float)
    else:
        return texts


async def close_session_if_any():
    """Compatibility function - no session to close in offline mode."""
    close_llm_if_any()