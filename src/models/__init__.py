from .llm import agent as LLM  # module import

_MODELS = {
    "LLM": LLM,
}

def get_model(name: str):
    try:
        return _MODELS[name]
    except KeyError:
        raise ValueError(f"Unknown opinion model: {name}. Available: {list(_MODELS)}")
