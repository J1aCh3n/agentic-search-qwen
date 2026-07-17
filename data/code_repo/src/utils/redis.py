"""Redis helpers."""

def get_cache_key(prefix: str, value: str) -> str:
    return f"{prefix}:{value}"
