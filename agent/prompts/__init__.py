from pathlib import Path

_PROMPTS_DIR = Path(__file__).parent


def load(name: str) -> str:
    """Load a system prompt by name (without extension) from this directory."""
    return (_PROMPTS_DIR / f"{name}.txt").read_text(encoding="utf-8")
