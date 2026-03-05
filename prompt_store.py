from pathlib import Path


_BASE = Path(__file__).resolve().parent


def load_prompt(name: str) -> str:
    p = _BASE / "prompts" / name
    return p.read_text(encoding="utf-8")
