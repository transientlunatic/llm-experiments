"""Model catalogue and GPU-memory/context fitting, shared by every Igor job type."""
from __future__ import annotations

import json
import math
from pathlib import Path

DATA = Path(__file__).parent / "data"
DEFAULT_CATALOG = DATA / "models.json"

OUT_RESERVE = 3000   # tokens reserved for a model's output when sizing context
OVERHEAD_MB = 1500   # CUDA context + compute buffers (rough)


def load_catalog(path=None) -> list[dict]:
    return json.loads(Path(path or DEFAULT_CATALOG).read_text())["models"]


def find_model(name: str, models: list[dict]) -> dict | None:
    return next((m for m in models if m["name"] == name), None)


def vram_need(m: dict, ctx: int) -> float:
    """Approximate GPU memory (MB) for model `m` at context length `ctx` (weights + fp16 KV cache + overhead)."""
    return (m["weights_mb"] + m["kv_mb_per_1k"] * ctx / 1000 + OVERHEAD_MB) * 1.1


def vram_request(m: dict, ctx: int, budget: int) -> int:
    """The `GlobalMemoryMb` requirement to put in a submit file, rounded up to 1000 MB and capped at the budget."""
    return min(budget, math.ceil(vram_need(m, ctx) / 1000) * 1000)


def fit(m: dict, tokens: int, budget: int, ctx_cap: int):
    """Return (mode, ctx) for model `m` on a prompt of `tokens`, or None if it cannot run within `budget` MB.

    mode is "single" if everything fits in one context window, else "chunked" (document read in pieces)."""
    need = tokens + OUT_RESERVE + 800
    single = math.ceil(need / 4096) * 4096
    limit = min(m["max_ctx"], ctx_cap)
    if single <= limit and vram_need(m, single) <= budget:
        return "single", single
    ctx = min(limit, max(single - 4096, 8192)) // 4096 * 4096
    while ctx >= 8192:
        if vram_need(m, ctx) <= budget:
            return "chunked", ctx
        ctx -= 4096
    return None


def score(m: dict, mode: str) -> float:
    return m["quality"] - (1.5 if mode == "chunked" else 0)
