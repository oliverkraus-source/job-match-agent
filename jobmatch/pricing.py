"""Estimate what a run cost, from the token counts the API reports.

Prices in USD per million tokens (Anthropic list prices, October 2026; check
https://claude.com/pricing for current values). Cache writes cost 1.25x input,
cache reads are much cheaper - which is why the agent enables prompt caching.
"""

from __future__ import annotations

# model: (input, output, cache write, cache read)
PRICES: dict[str, tuple[float, float, float, float]] = {
    "claude-opus-5-5": (4.00, 20.00, 5.00, 0.20),
    "claude-sonnet-5-5": (2.00, 10.00, 2.50, 0.20),
    "claude-haiku-5-5": (0.10, 0.50, 0.125, 0.01),
}


def cost_usd(model: str, usage) -> float | None:
    """Cost of one response, or None if the model is not in the price table."""
    prices = PRICES.get(model)
    if prices is None:
        return None
    inp, out, write, read = prices
    return (
        (usage.input_tokens or 0) * inp
        + (usage.output_tokens or 0) * out
        + (usage.cache_creation_input_tokens or 0) * write
        + (usage.cache_read_input_tokens or 0) * read
    ) / 1_000_000
