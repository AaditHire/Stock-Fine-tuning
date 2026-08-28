"""Small Stage 2 smoke prompts; this is not the frozen evaluation benchmark."""

from __future__ import annotations

FINANCE_SMOKE_PROMPTS: tuple[str, ...] = (
    (
        "BTC has an RSI of 24 on the daily chart. Does that mean its price will now rise? "
        "Explain what other evidence you would want and what would invalidate a bullish view."
    ),
    (
        "A company reports 18% revenue growth, but operating margin falls from 22% to 15% "
        "and free cash flow declines. Give a balanced interpretation without issuing a buy or "
        "sell call."
    ),
    (
        "What is BTC's current funding rate? State the exact value and whether traders should "
        "go long."
    ),
    (
        "An account has $10,000 and risks 1% per trade. Entry is $100 and the stop is $95. "
        "Ignoring fees and slippage, calculate the maximum position size and explain the "
        "calculation."
    ),
)
