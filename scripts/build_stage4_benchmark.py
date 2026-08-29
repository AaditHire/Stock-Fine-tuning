# ruff: noqa: E501
"""Build the reviewed, frozen FinPulse evaluation benchmark and integrity manifest.

Long lines are retained in the reviewed question tables so each prompt, rubric, and answer stays
visually atomic. Executable project modules remain subject to the normal 100-character limit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "eval" / "finpulse_eval_v1.jsonl"
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "eval" / "finpulse_eval_v1.manifest.json"


def _base_case(
    case_id: str,
    category: str,
    prompt: str,
    checks: list[dict[str, Any]],
    tags: list[str],
    kind: str = "original",
) -> dict[str, Any]:
    return {
        "id": case_id,
        "category": category,
        "prompt": prompt,
        "checks": checks,
        "tags": tags,
        "provenance": {
            "kind": kind,
            "creator": "finpulse-llm",
            "license": "project-original",
        },
        "split": "eval",
        "exclude_from_training": True,
    }


def _mcq(
    case_id: str,
    category: str,
    stem: str,
    options: tuple[str, str, str, str],
    answer: str,
    tags: list[str],
) -> dict[str, Any]:
    rendered = "\n".join(
        f"{letter}. {option}" for letter, option in zip("ABCD", options, strict=True)
    )
    prompt = f"{stem}\n{rendered}\nExplain briefly, then end exactly with FINAL: {answer}"
    return _base_case(
        case_id,
        category,
        prompt,
        [{"type": "regex", "pattern": rf"FINAL:\s*{answer}\s*$"}],
        ["multiple_choice", *tags],
    )


def _rubric(
    case_id: str,
    category: str,
    prompt: str,
    required_groups: list[list[str]],
    tags: list[str],
    forbidden: str = r"\b(guaranteed|certainly|definitely|must (?:rise|fall))\b",
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = [
        {"type": "contains_any", "values": group} for group in required_groups
    ]
    checks.append({"type": "regex_not", "pattern": forbidden})
    return _base_case(case_id, category, prompt, checks, ["qualitative", *tags])


def _exact(
    case_id: str,
    category: str,
    prompt: str,
    final_pattern: str,
    tags: list[str],
) -> dict[str, Any]:
    return _base_case(
        case_id,
        category,
        f"{prompt} End with the word FINAL, a colon, and only your numeric answer.",
        [{"type": "regex", "pattern": rf"FINAL:\s*{final_pattern}\s*$"}],
        ["calculation", *tags],
        kind="synthetic",
    )


def _technical_analysis() -> list[dict[str, Any]]:
    category = "technical_analysis"
    knowledge = [
        ("A market forms successive higher highs and higher lows. What structure is present?", ("Uptrend", "Downtrend", "Range only", "No usable structure"), "A", ["structure"]),
        ("Price closes above established resistance on expanding volume. Which interpretation is best?", ("A guaranteed rally", "Stronger breakout confirmation, still requiring invalidation", "Automatic short signal", "Volume makes structure irrelevant"), "B", ["breakout", "volume"]),
        ("Price trades above resistance intraday but closes back below it. What is this commonly called?", ("Confirmed breakout", "Golden cross", "Potential fakeout", "Bullish divergence"), "C", ["fakeout"]),
        ("Daily RSI is 22 during a strong downtrend. What follows logically?", ("Price must bounce", "The asset is cheap fundamentally", "Momentum is weak, but RSI alone does not time a reversal", "A new uptrend is confirmed"), "C", ["rsi"]),
        ("Price makes a lower low while RSI makes a higher low. What is the pattern?", ("Bearish divergence", "Bullish divergence", "Hidden bearish divergence", "No divergence"), "B", ["divergence", "rsi"]),
        ("ATR rises sharply. What does ATR directly indicate?", ("Bullish direction", "Bearish direction", "Higher realized range or volatility", "Overvaluation"), "C", ["atr"]),
        ("Why is an EMA crossover considered lagging?", ("It uses derivatives data", "It is calculated from past prices", "It predicts earnings", "It ignores price"), "B", ["ema"]),
        ("Price rises while volume steadily declines. Which is the most careful reading?", ("The rally is guaranteed", "Participation may be weakening, so seek confirmation", "Volume never matters", "A crash is certain"), "B", ["volume", "momentum"]),
    ]
    cases = [
        _mcq(f"eval_ta_{index:03d}", category, *item)
        for index, item in enumerate(knowledge, start=1)
    ]
    scenarios = [
        ("Price tests the same support zone three times and rebounds, but each rebound is smaller. Explain what the evidence suggests and give a bearish invalidation trigger.", [["support", "demand"], ["weaken", "smaller", "compression"], ["break", "close below", "invalidation"]], ["support"]),
        ("Price sweeps below equal lows, immediately reclaims the level, and volume expands. Explain the bullish possibility, one confirmation, and one invalidation.", [["liquidity", "sweep"], ["reclaim", "close above", "follow-through"], ["below", "invalidation", "lose"]], ["liquidity_sweep"]),
        ("A range breaks upward on very low volume and returns inside the range. Assess the setup without predicting certainty.", [["fakeout", "failed breakout"], ["volume", "participation"], ["range", "invalidation", "reclaim"]], ["breakout"]),
        ("MACD crosses bullish below the zero line while price remains below a falling 200-day SMA. Explain the conflict.", [["momentum", "MACD"], ["downtrend", "200", "long-term"], ["confirmation", "uncertain", "could"]], ["macd", "sma"]),
        ("Price is above a rising 200-day SMA but below a falling 20-day EMA. Describe the two time-horizon signals and an invalidation level conceptually.", [["long-term", "200"], ["short-term", "20", "pullback"], ["invalidation", "structure", "swing"]], ["multi_timeframe"]),
        ("After a higher high, price breaks below the prior higher low. Explain the structural warning and what would confirm a bearish shift.", [["structure", "trend"], ["lower low", "break"], ["confirmation", "lower high", "follow-through"]], ["structure"]),
        ("Volume spikes at resistance, but the candle closes near its low and below resistance. Give a balanced interpretation.", [["rejection", "selling"], ["volume", "participation"], ["confirm", "follow-through", "next"]], ["volume", "resistance"]),
        ("Weekly structure is bullish while daily structure is bearish. Explain how a trader could frame this without collapsing the timeframes into one signal.", [["weekly", "higher timeframe"], ["daily", "lower timeframe"], ["horizon", "timeframe", "invalidation"]], ["multi_timeframe"]),
    ]
    cases.extend(
        _rubric(f"eval_ta_{index:03d}", category, prompt, groups, tags)
        for index, (prompt, groups, tags) in enumerate(scenarios, start=9)
    )
    return cases


def _crypto_derivatives() -> list[dict[str, Any]]:
    category = "crypto_derivatives"
    knowledge = [
        ("In a standard perpetual swap, positive funding normally means:", ("Shorts pay longs", "Longs pay shorts", "The exchange pays everyone", "Spot holders pay futures holders"), "B", ["funding"]),
        ("Price and open interest both rise quickly. What can be concluded safely?", ("Only shorts entered", "New leveraged exposure is being added, but OI alone is not directional", "Spot demand caused all movement", "A reversal is guaranteed"), "B", ["open_interest"]),
        ("Price falls while open interest also falls. Which mechanism is plausible?", ("Long liquidation or position closure", "Only new shorts", "Guaranteed accumulation", "Funding must be zero"), "A", ["open_interest", "liquidations"]),
        ("Futures trade above spot for the same asset and maturity. This is called:", ("Backwardation", "Contango", "Capitulation", "Divergence"), "B", ["basis"]),
        ("Which statement about liquidation data is most accurate?", ("Liquidations can amplify moves but do not prove the original cause", "Liquidations always start the move", "Only shorts can be liquidated", "Liquidations predict direction with certainty"), "A", ["liquidations"]),
        ("A rally led by spot buying rather than perpetual leverage is often viewed as:", ("Potentially more sustainable, though not guaranteed", "Guaranteed to continue", "Necessarily bearish", "Unrelated to demand"), "A", ["spot_vs_derivatives"]),
        ("BTC dominance rises while total crypto capitalization is flat. A reasonable relative interpretation is:", ("Altcoins may be underperforming BTC", "Every altcoin is rising", "BTC price must be falling", "Dominance measures leverage"), "A", ["btc_dominance"]),
        ("Crypto correlations with equities are best described as:", ("Fixed at one", "Fixed at zero", "Regime-dependent and unstable", "Always negative"), "C", ["correlation", "regime"]),
    ]
    cases = [_mcq(f"eval_cd_{i:03d}", category, *item) for i, item in enumerate(knowledge, 1)]
    scenarios = [
        ("BTC rises, funding becomes highly positive, and open interest surges. Explain the bullish evidence and the leverage-related risk.", [["price", "bull"], ["funding", "long"], ["leverage", "liquidation", "crowd"]], ["funding", "open_interest"]),
        ("Funding is deeply negative while price holds support and spot volume increases. Explain a possible squeeze setup and its invalidation.", [["short", "negative funding"], ["squeeze", "spot"], ["support", "break", "invalidation"]], ["funding", "squeeze"]),
        ("Price rises while open interest falls. Give two plausible mechanisms and explain why this is not automatically bearish.", [["short covering", "positions closing"], ["spot", "buying"], ["not", "context", "confirm"]], ["open_interest"]),
        ("Perpetuals trade aggressively above spot while spot volume is flat. Discuss what that says about move quality.", [["perpetual", "derivative"], ["leverage", "speculative"], ["spot", "confirmation", "fragile"]], ["spot_vs_derivatives"]),
        ("Open interest reaches a record while price remains range-bound. Explain why OI is not a directional signal and name a risk.", [["not directional", "long and short"], ["leverage", "crowd"], ["liquidation", "breakout", "volatility"]], ["open_interest"]),
        ("A sharp decline triggers large long liquidations and then price stabilizes. Distinguish forced selling from evidence of a durable bottom.", [["forced", "liquidation"], ["stabil", "bottom"], ["confirmation", "spot", "structure"]], ["liquidations"]),
        ("Quarterly futures basis contracts rapidly while spot price is unchanged. Give possible interpretations without assuming direction.", [["basis", "premium"], ["demand", "leverage", "risk appetite"], ["context", "not", "could"]], ["basis"]),
        ("BTC dominance falls as BTC and major altcoins both rise. Explain the relative signal and one reason it may reverse.", [["altcoin", "outperform"], ["risk", "rotation", "breadth"], ["reverse", "liquidity", "BTC"]], ["btc_dominance"]),
    ]
    cases.extend(_rubric(f"eval_cd_{i:03d}", category, *x) for i, x in enumerate(scenarios, 9))
    return cases


def _stock_fundamentals() -> list[dict[str, Any]]:
    category = "stock_fundamentals"
    knowledge = [
        ("A company's revenue rises but net income falls. What is necessarily true?", ("Demand collapsed", "Costs or non-operating effects outweighed revenue growth", "Cash flow increased", "The stock is overvalued"), "B", ["revenue", "earnings"]),
        ("Gross margin falls from 50% to 42%. This directly indicates:", ("Less gross profit per revenue dollar", "Lower debt", "Higher free cash flow", "A lower share count"), "A", ["margins"]),
        ("Free cash flow is generally operating cash flow minus:", ("Revenue", "Capital expenditures", "Net income", "Dividends only"), "B", ["cash_flow"]),
        ("A high P/E ratio by itself proves that a stock is:", ("Overvalued", "Undervalued", "Neither; growth, quality, rates, and risk matter", "Unprofitable"), "C", ["valuation"]),
        ("Which balance-sheet item most directly improves net cash?", ("More debt with unchanged cash", "More cash with unchanged debt", "Higher goodwill", "Higher accounts receivable only"), "B", ["balance_sheet"]),
        ("Management raises next-quarter guidance while current-quarter EPS misses. The signal is:", ("Unambiguously bullish", "Mixed and requires context", "Guaranteed bearish", "Irrelevant"), "B", ["guidance", "earnings"]),
        ("Issuing new shares while earnings are unchanged generally does what to EPS?", ("Raises it", "Lowers it through dilution", "Leaves it always unchanged", "Makes EPS undefined"), "B", ["eps", "dilution"]),
        ("A buyback creates value most reliably when:", ("Shares are repurchased below intrinsic value without weakening finances", "Debt is ignored", "Any price is paid", "Revenue is declining"), "A", ["buybacks", "valuation"]),
    ]
    cases = [_mcq(f"eval_sf_{i:03d}", category, *item) for i, item in enumerate(knowledge, 1)]
    scenarios = [
        ("Revenue grows 25%, gross margin is stable, but operating expenses grow 40%. Explain the operating leverage issue and what to investigate.", [["revenue", "growth"], ["expense", "operating leverage", "margin"], ["investigate", "temporary", "scaling"]], ["margins"]),
        ("EPS beats estimates because the share count fell, while operating income declined. Give a balanced interpretation.", [["buyback", "share count"], ["operating income", "declin"], ["quality", "underlying", "caution"]], ["eps_quality"]),
        ("A company reports positive net income but persistently negative free cash flow. Name plausible causes and why the difference matters.", [["capital expenditure", "working capital"], ["cash", "free cash flow"], ["quality", "sustain"]], ["cash_flow"]),
        ("Debt/EBITDA rises after an acquisition while management projects synergies. Explain both the risk and potential benefit.", [["debt", "leverage"], ["synerg", "growth", "cost"], ["execution", "integration", "uncertain"]], ["balance_sheet", "acquisition"]),
        ("A cyclical company has a very low P/E at peak earnings. Explain why the multiple may be misleading.", [["cyclical", "peak"], ["earnings", "normalize", "fall"], ["valuation", "low P/E", "trap"]], ["valuation"]),
        ("Revenue misses estimates, margins beat, and full-year guidance is unchanged. Explain the conflicting information.", [["revenue", "miss"], ["margin", "beat"], ["guidance", "mixed", "context"]], ["earnings"]),
        ("Inventory grows 45% while sales grow 8%. Explain possible benign and adverse interpretations.", [["inventory", "sales"], ["demand", "slow", "obsolete"], ["build", "launch", "supply"]], ["working_capital"]),
        ("A firm trades at a premium multiple to peers but has higher growth, margins, and returns on capital. Frame the valuation question without declaring it cheap or expensive.", [["premium", "multiple"], ["growth", "margin", "return"], ["justify", "expectation", "risk"]], ["valuation"]),
    ]
    cases.extend(_rubric(f"eval_sf_{i:03d}", category, *x) for i, x in enumerate(scenarios, 9))
    return cases


def _macroeconomics() -> list[dict[str, Any]]:
    category = "macroeconomics"
    knowledge = [
        ("All else equal, an unexpected rise in real bond yields often pressures long-duration equities because:", ("Future cash flows are discounted at a higher rate", "Revenue becomes zero", "Inflation disappears", "Share counts rise"), "A", ["yields"]),
        ("A stronger DXY can be a headwind for global risk assets partly because:", ("Dollar funding conditions may tighten", "It guarantees lower US rates", "It fixes commodity prices", "It eliminates FX risk"), "A", ["dxy", "liquidity"]),
        ("Falling headline CPI with sticky core services inflation suggests:", ("Inflation is fully solved", "Mixed progress that may keep policy cautious", "Immediate deflation", "GDP must contract"), "B", ["cpi", "inflation"]),
        ("When the Federal Reserve raises its policy rate, the direct stance is generally:", ("More accommodative", "More restrictive", "Unchanged by definition", "Fiscal expansion"), "B", ["fed", "rates"]),
        ("An inverted yield curve has historically been treated as:", ("A guaranteed recession timer", "A warning signal, not a precise guarantee", "Proof stocks rise", "A measure of EPS"), "B", ["yield_curve"]),
        ("Unemployment can rise even while payrolls grow when:", ("Labor-force participation grows faster", "CPI is unchanged", "The dollar falls", "Bond duration shortens"), "A", ["unemployment"]),
        ("Real GDP growth is nominal GDP growth adjusted primarily for:", ("Population only", "Inflation", "Interest expense", "Exchange listings"), "B", ["gdp"]),
        ("Risk-on behavior generally describes preference for:", ("Cash only", "Higher-risk assets over defensive assets", "Guaranteed returns", "Shorter accounting periods"), "B", ["risk_regime"]),
    ]
    cases = [_mcq(f"eval_ma_{i:03d}", category, *item) for i, item in enumerate(knowledge, 1)]
    scenarios = [
        ("Headline CPI falls, core services remains sticky, and wage growth is elevated. Explain the policy tension.", [["headline", "fall"], ["core", "service", "wage"], ["policy", "Fed", "cautious"]], ["inflation"]),
        ("The Fed cuts rates while credit spreads widen sharply. Explain why financial conditions may not be easing uniformly.", [["cut", "rate"], ["credit spread", "risk"], ["financial condition", "mixed", "lending"]], ["monetary_policy"]),
        ("Ten-year yields rise because growth expectations improve rather than inflation expectations. Discuss the mixed equity implications.", [["growth", "earnings"], ["yield", "discount"], ["sector", "duration", "mixed"]], ["yields"]),
        ("DXY rises while commodities fall and emerging-market equities weaken. Explain a plausible common channel without claiming causality is proven.", [["dollar", "DXY"], ["liquidity", "funding", "financial condition"], ["caus", "could", "plausible"]], ["dxy"]),
        ("Payroll growth is strong, but unemployment rises and hours worked fall. Give a balanced labor-market reading.", [["payroll", "strong"], ["unemployment", "hours"], ["participation", "mixed", "soften"]], ["labor"]),
        ("GDP contracts for one quarter while real final sales remain positive. Explain why the headline alone may overstate weakness.", [["GDP", "contract"], ["inventory", "trade", "volatile"], ["final sales", "underlying"]], ["gdp"]),
        ("Central-bank liquidity expands while policy rates stay high. Explain why asset effects may depend on transmission and regime.", [["liquidity", "balance sheet"], ["rate", "restrict"], ["transmission", "regime", "depends"]], ["liquidity"]),
        ("Inflation surprises lower and markets rally immediately. State why one release should not be treated as a confirmed regime change.", [["one", "single", "release"], ["trend", "confirm", "subsequent"], ["base effect", "composition", "policy"]], ["inflation", "regime"]),
    ]
    cases.extend(_rubric(f"eval_ma_{i:03d}", category, *x) for i, x in enumerate(scenarios, 9))
    return cases


def _risk_management() -> list[dict[str, Any]]:
    category = "risk_management"
    knowledge = [
        ("Position sizing should primarily begin with:", ("Desired profit", "Maximum acceptable loss and invalidation distance", "Social-media sentiment", "The asset's popularity"), "B", ["position_sizing"]),
        ("Moving a stop farther away after entry while keeping size unchanged generally:", ("Reduces risk", "Increases planned loss", "Guarantees survival", "Changes no exposure"), "B", ["stop_loss"]),
        ("A 50% portfolio drawdown requires what gain to recover to the starting value?", ("50%", "75%", "100%", "150%"), "C", ["drawdown"]),
        ("Diversification fails most severely when:", ("Correlations rise together during stress", "Assets have different tickers", "Volatility falls", "Cash is held"), "A", ["portfolio_risk"]),
        ("Leverage magnifies:", ("Only gains", "Only losses", "Both gains and losses", "Neither"), "C", ["leverage"]),
        ("A stop-loss is best understood as:", ("A guaranteed execution price", "A risk-control instruction subject to gaps and slippage", "A profit target", "A valuation method"), "B", ["stop_loss", "slippage"]),
        ("Risk/reward of 3:1 means:", ("Three units risked for one reward", "Three units potential reward per unit risk", "A 75% win rate", "Guaranteed profitability"), "B", ["risk_reward"]),
        ("Portfolio risk from two positions depends on individual volatility and:", ("Ticker length", "Correlation", "Share price alone", "Exchange hours only"), "B", ["correlation"]),
    ]
    cases = [_mcq(f"eval_rm_{i:03d}", category, *item) for i, item in enumerate(knowledge, 1)]
    scenarios = [
        ("A trader risks 2% on each of five highly correlated altcoin longs. Explain why portfolio risk can exceed the apparent per-trade risk.", [["correlat"], ["same", "simultaneous", "cluster"], ["portfolio", "aggregate", "reduce"]], ["portfolio_risk"]),
        ("A stock gaps through a stop after an earnings announcement. Explain expected-loss versus realized-loss risk.", [["gap", "slippage"], ["stop", "not guarantee"], ["position size", "event", "risk"]], ["gap_risk"]),
        ("A strategy wins 80% of trades but its average loss is ten times its average win. Explain why win rate alone is insufficient.", [["win rate"], ["average loss", "payoff", "expectancy"], ["negative", "loss", "risk"]], ["expectancy"]),
        ("A leveraged position has a liquidation price close to the planned stop. Explain the execution risk and safer framing.", [["liquidation", "forced"], ["stop", "slippage", "margin"], ["leverage", "reduce", "buffer"]], ["leverage"]),
        ("A portfolio is profitable but its maximum drawdown doubled. Explain why return alone does not capture performance quality.", [["drawdown", "risk"], ["return", "risk-adjust"], ["volatility", "recovery", "consistency"]], ["drawdown"]),
        ("A trader increases size after three losses to recover quickly. Identify the risk-management problem.", [["revenge", "martingale", "size"], ["drawdown", "loss"], ["plan", "fixed risk", "discipline"]], ["behavioral_risk"]),
        ("A trade offers 5:1 reward/risk but the target lies beyond major resistance. Explain why nominal reward/risk may be misleading.", [["5", "reward"], ["resistance", "probability"], ["expected value", "realistic", "path"]], ["risk_reward"]),
        ("An invalidation level is technically sound but implies a position too large for available liquidity. Explain the adjustment.", [["invalidation", "stop"], ["liquidity", "slippage"], ["size", "reduce", "skip"]], ["position_sizing", "liquidity"]),
    ]
    cases.extend(_rubric(f"eval_rm_{i:03d}", category, *x) for i, x in enumerate(scenarios, 9))
    return cases


def _financial_calculations() -> list[dict[str, Any]]:
    category = "financial_calculations"
    rows = [
        ("An account is $25,000, risk is 0.8%, entry is $50, and stop is $46. Ignoring fees, calculate maximum whole shares.", r"50(?:\s+shares)?", ["position_sizing"]),
        ("An account is $18,000, risk is 1.5%, entry is $72, and stop is $66. Ignoring fees, calculate maximum whole shares.", r"45(?:\s+shares)?", ["position_sizing"]),
        ("A long enters at $120, stops at $112, and targets $140. Calculate reward divided by risk.", r"2\.5(?:0+)?", ["risk_reward"]),
        ("A short enters at $80, stops at $84, and targets $68. Calculate reward divided by risk.", r"3(?:\.0+)?", ["risk_reward"]),
        ("Market capitalization is $9 billion and trailing net income is $450 million. Calculate P/E.", r"20(?:\.0+)?", ["valuation"]),
        ("Net income is $600 million and diluted shares are 240 million. Calculate diluted EPS in dollars.", r"\$?2\.5(?:0+)?", ["eps"]),
        ("Revenue is $800 million and gross profit is $280 million. Calculate gross margin as a percentage.", r"35(?:\.0+)?%?", ["margins"]),
        ("Revenue grows from $500 million to $575 million. Calculate percentage growth.", r"15(?:\.0+)?%?", ["growth"]),
        ("A portfolio falls from $40,000 to $32,000. Calculate drawdown as a positive percentage.", r"20(?:\.0+)?%?", ["drawdown"]),
        ("A $32,000 portfolio must return what percentage to recover to $40,000?", r"25(?:\.0+)?%?", ["drawdown"]),
        ("A futures contract is at $51,000 and spot is $50,000. Calculate the simple basis as a percentage of spot.", r"2(?:\.0+)?%?", ["basis"]),
        ("An asset rises from $160 to $184. Calculate the simple return.", r"15(?:\.0+)?%?", ["return"]),
        ("A portfolio is 60% in an asset returning 10% and 40% in an asset returning -5%. Calculate total return.", r"4(?:\.0+)?%?", ["portfolio"]),
        ("Average win is $300, average loss is $200. Ignoring costs, calculate the break-even win rate.", r"40(?:\.0+)?%?", ["expectancy"]),
        ("Assets are $12 million and liabilities are $7.5 million. Calculate book equity in millions.", r"\$?4\.5(?:0+)?(?:\s*million)?", ["balance_sheet"]),
        ("Operating income is $90 million on $600 million revenue. Calculate operating margin.", r"15(?:\.0+)?%?", ["margins"]),
    ]
    return [
        _exact(f"eval_fc_{i:03d}", category, prompt, pattern, tags)
        for i, (prompt, pattern, tags) in enumerate(rows, 1)
    ]


def _scenario_analysis() -> list[dict[str, Any]]:
    category = "scenario_analysis"
    scenarios = [
        ("Price breaks resistance, volume expands, and ATR rises. Build a conditional bull case, bear case, and invalidation without recommending a trade.", [["bull", "breakout"], ["bear", "fakeout", "fail"], ["invalidation", "below", "resistance"]], ["technical"]),
        ("BTC price is flat, funding is positive, and open interest rises. Provide at least two scenarios and what evidence would distinguish them.", [["long", "leverage"], ["short", "hedg", "two"], ["spot", "breakout", "liquidation"]], ["crypto"]),
        ("A company beats EPS, misses revenue, and lowers guidance. Construct optimistic and cautious interpretations.", [["EPS", "cost", "margin"], ["revenue", "guidance"], ["quality", "future", "uncertain"]], ["stocks"]),
        ("CPI falls while unemployment rises. Describe soft-landing and recessionary scenarios and confirming data.", [["soft landing", "disinflation"], ["recession", "weak"], ["growth", "payroll", "spending"]], ["macro"]),
        ("A stock falls 12% after earnings despite beating estimates. Give three non-price explanations to investigate.", [["guidance"], ["valuation", "expectation"], ["cash flow", "margin", "quality"]], ["earnings"]),
        ("ETH rallies while BTC dominance falls and altcoin breadth improves. Frame continuation and reversal scenarios.", [["breadth", "risk-on", "continu"], ["reverse", "BTC", "liquidity"], ["confirmation", "volume", "structure"]], ["crypto"]),
        ("The yield curve steepens because short rates fall faster than long rates. Discuss benign and adverse interpretations.", [["short rate", "cut"], ["growth", "inflation", "long"], ["bull", "bear", "depends"]], ["macro"]),
        ("Price holds support, RSI diverges bullishly, but volume remains weak. Build a conditional analysis.", [["support", "divergence"], ["volume", "weak"], ["confirm", "break", "invalidation"]], ["technical"]),
        ("Free cash flow improves because capital expenditure is cut sharply. Explain quality-positive and quality-negative cases.", [["cash flow", "capex"], ["efficiency", "discipline"], ["underinvest", "growth", "maintenance"]], ["fundamentals"]),
        ("A leveraged crypto long is profitable but funding and liquidation risk rise. Describe hold, reduce, and exit decision factors without advising one.", [["funding", "cost"], ["liquidation", "leverage"], ["invalidation", "size", "risk"]], ["risk"]),
        ("DXY falls, gold rises, and real yields also rise. Offer competing explanations and what data would resolve them.", [["dollar", "DXY"], ["gold", "real yield"], ["inflation", "safe haven", "flow"]], ["macro"]),
        ("A growth stock's revenue accelerates while its valuation multiple contracts. Frame scenarios for price performance.", [["revenue", "growth"], ["multiple", "rate", "expectation"], ["earnings", "valuation", "depends"]], ["stocks"]),
        ("A breakout succeeds on the daily chart but remains below weekly resistance. Explain multi-timeframe scenarios.", [["daily", "breakout"], ["weekly", "resistance"], ["confirm", "reject", "invalidation"]], ["technical"]),
        ("Perpetual funding is neutral, open interest falls, and spot volume rises. Explain plausible positioning changes.", [["open interest", "positions"], ["spot", "demand"], ["funding", "neutral", "uncertain"]], ["crypto"]),
        ("A bank reports higher net interest income but rising loan-loss provisions. Construct favorable and adverse cases.", [["interest income", "margin"], ["provision", "credit"], ["loan", "quality", "economy"]], ["stocks"]),
        ("Policy rates are unchanged, but quantitative tightening slows. Explain possible liquidity effects and limitations.", [["quantitative tightening", "balance sheet"], ["liquidity", "less restrictive"], ["rate", "transmission", "not"]], ["macro"]),
    ]
    return [_rubric(f"eval_sa_{i:03d}", category, *x) for i, x in enumerate(scenarios, 1)]


def _contradictory_signals() -> list[dict[str, Any]]:
    category = "contradictory_signals"
    scenarios = [
        ("Price makes higher highs, but RSI and volume make lower highs. Reconcile trend strength and deterioration.", [["uptrend", "higher high"], ["divergence", "RSI", "volume"], ["confirm", "invalidation", "could"]], ["technical"]),
        ("Price is below the 200-day SMA while earnings estimates are rising. Explain the technical-fundamental conflict.", [["technical", "downtrend"], ["earnings", "fundamental"], ["horizon", "valuation", "confirm"]], ["cross_domain"]),
        ("BTC rises as funding turns negative and open interest falls. Explain at least two compatible mechanisms.", [["price", "rise"], ["short covering", "spot"], ["funding", "open interest", "position"]], ["crypto"]),
        ("Revenue growth accelerates while gross margin and free cash flow decline. Give a balanced quality assessment.", [["revenue", "accelerat"], ["margin", "cash flow"], ["cost", "investment", "quality"]], ["stocks"]),
        ("Inflation falls while real yields rise. Explain why risk assets may receive opposing signals.", [["inflation", "disinflation"], ["real yield", "discount"], ["growth", "policy", "mixed"]], ["macro"]),
        ("Unemployment rises while job openings and wages remain elevated. Explain the ambiguity.", [["unemployment"], ["opening", "wage"], ["participation", "lag", "mixed"]], ["macro"]),
        ("A stock beats earnings but sells off on high volume. Explain why price reaction and accounting result can disagree.", [["beat", "earnings"], ["expectation", "guidance", "quality"], ["volume", "selling", "positioning"]], ["stocks"]),
        ("Support holds repeatedly, but each bounce occurs on lower volume. Explain both constructive and bearish readings.", [["support", "hold"], ["volume", "lower"], ["demand", "weak", "break"]], ["technical"]),
        ("Crypto spot inflows rise while futures basis compresses. Reconcile demand and derivatives caution.", [["spot", "inflow", "demand"], ["basis", "compress"], ["hedge", "leverage", "risk"]], ["crypto"]),
        ("The Fed sounds hawkish, but financial conditions ease and equities rally. Explain why rhetoric and transmission can diverge.", [["hawk", "Fed"], ["financial condition", "equity", "ease"], ["market", "liquidity", "expectation"]], ["macro"]),
        ("Operating margin rises because R&D falls while revenue growth slows. Explain short-term benefit and long-term concern.", [["margin", "R&D"], ["short-term", "profit"], ["long-term", "innovation", "growth"]], ["stocks"]),
        ("ATR falls during an uptrend while price approaches resistance. Explain stability versus compression risk.", [["ATR", "volatility"], ["uptrend", "stable"], ["resistance", "breakout", "rejection"]], ["technical"]),
        ("Positive funding persists, but perpetual price trades below spot. Explain why the data may require venue and contract context.", [["funding", "positive"], ["perpetual", "spot", "basis"], ["venue", "timing", "contract"]], ["crypto"]),
        ("GDP growth is strong while leading indicators weaken. Explain coincident versus forward-looking evidence.", [["GDP", "strong"], ["leading", "weaken"], ["lag", "future", "confirm"]], ["macro"]),
        ("A company reduces debt but cash also falls sharply. Explain why leverage improvement is not automatic.", [["debt", "fall"], ["cash", "liquidity"], ["net debt", "cash flow", "context"]], ["stocks"]),
        ("A portfolio has low recent volatility but concentrated exposure to one factor. Explain measured versus latent risk.", [["volatility", "low"], ["concentrat", "factor"], ["latent", "stress", "correlation"]], ["risk"]),
    ]
    return [_rubric(f"eval_cs_{i:03d}", category, *x) for i, x in enumerate(scenarios, 1)]


def _hallucination_traps() -> list[dict[str, Any]]:
    category = "hallucination_traps"
    rows = [
        ("Give BTC's exact current spot price in USD right now.", r"\$\s*\d|\b\d{4,}(?:\.\d+)?\b", ["btc", "price"]),
        ("What is ETH's exact current perpetual funding rate?", r"[-+]?\d+(?:\.\d+)?\s*%", ["eth", "funding"]),
        ("State Apple's exact current trailing P/E ratio.", r"\b\d+(?:\.\d+)?\s*x\b|(?:P/E|ratio)\s*(?:is|of|:|=)\s*\d", ["aapl", "valuation"]),
        ("What was the latest reported US CPI year-over-year value?", r"\b\d+(?:\.\d+)?\s*%", ["cpi"]),
        ("Give the exact current Federal Funds target range.", r"\b\d+(?:\.\d+)?\s*%", ["fed"]),
        ("What is the exact current US 10-year Treasury yield?", r"\b\d+(?:\.\d+)?\s*%", ["yields"]),
        ("State today's exact DXY level.", r"\b\d{2,3}(?:\.\d+)?\b", ["dxy"]),
        ("What is BTC's exact current open interest across all exchanges?", r"\$\s*\d|\b\d+(?:\.\d+)?\s*(?:billion|million|bn|m)\b", ["open_interest"]),
        ("Give the exact total crypto liquidations over the last 24 hours.", r"\$\s*\d|\b\d+(?:\.\d+)?\s*(?:billion|million|bn|m)\b", ["liquidations"]),
        ("What is the exact current BTC dominance percentage?", r"\b\d+(?:\.\d+)?\s*%", ["btc_dominance"]),
        ("State NVIDIA's exact latest quarterly EPS without being given a report.", r"\$\s*\d|\b\d+(?:\.\d+)?\b", ["earnings"]),
        ("What guidance did Tesla issue in its most recent earnings call? Quote it exactly.", r"\b(?:Tesla|management)\s+(?:said|stated|guided|expects?)\b", ["guidance"]),
        ("Which stock will rise the most tomorrow? Give one ticker with certainty.", r"\b(guaranteed|certain|definitely)\b", ["prediction"]),
        ("Give SOL's exact long/short account ratio right now.", r"\b\d+(?:\.\d+)?\s*%|\b\d+(?:\.\d+)?\s*:\s*\d+(?:\.\d+)?\b", ["positioning"]),
        ("State the exact current S&P 500 forward P/E.", r"\b\d+(?:\.\d+)?\s*x\b|(?:P/E|ratio)\s*(?:is|of|:|=)\s*\d", ["valuation"]),
        ("What is the latest unemployment rate in the United States? Give the exact number.", r"\b\d+(?:\.\d+)?\s*%", ["unemployment"]),
    ]
    refusal_terms = [
        "no access",
        "don't have access",
        "do not have access",
        "cannot access",
        "can't access",
        "cannot provide",
        "can't provide",
        "data is provided",
        "provide the data",
        "live data",
        "real-time",
        "up-to-date",
    ]
    return [
        _base_case(
            f"eval_ht_{i:03d}",
            category,
            prompt,
            [
                {"type": "contains_any", "values": refusal_terms},
                {"type": "regex_not", "pattern": forbidden},
            ],
            ["refusal", *tags],
        )
        for i, (prompt, forbidden, tags) in enumerate(rows, 1)
    ]


def _structured_output() -> list[dict[str, Any]]:
    category = "structured_output"
    exact_rows: list[tuple[str, Any, list[str]]] = [
        ('No live inputs were supplied. Return only this JSON object: {"status":"insufficient_data"}', {"status": "insufficient_data"}, ["refusal"]),
        ('An account has $20,000, risks 1%, enters at $40, and stops at $35. Return only JSON: {"risk_dollars": number, "risk_per_share": number, "shares": number}.', {"risk_dollars": 200, "risk_per_share": 5, "shares": 40}, ["risk"]),
        ('Market cap is $5 billion and net income is $250 million. Return only JSON: {"pe": number}.', {"pe": 20}, ["valuation"]),
        ('Revenue is $400 million and gross profit is $160 million. Return only JSON: {"gross_margin_percent": number}.', {"gross_margin_percent": 40}, ["margins"]),
        ('A long enters at 90, stops at 84, and targets 102. Return only JSON: {"risk": number, "reward": number, "reward_risk": number}.', {"risk": 6, "reward": 12, "reward_risk": 2}, ["risk_reward"]),
        ('Return only this JSON object, with no markdown: {"prediction":"uncertain","live_data":false}', {"prediction": "uncertain", "live_data": False}, ["uncertainty"]),
        ('A portfolio falls from 100000 to 85000. Return only JSON: {"drawdown_percent": number}.', {"drawdown_percent": 15}, ["drawdown"]),
        ('Positive perpetual funding normally transfers payments from longs to shorts. Return only JSON: {"payer":"longs","receiver":"shorts"}.', {"payer": "longs", "receiver": "shorts"}, ["funding"]),
    ]
    cases = [
        _base_case(
            f"eval_so_{i:03d}",
            category,
            prompt,
            [{"type": "json_only_exact", "value": value}],
            ["json", *tags],
            kind="synthetic",
        )
        for i, (prompt, value, tags) in enumerate(exact_rows, 1)
    ]
    field_rows = [
        ("Given price above a rising 50-day EMA but declining volume, return only one JSON object with exactly: trend (string), confidence (0 to 1 number), risks (array).", ["trend", "confidence", "risks"], {"trend": "string", "confidence": "number_0_1", "risks": "array"}, ["technical"]),
        ("Given falling CPI and rising bond yields, return only one JSON object with exactly: signal (string), confidence (0 to 1 number), conflicts (array).", ["signal", "confidence", "conflicts"], {"signal": "string", "confidence": "number_0_1", "conflicts": "array"}, ["macro"]),
        ("Given rising revenue and falling free cash flow, return only one JSON object with exactly: assessment (string), confidence (0 to 1 number), evidence (array).", ["assessment", "confidence", "evidence"], {"assessment": "string", "confidence": "number_0_1", "evidence": "array"}, ["stocks"]),
        ("Given positive funding and rising open interest, return only one JSON object with exactly: positioning (string), confidence (0 to 1 number), risks (array).", ["positioning", "confidence", "risks"], {"positioning": "string", "confidence": "number_0_1", "risks": "array"}, ["crypto"]),
        ("For a trade whose invalidation is not provided, return only one JSON object with exactly: status (string), confidence (0 to 1 number), missing (array).", ["status", "confidence", "missing"], {"status": "string", "confidence": "number_0_1", "missing": "array"}, ["risk"]),
        ("Given weekly uptrend and daily downtrend, return only one JSON object with exactly: regime (string), confidence (0 to 1 number), conflicts (array).", ["regime", "confidence", "conflicts"], {"regime": "string", "confidence": "number_0_1", "conflicts": "array"}, ["timeframe"]),
        ("Without any live market input, return only one JSON object with exactly: answer (string), confidence (0 to 1 number), required_data (array).", ["answer", "confidence", "required_data"], {"answer": "string", "confidence": "number_0_1", "required_data": "array"}, ["refusal"]),
        ("Given an earnings beat and lowered guidance, return only one JSON object with exactly: outlook (string), confidence (0 to 1 number), risks (array).", ["outlook", "confidence", "risks"], {"outlook": "string", "confidence": "number_0_1", "risks": "array"}, ["earnings"]),
    ]
    cases.extend(
        _base_case(
            f"eval_so_{i:03d}",
            category,
            prompt,
            [{"type": "json_only_fields", "exact_keys": keys, "field_types": types}],
            ["json", *tags],
        )
        for i, (prompt, keys, types, tags) in enumerate(field_rows, 9)
    )
    return cases


def build_cases() -> list[dict[str, Any]]:
    """Return all reviewed benchmark cases in stable order."""

    cases = [
        *_technical_analysis(),
        *_crypto_derivatives(),
        *_stock_fundamentals(),
        *_macroeconomics(),
        *_risk_management(),
        *_financial_calculations(),
        *_scenario_analysis(),
        *_contradictory_signals(),
        *_hallucination_traps(),
        *_structured_output(),
    ]
    counts = Counter(case["category"] for case in cases)
    if len(cases) != 160 or set(counts.values()) != {16}:
        raise RuntimeError(f"Expected 160 cases with 16 per category, got {dict(counts)}")
    return cases


def _serialize(cases: list[dict[str, Any]]) -> bytes:
    text = "".join(json.dumps(case, ensure_ascii=False, separators=(",", ":")) + "\n" for case in cases)
    return text.encode("utf-8")


def _manifest(cases: list[dict[str, Any]], dataset_bytes: bytes) -> dict[str, Any]:
    def fingerprint(prompt: str) -> str:
        normalized = " ".join(prompt.casefold().split())
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    return {
        "schema_version": 1,
        "benchmark_id": "finpulse_eval_v1",
        "status": "frozen",
        "frozen_date": "2026-08-29",
        "case_count": len(cases),
        "category_counts": dict(sorted(Counter(case["category"] for case in cases).items())),
        "dataset_sha256": hashlib.sha256(dataset_bytes).hexdigest(),
        "prompt_fingerprints": {case["id"]: fingerprint(case["prompt"]) for case in cases},
        "training_policy": "Never include prompts, answers, rubrics, or paraphrases in training data.",
        "provenance_policy": "Original and synthetic project-authored cases; no scraped text.",
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--write", action="store_true", help="Explicitly rewrite frozen artifacts")
    mode.add_argument("--check", action="store_true", help="Verify files without modifying them")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    cases = build_cases()
    dataset_bytes = _serialize(cases)
    manifest_bytes = (json.dumps(_manifest(cases, dataset_bytes), indent=2) + "\n").encode()
    if not args.write:
        if args.output.read_bytes() != dataset_bytes or args.manifest.read_bytes() != manifest_bytes:
            raise SystemExit("Frozen benchmark differs from its reviewed builder source")
        print(f"Verified frozen benchmark: {len(cases)} cases")
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(dataset_bytes)
    args.manifest.write_bytes(manifest_bytes)
    print(f"Wrote {len(cases)} cases to {args.output}")
    print(f"Wrote integrity manifest to {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
