# ruff: noqa: E501
"""Build the deterministic, project-original Stage 5B corrective corpus."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

from finpulse_llm.data.config import load_data_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/data/training_pipeline_stage5b.toml"
DEFAULT_OUTPUT = PROJECT_ROOT / "data/raw/finpulse_stage5b_v1.jsonl"

CATEGORY_COUNTS = {
    "technical_analysis": 100,
    "crypto_derivatives": 75,
    "stock_fundamentals": 75,
    "macroeconomics": 60,
    "risk_management": 75,
    "scenario_analysis": 65,
    "terminology_misc": 50,
}
CATEGORY_PREFIX = {
    "technical_analysis": "ta",
    "crypto_derivatives": "cd",
    "stock_fundamentals": "sf",
    "macroeconomics": "ma",
    "risk_management": "rm",
    "scenario_analysis": "sa",
    "terminology_misc": "tm",
}


@dataclass(frozen=True)
class TaskSpec:
    task_type: str
    response_format: str
    response_length: str


def _task_specs() -> list[TaskSpec]:
    return (
        [TaskSpec("calculation", "final_marker", "short") for _ in range(75)]
        + [TaskSpec("calculation", "final_marker", "medium") for _ in range(75)]
        + [TaskSpec("multiple_choice", "final_marker", "short") for _ in range(100)]
        + [TaskSpec("factual", "json_only", "short") for _ in range(25)]
        + [TaskSpec("factual", "plain", "short") for _ in range(50)]
        + [TaskSpec("analysis", "plain", "medium") for _ in range(50)]
        + [TaskSpec("analysis", "plain", "long") for _ in range(50)]
        + [TaskSpec("refusal", "plain", "medium") for _ in range(25)]
        + [TaskSpec("instruction_following", "json_only", "medium") for _ in range(50)]
    )


def _number(value: float, decimals: int = 2) -> str:
    rendered = f"{value:.{decimals}f}"
    return rendered.rstrip("0").rstrip(".")


def _calculation(category: str, n: int) -> tuple[str, str, tuple[str, ...], str]:
    family = n % 5
    if category == "technical_analysis":
        if family == 0:
            low = 80 + n % 31
            high = low + 10 + n % 30
            result = (high - low) / low * 100
            return (f"A chart moves from a swing low of {low} to a swing high of {high}. Calculate the advance as a percentage of the low. Show the formula and end with FINAL: <percentage>.", f"Advance = ({high} - {low}) / {low} × 100 = {_number(result)}%.\nFINAL: {_number(result)}%", ("percent_change", "price_action"), "beginner")
        if family == 1:
            high, low = 140 + n % 35, 92 + n % 21
            midpoint = (high + low) / 2
            return (f"For a range with high {high} and low {low}, compute its midpoint. End with FINAL: <value>.", f"Midpoint = ({high} + {low}) / 2 = {_number(midpoint)}.\nFINAL: {_number(midpoint)}", ("range", "midpoint"), "beginner")
        if family == 2:
            values = [40 + n % 12, 44 + (n * 2) % 13, 48 + (n * 3) % 14]
            result = sum(values) / 3
            return (f"The last three closing prices are {values[0]}, {values[1]}, and {values[2]}. Calculate the three-period simple moving average. End with FINAL: <value>.", f"SMA = ({' + '.join(map(str, values))}) / 3 = {_number(result)}.\nFINAL: {_number(result)}", ("moving_average", "calculation"), "beginner")
        if family == 3:
            entry, atr, multiple = 120 + n % 27, 2 + n % 5, 2
            stop = entry - atr * multiple
            return (f"A long entry is {entry}, ATR is {atr}, and the plan places the stop two ATR below entry. Calculate the stop level. End with FINAL: <value>.", f"Stop = {entry} - (2 × {atr}) = {stop}. ATR sizes the distance; it does not predict direction.\nFINAL: {stop}", ("atr", "stop_distance"), "intermediate")
        entry = 100 + n % 20
        stop = entry - (4 + n % 6)
        target = entry + 8 + n % 15
        risk, reward = entry - stop, target - entry
        ratio = reward / risk
        return (f"A planned long has entry {entry}, stop {stop}, and target {target}. Calculate reward-to-risk. End with FINAL: <ratio>x.", f"Risk = {entry} - {stop} = {risk}; reward = {target} - {entry} = {reward}; reward-to-risk = {reward} / {risk} = {_number(ratio)}x.\nFINAL: {_number(ratio)}x", ("reward_risk", "trade_plan"), "intermediate")

    if category == "crypto_derivatives":
        if family == 0:
            notional, rate = 8000 + 250 * (n % 40), 0.01 + 0.005 * (n % 7)
            payment = notional * rate / 100
            return (f"A perpetual long has ${notional:,} notional and pays a {rate:.3f}% funding rate for one interval. Calculate the funding payment, ignoring fees. End with FINAL: $<amount>.", f"Funding payment = ${notional:,} × {rate:.3f}% = ${_number(payment)}. A positive rate means the long pays.\nFINAL: ${_number(payment)}", ("funding", "notional"), "intermediate")
        if family == 1:
            quantity, entry, exit_price = 2 + n % 4, 1800 + 20 * (n % 20), 1840 + 25 * (n % 23)
            pnl = quantity * (exit_price - entry)
            return (f"A trader is long {quantity} units from ${entry:,} and exits at ${exit_price:,}. Compute P&L before costs. End with FINAL: $<amount>.", f"Long P&L = {quantity} × (${exit_price:,} - ${entry:,}) = ${pnl:,}.\nFINAL: ${pnl:,}", ("pnl", "perpetuals"), "beginner")
        if family == 2:
            spot, future = 100 + n % 50, 102 + n % 55
            basis = (future - spot) / spot * 100
            return (f"Spot is {spot} and the futures contract trades at {future}. Calculate the simple futures basis as a percentage of spot. End with FINAL: <percentage>.", f"Basis = ({future} - {spot}) / {spot} × 100 = {_number(basis)}%.\nFINAL: {_number(basis)}%", ("basis", "futures"), "intermediate")
        if family == 3:
            notional, leverage = 12000 + 500 * (n % 25), 3 + n % 8
            margin = notional / leverage
            return (f"A ${notional:,} crypto position uses {leverage}x leverage. Calculate simplified initial margin as notional divided by leverage. End with FINAL: $<amount>.", f"Initial margin = ${notional:,} / {leverage} = ${_number(margin)}. This ignores exchange maintenance-margin rules.\nFINAL: ${_number(margin)}", ("leverage", "margin"), "beginner")
        entry, liquidation = 200 + n % 60, 150 + n % 45
        buffer = (entry - liquidation) / entry * 100
        return (f"A long enters at {entry} and has an estimated liquidation level at {liquidation}. Calculate the downside buffer as a percentage of entry. End with FINAL: <percentage>.", f"Buffer = ({entry} - {liquidation}) / {entry} × 100 = {_number(buffer)}%. Actual liquidation depends on venue rules and fees.\nFINAL: {_number(buffer)}%", ("liquidation", "leverage"), "intermediate")

    if category == "stock_fundamentals":
        if family == 0:
            income, shares = 60 + 5 * (n % 25), 20 + n % 15
            eps = income / shares
            return (f"A company earns ${income} million and has {shares} million diluted shares. Calculate diluted EPS. End with FINAL: $<value>.", f"EPS = ${income} million / {shares} million shares = ${_number(eps)}.\nFINAL: ${_number(eps)}", ("eps", "share_count"), "beginner")
        if family == 1:
            price, eps = 30 + n % 60, 2 + (n % 12) / 2
            pe = price / eps
            return (f"A stock trades at ${price} with trailing EPS of ${_number(eps)}. Calculate its P/E ratio. End with FINAL: <multiple>x.", f"P/E = ${price} / ${_number(eps)} = {_number(pe)}x.\nFINAL: {_number(pe)}x", ("valuation", "pe_ratio"), "beginner")
        if family == 2:
            fcf, cap = 120 + 10 * (n % 30), 2400 + 100 * (n % 30)
            result = fcf / cap * 100
            return (f"Free cash flow is ${fcf} million and market capitalization is ${cap} million. Calculate free-cash-flow yield. End with FINAL: <percentage>.", f"FCF yield = ${fcf}m / ${cap}m × 100 = {_number(result)}%.\nFINAL: {_number(result)}%", ("free_cash_flow", "valuation"), "intermediate")
        if family == 3:
            revenue, cost = 500 + 20 * (n % 25), 280 + 15 * (n % 20)
            margin = (revenue - cost) / revenue * 100
            return (f"Revenue is ${revenue} million and cost of revenue is ${cost} million. Calculate gross margin. End with FINAL: <percentage>.", f"Gross profit = ${revenue - cost}m; gross margin = ${revenue - cost}m / ${revenue}m × 100 = {_number(margin)}%.\nFINAL: {_number(margin)}%", ("gross_margin", "income_statement"), "beginner")
        shares, price = 40 + n % 35, 25 + n % 70
        cap = shares * price
        return (f"A company has {shares} million shares outstanding at ${price} per share. Calculate market capitalization in millions. End with FINAL: $<amount> million.", f"Market cap = {shares} million × ${price} = ${cap} million.\nFINAL: ${cap} million", ("market_cap", "share_count"), "beginner")

    if category == "macroeconomics":
        if family == 0:
            nominal, inflation = 4 + (n % 10) / 2, 2 + (n % 6) / 2
            real = nominal - inflation
            return (f"A nominal policy rate is {_number(nominal)}% and inflation is {_number(inflation)}%. Using the simple subtraction approximation, calculate the real policy rate. End with FINAL: <percentage>.", f"Approximate real rate = {_number(nominal)}% - {_number(inflation)}% = {_number(real)}%.\nFINAL: {_number(real)}%", ("real_rates", "inflation"), "beginner")
        if family == 1:
            before, after = 3.25 + (n % 5) / 4, 3.75 + (n % 7) / 4
            bps = round((after - before) * 100)
            return (f"A yield moves from {before:.2f}% to {after:.2f}%. Calculate the change in basis points. End with FINAL: <number> bps.", f"Change = ({after:.2f}% - {before:.2f}%) × 100 = {bps} basis points.\nFINAL: {bps} bps", ("basis_points", "bond_yields"), "beginner")
        if family == 2:
            nominal, inflation = 6 + n % 5, 2 + n % 4
            real = nominal - inflation
            return (f"Nominal GDP grows {nominal}% while the GDP deflator rises {inflation}%. Approximate real GDP growth using subtraction. End with FINAL: <percentage>.", f"Approximate real growth = {nominal}% - {inflation}% = {real}%. This is a simplified approximation.\nFINAL: {real}%", ("gdp", "inflation"), "intermediate")
        if family == 3:
            old, new = 80 + n % 20, 84 + n % 25
            change = (new - old) / old * 100
            return (f"A trade-weighted currency index changes from {old} to {new}. Calculate the signed percentage change. End with FINAL: <percentage>.", f"Change = ({new} - {old}) / {old} × 100 = {_number(change)}%.\nFINAL: {_number(change)}%", ("currency", "percent_change"), "beginner")
        old, new = 4 + (n % 5) / 2, 5 + (n % 7) / 2
        change = new - old
        return (f"The unemployment rate changes from {_number(old)}% to {_number(new)}%. Calculate the change in percentage points. End with FINAL: <value> percentage points.", f"Percentage-point change = {_number(new)}% - {_number(old)}% = {_number(change)} percentage points.\nFINAL: {_number(change)} percentage points", ("unemployment", "percentage_points"), "beginner")

    if category == "risk_management":
        if family == 0:
            account, risk_pct, entry = 20000 + 1000 * (n % 30), 0.5, 60 + n % 30
            stop = entry - (3 + n % 7)
            per_unit = entry - stop
            units = int(account * risk_pct / 100 / per_unit)
            return (f"An account is ${account:,}, risk per trade is {risk_pct}%, entry is ${entry}, and stop is ${stop}. Calculate the maximum whole-share position before costs. End with FINAL: <number> shares.", f"Risk budget = ${account:,} × {risk_pct}% = ${_number(account * risk_pct / 100)}. Risk per share = ${per_unit}. Size = floor(${_number(account * risk_pct / 100)} / ${per_unit}) = {units}.\nFINAL: {units} shares", ("position_sizing", "stop_loss"), "intermediate")
        if family == 1:
            loss = 10 + n % 41
            recovery = loss / (100 - loss) * 100
            return (f"A portfolio loses {loss}% from its peak. Calculate the percentage gain required on remaining capital to recover. End with FINAL: <percentage>.", f"Recovery gain = {loss} / (100 - {loss}) × 100 = {_number(recovery)}%.\nFINAL: {_number(recovery)}%", ("drawdown", "recovery"), "intermediate")
        if family == 2:
            positions, each = 3 + n % 6, 100 + 25 * (n % 12)
            total = positions * each
            return (f"A portfolio has {positions} positions, each capped at ${each} planned loss. Assuming all stops are hit, calculate nominal aggregate planned loss. End with FINAL: $<amount>.", f"Aggregate planned loss = {positions} × ${each} = ${total}. Correlation and gaps can make realized loss differ.\nFINAL: ${total}", ("portfolio_risk", "exposure"), "beginner")
        if family == 3:
            entry, shares = 50 + n % 30, 20 + n % 40
            stop = entry - (2 + n % 8)
            loss = (entry - stop) * shares
            return (f"A long position has {shares} shares, entry ${entry}, and stop ${stop}. Calculate planned dollar loss if filled at the stop. End with FINAL: $<amount>.", f"Planned loss = ({entry} - {stop}) × {shares} = ${loss}. Slippage and gaps are excluded.\nFINAL: ${loss}", ("stop_loss", "position_risk"), "beginner")
        winners, win, losers, loss = 4 + n % 4, 150 + 10 * (n % 12), 3 + n % 3, 100 + 5 * (n % 10)
        net = winners * win - losers * loss
        return (f"A strategy records {winners} winners averaging ${win} and {losers} losers averaging ${loss}. Calculate net P&L before costs. End with FINAL: $<amount>.", f"Net P&L = ({winners} × ${win}) - ({losers} × ${loss}) = ${net}.\nFINAL: ${net}", ("expectancy", "pnl"), "beginner")

    if category == "scenario_analysis":
        if family in {0, 1}:
            up_prob, up_return, down_return = 45 + n % 16, 10 + n % 11, -(5 + n % 9)
            expected = up_prob / 100 * up_return + (100 - up_prob) / 100 * down_return
            return (f"A two-outcome scenario assigns {up_prob}% probability to a {up_return}% return and {100-up_prob}% probability to a {down_return}% return. Calculate expected return. End with FINAL: <percentage>.", f"Expected return = ({up_prob/100:.2f} × {up_return}%) + ({(100-up_prob)/100:.2f} × {down_return}%) = {_number(expected)}%. This is a scenario average, not a forecast guarantee.\nFINAL: {_number(expected)}%", ("expected_value", "probability"), "intermediate")
        if family == 2:
            bull, base, bear = 130 + n % 40, 100 + n % 25, 70 + n % 20
            value = 0.25 * bull + 0.5 * base + 0.25 * bear
            return (f"A valuation exercise assigns 25% to bull value {bull}, 50% to base value {base}, and 25% to bear value {bear}. Calculate the probability-weighted value. End with FINAL: <value>.", f"Weighted value = 0.25 × {bull} + 0.50 × {base} + 0.25 × {bear} = {_number(value)}.\nFINAL: {_number(value)}", ("valuation", "probability"), "intermediate")
        if family == 3:
            upside, downside = 12 + n % 15, 6 + n % 10
            ratio = upside / downside
            return (f"A scenario has {upside}% estimated upside and {downside}% estimated downside. Calculate the upside-to-downside ratio. End with FINAL: <ratio>x.", f"Upside-to-downside = {upside} / {downside} = {_number(ratio)}x. Probabilities still need separate assessment.\nFINAL: {_number(ratio)}x", ("asymmetry", "scenario_analysis"), "beginner")
        revenue, margin = 400 + 20 * (n % 20), 12 + n % 10
        profit = revenue * margin / 100
        return (f"In a base scenario, revenue is ${revenue} million and operating margin is {margin}%. Calculate operating profit. End with FINAL: $<amount> million.", f"Operating profit = ${revenue}m × {margin}% = ${_number(profit)}m.\nFINAL: ${_number(profit)} million", ("operating_margin", "scenario_analysis"), "beginner")

    if family == 0:
        percent = 1 + (n % 20) / 4
        bps = round(percent * 100)
        return (f"Convert {_number(percent)} percentage points into basis points. End with FINAL: <number> bps.", f"One percentage point equals 100 basis points, so {_number(percent)} × 100 = {bps} bps.\nFINAL: {bps} bps", ("basis_points", "terminology"), "beginner")
    if family == 1:
        old, new = 100 + n % 20, 110 + n % 35
        result = (new - old) / old * 100
        return (f"A value changes from {old} to {new}. Calculate percentage change, not percentage-point change. End with FINAL: <percentage>.", f"Percentage change = ({new} - {old}) / {old} × 100 = {_number(result)}%.\nFINAL: {_number(result)}%", ("percent_change", "terminology"), "beginner")
    if family == 2:
        assets, liabilities = 500 + 25 * (n % 20), 300 + 20 * (n % 15)
        equity = assets - liabilities
        return (f"Assets are ${assets} and liabilities are ${liabilities}. Use the accounting identity to calculate equity. End with FINAL: $<amount>.", f"Equity = assets - liabilities = ${assets} - ${liabilities} = ${equity}.\nFINAL: ${equity}", ("accounting_identity", "terminology"), "beginner")
    if family == 3:
        notional, move = 10000 + 500 * (n % 20), 1 + n % 7
        pnl = notional * move / 100
        return (f"An unlevered ${notional:,} exposure gains {move}%. Calculate the dollar gain before costs. End with FINAL: $<amount>.", f"Gain = ${notional:,} × {move}% = ${_number(pnl)}.\nFINAL: ${_number(pnl)}", ("notional", "percent_change"), "beginner")
    start, end = 200 + n % 40, 230 + n % 60
    change = end - start
    return (f"An index moves from {start} to {end}. Calculate the absolute point change. End with FINAL: <number> points.", f"Point change = {end} - {start} = {change}.\nFINAL: {change} points", ("point_change", "terminology"), "beginner")


MCQ_BANK = {
    "technical_analysis": [
        ("Price closes above resistance on expanding volume, then holds that level on a retest. Which interpretation is best?", ["The breakout has confirmation but remains fallible", "Volume guarantees a rally", "The retest is irrelevant", "Resistance can never matter again"], "A"),
        ("Price makes a lower low while RSI makes a higher low. What does this establish?", ["A guaranteed reversal", "Possible momentum divergence requiring confirmation", "A confirmed new uptrend", "No information at all"], "B"),
        ("ATR rises sharply while trend direction is unchanged. What did ATR primarily indicate?", ["Higher volatility", "Guaranteed bullish direction", "Fair value", "Earnings growth"], "A"),
        ("A daily uptrend contains a four-hour pullback. Which statement is most accurate?", ["The timeframes can describe different horizons", "One timeframe must be false", "The daily trend guarantees the pullback ends", "The pullback predicts earnings"], "A"),
        ("A breakout wick returns inside the range by the close. What is the most cautious reading?", ["Confirmed trend continuation", "Potential rejection or failed breakout", "Guaranteed short entry", "Volume no longer matters"], "B"),
    ],
    "crypto_derivatives": [
        ("Perpetual funding is positive. Who generally pays whom for that interval?", ["Longs pay shorts", "Shorts pay longs", "The exchange pays both", "Spot holders pay miners"], "A"),
        ("Price and open interest rise together. What can be concluded safely?", ["New leveraged exposure is entering, but direction durability is uncertain", "All new positions are longs", "Liquidation is impossible", "Spot demand is proven"], "A"),
        ("Futures trade above spot. What is this relationship called?", ["Backwardation only", "Contango or positive basis", "Negative delta", "Real yield"], "B"),
        ("Open interest falls during a sharp price decline. Which mechanism is plausible?", ["Position closing or liquidations", "Guaranteed fresh short buildup", "Rising spot supply is proven", "Funding must be zero"], "A"),
        ("Why should liquidation sit beyond the planned stop?", ["To preserve control of exit execution", "To increase leverage automatically", "To guarantee profit", "To eliminate slippage"], "A"),
    ],
    "stock_fundamentals": [
        ("EPS rises while operating income is flat and share count falls. What is a likely contributor?", ["Share repurchases", "Higher inventory alone", "A stock split", "Lower revenue by definition"], "A"),
        ("Which item is subtracted from operating cash flow to obtain a common free-cash-flow measure?", ["Capital expenditure", "Revenue", "Gross profit", "Market capitalization"], "A"),
        ("A very low P/E can reflect what risk?", ["Peak or unsustainable earnings", "Guaranteed undervaluation", "No business risk", "Automatic cash growth"], "A"),
        ("Revenue rises but gross margin falls. Which question is most relevant?", ["Whether pricing, mix, or input costs deteriorated", "Whether the ticker changed", "Whether shares have a logo", "Whether revenue is always sufficient"], "A"),
        ("Debt falls because nearly all cash was spent. Why may resilience not improve?", ["Liquidity also weakened", "Gross debt never matters", "Cash is a liability", "Interest expense must rise"], "A"),
    ],
    "macroeconomics": [
        ("Headline inflation falls while services inflation stays firm. What is the best policy reading?", ["Underlying pressure may keep policy cautious", "Cuts are guaranteed", "Inflation is exactly zero", "Employment data becomes irrelevant"], "A"),
        ("Payrolls rise while unemployment also rises. Which explanation is possible?", ["The labor force grew faster than employment", "The data are logically impossible", "Every worker lost two jobs", "Participation must have fallen"], "A"),
        ("Long-term yields rise because real growth expectations improve. Why can the equity effect be mixed?", ["Higher earnings hopes compete with higher discount rates", "Yields never affect valuation", "Growth always lowers earnings", "All sectors have identical duration"], "A"),
        ("A stronger currency commonly tightens conditions for which borrowers?", ["Entities with unhedged debt in that currency", "Only domestic cash savers", "Companies with no debt", "Nobody"], "A"),
        ("Why can inventory accumulation temporarily flatter GDP?", ["Production can exceed final demand", "Inventories are excluded from GDP", "It permanently raises productivity", "It guarantees consumption"], "A"),
    ],
    "risk_management": [
        ("A volatility-based stop becomes twice as wide. To keep dollar risk constant, what should usually happen?", ["Position size should fall", "Position size should double", "Leverage should become unlimited", "The stop should be ignored"], "A"),
        ("Five holdings share the same growth factor. Why may ticker count overstate diversification?", ["Correlations can rise together during stress", "Different names guarantee independence", "Factors affect only bonds", "Diversification removes all loss"], "A"),
        ("Which loss requires the larger percentage recovery?", ["A 40% drawdown", "A 10% drawdown", "They require the same recovery", "Neither requires recovery"], "A"),
        ("A stop order fills below its trigger after a gap. What risk materialized?", ["Slippage or gap risk", "Accounting risk", "Dividend yield", "Duration matching"], "A"),
        ("Why cap aggregate portfolio exposure?", ["Several individually small risks can occur together", "It guarantees every trade wins", "It increases concentration", "It removes market hours"], "A"),
    ],
    "scenario_analysis": [
        ("A bullish scenario has strong evidence but severe downside if invalidated. What must analysis include?", ["Probability and payoff asymmetry", "Only the bullish narrative", "A certainty claim", "No invalidation"], "A"),
        ("Two signals conflict. What is the best next step?", ["Define conditions that discriminate between scenarios", "Average them into certainty", "Discard all data", "Choose the newest signal automatically"], "A"),
        ("A base case is assigned the highest probability. What does that mean?", ["It is most likely among defined cases, not certain", "It must occur", "Other outcomes are impossible", "Its payoff is always highest"], "A"),
        ("Why stress-test a valuation with several margins?", ["To expose sensitivity to operating assumptions", "To find the one guaranteed price", "To eliminate uncertainty", "To avoid using revenue"], "A"),
        ("What invalidates a conditional thesis?", ["A predefined observation inconsistent with its mechanism", "Any small price movement", "A different opinion", "The passage of one minute"], "A"),
    ],
    "terminology_misc": [
        ("What is one basis point?", ["0.01 percentage point", "1 percentage point", "10 percentage points", "0.1 dollar"], "A"),
        ("Which statement distinguishes possibility from probability?", ["Possible means not ruled out; probable asserts greater likelihood", "They are identical", "Possible means certain", "Probable means impossible"], "A"),
        ("What does market capitalization measure?", ["Share price multiplied by shares outstanding", "Revenue minus expenses", "Cash plus debt", "Dividend divided by yield"], "A"),
        ("What is notional value?", ["The reference exposure used to size a contract", "Guaranteed maximum profit", "A company's cash balance", "A central-bank rate"], "A"),
        ("What does liquidity risk describe?", ["Difficulty transacting near expected price and size", "Guaranteed solvency", "Only tax expense", "Historical revenue growth"], "A"),
    ],
}


def _multiple_choice(category: str, n: int) -> tuple[str, str, tuple[str, ...], str]:
    question, options, correct = MCQ_BANK[category][n % len(MCQ_BANK[category])]
    contexts = ["During a portfolio review", "For a training case", "When documenting a thesis", "In a risk meeting", "For an end-of-day note"]
    labels = " ".join(f"{letter}) {option}" for letter, option in zip("ABCD", options, strict=True))
    prompt = f"{contexts[(n // 5) % len(contexts)]}, case {n + 1}: {question} {labels} End with exactly FINAL: <letter>."
    answer = options[ord(correct) - ord("A")]
    return prompt, f"{answer}.\nFINAL: {correct}", ("multiple_choice", category), "beginner" if n % 3 == 0 else "intermediate"


FACT_BANK = {
    "technical_analysis": [("What does ATR measure?", "Typical recent price range or volatility, not direction."), ("What is a support break?", "Price acceptance below an area where demand previously emerged."), ("What is relative strength?", "An asset's performance compared with a selected benchmark over a stated period.")],
    "crypto_derivatives": [("What is perpetual funding?", "A periodic transfer between long and short perpetual positions used to anchor the contract toward spot."), ("What is open interest?", "The number or notional value of derivative contracts that remain open."), ("What is futures basis?", "The difference between a futures price and its underlying spot price.")],
    "stock_fundamentals": [("What is diluted EPS?", "Net income attributable to common shareholders divided by diluted weighted-average shares."), ("What is gross margin?", "Revenue minus cost of revenue, expressed as a percentage of revenue."), ("What is free cash flow?", "A cash measure commonly defined as operating cash flow minus capital expenditure.")],
    "macroeconomics": [("What is a real interest rate approximation?", "The nominal interest rate minus inflation."), ("What is labor-force participation?", "The labor force as a percentage of the working-age population."), ("What is yield-curve inversion?", "A condition where shorter-maturity yields exceed longer-maturity yields for the selected tenors.")],
    "risk_management": [("What is position sizing?", "Choosing exposure so loss at a defined invalidation remains within the risk budget."), ("What is maximum drawdown?", "The largest peak-to-trough decline over the measured period."), ("What is concentration risk?", "The risk that exposures share an issuer, sector, factor, or other common driver.")],
    "scenario_analysis": [("What is a base case?", "The scenario judged most plausible under stated assumptions, not a certainty."), ("What is an invalidation condition?", "A predefined observation that materially contradicts the thesis mechanism."), ("What is sensitivity analysis?", "Testing how an output changes when one or more assumptions vary.")],
    "terminology_misc": [("What is a basis point?", "One hundredth of one percentage point."), ("What is market liquidity?", "The ability to transact meaningful size near the expected price without excessive impact."), ("What is uncertainty calibration?", "Matching confidence language to the strength and limitations of available evidence.")],
}


def _factual(category: str, n: int, json_only: bool) -> tuple[str, str, tuple[str, ...], str]:
    question, answer = FACT_BANK[category][n % len(FACT_BANK[category])]
    prompt = f"Reference check {n + 1}: {question} "
    if json_only:
        prompt += 'Return only valid JSON with keys "term", "definition", and "live_data_required".'
        term = question.removeprefix("What is ").removeprefix("What does ").rstrip("?")
        response = json.dumps({"term": term, "definition": answer, "live_data_required": False}, separators=(",", ":"))
    else:
        prompt += "Answer in one sentence of no more than 35 words."
        response = answer
    return prompt, response, ("definition", category), "beginner"


SIGNALS = {
    "technical_analysis": [("price breaks resistance", "volume contracts"), ("the weekly trend is up", "the daily structure turns down"), ("momentum improves", "price remains below a falling long-term average"), ("a support retest holds", "market breadth weakens")],
    "crypto_derivatives": [("spot price rises", "open interest falls"), ("funding becomes strongly positive", "spot buying remains firm"), ("price declines", "open interest also declines"), ("basis widens", "liquidation volume increases")],
    "stock_fundamentals": [("revenue accelerates", "operating margin contracts"), ("EPS rises", "operating cash flow falls"), ("debt declines", "cash liquidity also declines"), ("guidance rises", "customer concentration increases")],
    "macroeconomics": [("headline inflation falls", "services inflation stays firm"), ("payrolls grow", "unemployment rises"), ("real yields rise", "growth expectations improve"), ("the currency strengthens", "domestic equities rally")],
    "risk_management": [("individual trade risk is small", "positions share one factor"), ("historical volatility declines", "event risk approaches"), ("a stop is structurally valid", "liquidation is closer than the stop"), ("win rate improves", "average loss grows")],
    "scenario_analysis": [("the bull case gains supporting evidence", "its downside payoff worsens"), ("the base case remains most likely", "tail risk becomes more severe"), ("valuation falls", "earnings estimates also fall"), ("price confirms", "fundamentals lag")],
    "terminology_misc": [("an outcome is possible", "the evidence does not make it probable"), ("notional exposure is low", "liquidity is even lower"), ("market capitalization rises", "enterprise value falls"), ("a percentage rises", "the percentage-point change is small")],
}


def _analysis(category: str, n: int, long: bool) -> tuple[str, str, tuple[str, ...], str]:
    positive, caution = SIGNALS[category][n % len(SIGNALS[category])]
    prompt = f"Analysis case {n + 1}: {positive}, but {caution}. Explain the competing interpretations, name evidence that would discriminate between them, and state an invalidation. {'Use 90-130 words.' if long else 'Use 45-80 words.'}"
    core = f"Constructive interpretation: {positive}, which may support the thesis. Cautious interpretation: {caution}, so the apparent strength may be incomplete or temporary. Track follow-through in the first signal, whether the second signal stabilizes, and whether independent evidence confirms the same mechanism. The thesis is invalidated if the constructive signal reverses and the cautionary condition persists. Until those observations resolve the conflict, the appropriate conclusion is conditional rather than certain."
    if long:
        core += " Position size and confidence should reflect that uncertainty; a persuasive narrative is not a substitute for confirmation. The horizon must also be stated because short-term and long-term evidence can legitimately disagree."
    return prompt, core, ("conflicting_signals", "invalidation"), "advanced" if long else "intermediate"


LIVE_METRICS = [
    ("Bitcoin funding rate", "exchange and perpetual contract"),
    ("Ethereum open interest", "venue, contract set, and timestamp"),
    ("US headline CPI release", "release date and official series"),
    ("the unemployment rate", "country, release date, and official series"),
    ("a stock's current price", "ticker, venue, and timestamp"),
    ("the latest policy rate", "central bank and effective date"),
    ("today's options skew", "underlying, expiry, delta, and venue"),
    ("the live dollar index", "index definition, source, and timestamp"),
]


def _refusal(category: str, n: int) -> tuple[str, str, tuple[str, ...], str]:
    metric, requirements = LIVE_METRICS[n % len(LIVE_METRICS)]
    prompts = ["Give me", "State", "Report", "What is", "Tell me"]
    prompt = f"{prompts[n % len(prompts)]} the exact latest {metric} right now and briefly interpret it."
    response = f"I do not have access to live data, so I cannot provide or verify that value. It also requires the {requirements}. If you supply a sourced value with those details, I can interpret its direction, context, limitations, and relevant competing signals. Inventing an exact current figure would be misleading."
    return prompt, response, ("live_data", "hallucination_resistance"), "beginner"


def _instruction(category: str, n: int) -> tuple[str, str, tuple[str, ...], str]:
    score = 40 + n % 61
    volatility = ["low", "moderate", "high"][n % 3]
    bias = ["bearish", "neutral", "bullish"][n % 3]
    prompt = f"Using only these supplied inputs—signal score {score}/100, volatility {volatility}, and bias {bias}—return only JSON with keys signal_score, volatility, bias, confidence, and live_data_used. No Markdown."
    confidence = "low" if volatility == "high" else "medium" if volatility == "moderate" else "high"
    response = json.dumps({"signal_score": score, "volatility": volatility, "bias": bias, "confidence": confidence, "live_data_used": False}, separators=(",", ":"))
    return prompt, response, ("json", "instruction_following"), "intermediate"


def build_records(config_path: Path = DEFAULT_CONFIG) -> list[dict]:
    config = load_data_config(config_path)
    categories = [category for category, count in CATEGORY_COUNTS.items() for _ in range(count)]
    tasks = _task_specs()
    rng = random.Random(config.seed)
    rng.shuffle(categories)
    rng.shuffle(tasks)
    counters = {category: 0 for category in CATEGORY_COUNTS}
    records: list[dict] = []
    for ordinal, (category, task) in enumerate(zip(categories, tasks, strict=True), start=1):
        counters[category] += 1
        example_id = f"fp_{CATEGORY_PREFIX[category]}_{1000 + counters[category]:04d}"
        if task.task_type == "calculation":
            user, assistant, subtopics, difficulty = _calculation(category, ordinal)
            user = f"Calculation exercise {ordinal}: {user}"
            if task.response_length == "medium" and len(assistant) < 60:
                body, final_value = assistant.rsplit("\nFINAL:", maxsplit=1)
                assistant = (
                    f"{body} The calculation uses only the supplied hypothetical inputs."
                    f"\nFINAL:{final_value}"
                )
        elif task.task_type == "multiple_choice":
            user, assistant, subtopics, difficulty = _multiple_choice(category, ordinal)
        elif task.task_type == "factual":
            user, assistant, subtopics, difficulty = _factual(category, ordinal, task.response_format == "json_only")
        elif task.task_type == "analysis":
            user, assistant, subtopics, difficulty = _analysis(category, ordinal, task.response_length == "long")
        elif task.task_type == "refusal":
            user, assistant, subtopics, difficulty = _refusal(category, ordinal)
            user = f"Live-data boundary exercise {ordinal}: {user}"
        else:
            user, assistant, subtopics, difficulty = _instruction(category, ordinal)
            user = f"JSON exercise {ordinal} for {category.replace('_', ' ')}: {user}"
        records.append({
            "id": example_id,
            "messages": [
                {"role": "system", "content": config.system_prompt},
                {"role": "user", "content": user},
                {"role": "assistant", "content": assistant},
            ],
            "metadata": {
                "category": category,
                "subtopics": list(subtopics),
                "difficulty": difficulty,
                "task_type": task.task_type,
                "response_format": task.response_format,
                "response_length": task.response_length,
                "source": {"type": "synthetic", "reference": "finpulse-llm-stage5b-rule-authored", "license": "project-original"},
                "review": {"status": "reviewed", "reviewer": "stage5b-deterministic-quality-review"},
            },
        })
    return sorted(records, key=lambda item: item["id"])


def _serialize(records: list[dict]) -> str:
    return "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in records)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--write", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    text = _serialize(build_records(args.config))
    if not args.write:
        if not args.output.is_file() or args.output.read_text(encoding="utf-8") != text:
            raise SystemExit("Stage 5B JSONL differs from its deterministic builder")
        print("Verified Stage 5B corrective corpus: 500 examples")
        return 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8")
    print(f"Wrote 500 Stage 5B examples to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
