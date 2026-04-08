"""
Reward function for Solar Grid Arbitrage Environment — V2 (improved).

Key improvements over V1:
1. Night buying only rewarded when SOC is low AND season has weak solar (monsoon)
2. Selling solar surplus when battery is full now rewarded (not just peak selling)
3. Stronger storage reward during solar hours
4. Much steeper solar waste penalty
5. End-of-episode bonus scales by season (winter max ~25, summer max ~60)
6. Self-consumption (Hold) properly valued — avoiding retail IS revenue
"""

from typing import Dict


def compute_reward(
    action_type: str,
    amount_kwh: float,
    actual_amount: float,
    current_price: float,
    retail_tariff: float,
    solar_available: float,
    consumption: float,
    battery_soc_before: float,
    battery_soc_after: float,
    battery_capacity: float,
    hour: int,
    hours_remaining: int,
    season: str,
    cumulative_revenue: float,
    price_profile: list,
) -> Dict[str, float]:
    """
    Compute multi-component reward. Returns dict with total + breakdown.
    """
    reward_components = {
        "profit_component": 0.0,
        "strategy_component": 0.0,
        "efficiency_component": 0.0,
        "penalty": 0.0,
    }

    net_solar = max(0, solar_available - consumption)

    # ================================================================
    # COMPONENT 1: IMMEDIATE PROFIT/LOSS (50% weight)
    # ================================================================
    if action_type == "sell":
        revenue = actual_amount * current_price
        if solar_available > consumption:
            # Selling genuine surplus — full credit
            reward_components["profit_component"] = revenue / 10.0
        else:
            net_benefit = revenue - (actual_amount * retail_tariff * 0.5)
            reward_components["profit_component"] = net_benefit / 10.0

    elif action_type == "buy":
        cost = actual_amount * current_price
        future_prices = price_profile[hour + 1:min(hour + 8, 24)] if hour < 23 else []
        if future_prices:
            max_future = max(future_prices)
            spread = max_future - current_price
            if spread > 2.0:
                potential_arbitrage = spread * actual_amount * 0.35
                reward_components["profit_component"] = (-cost + potential_arbitrage) / 10.0
            else:
                reward_components["profit_component"] = -cost / 10.0
        else:
            reward_components["profit_component"] = -cost / 10.0

    elif action_type == "store":
        future_prices = price_profile[hour + 1:min(hour + 10, 24)] if hour < 23 else []
        if future_prices:
            expected_peak = max(future_prices)
            if solar_available > consumption:
                # Storing free solar — high value
                storage_value = expected_peak * actual_amount * 0.4 / 10.0
            else:
                storage_value = (expected_peak - current_price) * actual_amount * 0.25 / 10.0
            reward_components["profit_component"] = max(-0.05, storage_value)
        else:
            reward_components["profit_component"] = -0.01

    elif action_type == "hold":
        self_consumed = min(solar_available, consumption)
        savings = self_consumed * retail_tariff
        reward_components["profit_component"] = savings / 10.0

    # ================================================================
    # COMPONENT 2: STRATEGIC SHAPING (25% weight)
    # ================================================================

    # Store during solar hours when battery isn't full
    if action_type == "store" and 7 <= hour <= 16:
        if solar_available > consumption and battery_soc_before < 0.9:
            reward_components["strategy_component"] = 0.20
        elif battery_soc_before < 0.5:
            reward_components["strategy_component"] = 0.25

    # Sell solar surplus when battery is full (free revenue!)
    if action_type == "sell" and battery_soc_before > 0.9 and solar_available > consumption:
        if current_price > 2.0:
            reward_components["strategy_component"] = 0.12

    # Sell during peak hours
    if action_type == "sell" and 18 <= hour <= 21:
        if current_price > 5.5:
            reward_components["strategy_component"] = max(
                reward_components["strategy_component"],
                0.20 * min(1.5, current_price / 7.0)
            )

    # Penalize selling cheap when peak is coming (but not if battery is full)
    if action_type == "sell" and hour < 16 and current_price < 4.0:
        future_peak = max(price_profile[17:22]) if len(price_profile) >= 22 else 4.0
        if future_peak > current_price * 1.5 and battery_soc_before < 0.95:
            reward_components["strategy_component"] = -0.12

    # Night buying: only reward when actually smart
    if action_type == "buy" and hour <= 5 and current_price < 3.0:
        if season == "monsoon" and battery_soc_before < 0.4:
            reward_components["strategy_component"] = 0.15
        elif battery_soc_before < 0.25:
            reward_components["strategy_component"] = 0.10
        else:
            reward_components["strategy_component"] = -0.05

    # Penalize buying when solar is available
    if action_type == "buy" and 8 <= hour <= 15 and solar_available > consumption:
        reward_components["strategy_component"] = -0.15

    # ================================================================
    # COMPONENT 3: EFFICIENCY (15% weight)
    # ================================================================

    if battery_soc_after < 0.05:
        reward_components["efficiency_component"] = -0.08
    elif battery_soc_after < 0.1:
        reward_components["efficiency_component"] = -0.04
    elif battery_soc_after > 0.95:
        reward_components["efficiency_component"] = -0.02
    elif 0.2 <= battery_soc_after <= 0.8:
        reward_components["efficiency_component"] = 0.02

    # Solar waste penalty — STRONGER in V2
    if solar_available > 0 and net_solar > 0.5:
        if action_type == "hold" and battery_soc_before > 0.95:
            wasted = net_solar
            waste_penalty = -0.12 * min(1.0, wasted / 3.0)
            reward_components["efficiency_component"] += waste_penalty
        elif action_type not in ("store", "sell") and net_solar > 0.5:
            wasted = net_solar
            waste_penalty = -0.10 * min(1.0, wasted / 3.0)
            reward_components["efficiency_component"] += waste_penalty

    # ================================================================
    # COMPONENT 4: PENALTIES (10% weight)
    # ================================================================

    if amount_kwh > actual_amount * 1.01 and actual_amount > 0:
        reward_components["penalty"] = -0.05

    if actual_amount == 0 and amount_kwh > 0.1:
        reward_components["penalty"] = -0.10

    # ================================================================
    # FINAL REWARD
    # ================================================================
    total = (
        reward_components["profit_component"] * 0.50
        + reward_components["strategy_component"] * 0.25
        + reward_components["efficiency_component"] * 0.15
        + reward_components["penalty"] * 0.10
    )

    # End-of-episode bonus: SCALED BY SEASON
    if hours_remaining == 0:
        net_profit = cumulative_revenue
        season_great = {"summer": 35, "monsoon": 18, "winter": 22}
        season_good = {"summer": 18, "monsoon": 10, "winter": 12}
        great_threshold = season_great.get(season, 25)
        good_threshold = season_good.get(season, 12)

        if net_profit > great_threshold:
            total += 0.5
        elif net_profit > good_threshold:
            total += 0.25
        elif net_profit > 0:
            total += 0.1
        else:
            total -= 0.3

    reward_components["total"] = max(-1.0, min(1.0, total))
    return reward_components
