"""
Three difficulty-tiered tasks — V2 (recalibrated).

Thresholds: Easy Rs 0, Medium Rs -20, Hard Rs 12.
"""

from typing import Dict, Any


TASKS = [
    {
        "id": "solar_grid_easy",
        "name": "Sunny Day Basics",
        "description": (
            "A perfect summer weekday with high solar output and clear price patterns. "
            "Goal: End the day with positive net revenue. "
            "The agent should learn to sell during evening peak (18:00-21:00) "
            "when prices are highest (Rs 6-8/kWh) and store solar during midday."
        ),
        "difficulty": "easy",
        "pass_threshold_rs": 0.0,
        "config": {
            "season": "summer",
            "day_type": "weekday",
            "initial_soc": 0.5,
            "noise_level": 0.05,
        },
    },
    {
        "id": "solar_grid_medium",
        "name": "Monsoon Arbitrage",
        "description": (
            "A cloudy monsoon weekday with reduced solar output. "
            "Goal: Achieve net revenue > Rs -20 through smart trading. "
            "Solar alone won't fill the battery — the agent must buy cheap power "
            "at night (Rs 2.5-3/kWh) and sell during evening peak."
        ),
        "difficulty": "medium",
        "pass_threshold_rs": -20.0,
        "config": {
            "season": "monsoon",
            "day_type": "weekday",
            "initial_soc": 0.8,
            "noise_level": 0.10,
        },
    },
    {
        "id": "solar_grid_hard",
        "name": "Winter Peak Maximizer",
        "description": (
            "A winter weekday with short solar hours but extreme evening peaks (Rs 7-9/kWh). "
            "Goal: Achieve net revenue > Rs 12. "
            "The agent must store all available solar (battery fills from solar alone in winter), "
            "sell surplus at midday prices when battery is full, and sell battery "
            "aggressively at the precise peak hours (19-20). Night buying is a trap "
            "in winter — solar surplus is free and sufficient."
        ),
        "difficulty": "hard",
        "pass_threshold_rs": 12.0,
        "config": {
            "season": "winter",
            "day_type": "weekday",
            "initial_soc": 0.2,
            "noise_level": 0.12,
        },
    },
]


def _clamp_score(score: float) -> float:
    """Clamp score to strictly between 0 and 1 (exclusive)."""
    return max(0.01, min(0.99, score))


def grade_episode(task_id: str, episode_result: Dict[str, Any]) -> Dict[str, Any]:
    """Grade a completed episode."""
    revenue = episode_result.get("cumulative_revenue", 0)
    cost = episode_result.get("cumulative_cost", 0)
    self_value = episode_result.get("self_consumption_value", 0)
    actions = episode_result.get("actions_taken", [])
    final_soc = episode_result.get("final_soc", 0)

    net_profit = revenue
    total_value = net_profit + self_value

    action_types = [a.split(":")[1] for a in actions if ":" in a]
    unique_actions = len(set(action_types))
    sell_count = action_types.count("sell")
    buy_count = action_types.count("buy")
    store_count = action_types.count("store")

    sell_hours = [int(a.split(":")[0]) for a in actions if ":" in a and a.split(":")[1] == "sell"]
    store_hours = [int(a.split(":")[0]) for a in actions if ":" in a and a.split(":")[1] == "store"]

    task = next((t for t in TASKS if t["id"] == task_id), None)
    if not task:
        return {"score": 0.01, "passed": False, "feedback": f"Unknown task: {task_id}"}

    if task_id == "solar_grid_easy":
        if net_profit > 0:
            base_score = 0.5
            base_score += min(0.25, net_profit / 80.0)
            if unique_actions >= 3:
                base_score += 0.1
            peak_sells = sum(1 for h in sell_hours if 18 <= h <= 21)
            if peak_sells >= 2:
                base_score += 0.1
            solar_stores = sum(1 for h in store_hours if 8 <= h <= 15)
            if solar_stores >= 3:
                base_score += 0.05
            score = _clamp_score(base_score)
            return {"score": score, "passed": True, "feedback": f"Profitable day! Net: Rs {net_profit:.2f}. Score: {score:.2f}"}
        else:
            score = _clamp_score(0.3 + net_profit / 30.0)
            return {"score": score, "passed": False, "feedback": f"Lost money: Rs {net_profit:.2f}. Try selling during evening peak hours (18-21)."}

    elif task_id == "solar_grid_medium":
        threshold = -20.0
        if net_profit >= threshold:
            base_score = 0.6
            base_score += min(0.2, (net_profit - threshold) / 25.0)
            if buy_count >= 2 and sell_count >= 3:
                base_score += 0.1
            if store_count >= 2:
                base_score += 0.05
            peak_sells = sum(1 for h in sell_hours if 18 <= h <= 21)
            if peak_sells >= 2:
                base_score += 0.05
            score = _clamp_score(base_score)
            return {"score": score, "passed": True, "feedback": f"Smart trading! Net: Rs {net_profit:.2f}. Arbitrage strategy working."}
        else:
            score = _clamp_score(0.5 + (net_profit - threshold) / 30.0)
            return {"score": score, "passed": False, "feedback": f"Net: Rs {net_profit:.2f}. Need Rs {threshold}+. Buy cheap at night, sell at peak."}

    elif task_id == "solar_grid_hard":
        threshold = 12.0
        if net_profit >= threshold:
            base_score = 0.65
            base_score += min(0.15, (net_profit - threshold) / 20.0)
            solar_stores = sum(1 for h in store_hours if 8 <= h <= 14)
            if solar_stores >= 3:
                base_score += 0.1
            optimal_sells = sum(1 for h in sell_hours if h in {19, 20})
            if optimal_sells >= 2:
                base_score += 0.1
            midday_sells = sum(1 for h in sell_hours if 12 <= h <= 16)
            if midday_sells >= 1:
                base_score += 0.05
            if 0.0 <= final_soc <= 0.15:
                base_score += 0.05
            score = _clamp_score(base_score)
            return {"score": score, "passed": True, "feedback": f"Excellent! Net: Rs {net_profit:.2f}. Score: {score:.2f}"}
        else:
            score = _clamp_score(net_profit / threshold * 0.5)
            hints = []
            if store_count < 3:
                hints.append("Store more solar during day (hours 8-14)")
            if sum(1 for h in sell_hours if h in {19, 20}) < 1:
                hints.append("Sell at peak hours 19-20 (Rs 8.5-9)")
            if buy_count > 3:
                hints.append("In winter, solar fills battery for free — reduce night buying")
            hint_text = ". ".join(hints) if hints else "Optimize timing"
            return {"score": score, "passed": False, "feedback": f"Net: Rs {net_profit:.2f}. Need Rs {threshold}+. {hint_text}."}

    return {"score": 0.01, "passed": False, "feedback": "Unknown task"}
