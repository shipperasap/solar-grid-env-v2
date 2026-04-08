"""
Inference script for Solar Grid Environment.
Runs a smart heuristic policy and prints results in OpenEnv format.

Works both as:
  python server/inference.py          (standalone)
  python -m server.inference          (module)
"""

import sys
import os

# Support both standalone and module execution
if __package__ is None or __package__ == "":
    # Running as standalone script — fix imports
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from environment import SolarGridEnvironment
    from models import SolarGridAction, ActionType
    from tasks import TASKS, grade_episode
    from price_engine import generate_price_profile, generate_solar_profile, generate_consumption_profile
else:
    # Running as python -m server.inference
    from .environment import SolarGridEnvironment
    from .models import SolarGridAction, ActionType
    from .tasks import TASKS, grade_episode
    from .price_engine import generate_price_profile, generate_solar_profile, generate_consumption_profile


def estimate_remaining_consumption(hour: int) -> float:
    """Estimate total kWh consumption from current hour to end of day."""
    hourly_avg = [
        0.35, 0.30, 0.30, 0.30, 0.30, 0.40,
        0.65, 1.00, 0.80, 0.55, 0.55, 0.65,
        0.65, 0.80, 1.00, 1.30, 1.55, 1.95,
        2.60, 3.25, 2.60, 1.95, 1.30, 0.65,
    ]
    return sum(hourly_avg[h] for h in range(hour + 1, 24))


def smart_policy(observation: dict) -> SolarGridAction:
    """
    V2 heuristic policy — season-aware, passes all 4 tasks.

    Key insight per season:
    - Summer: Battery fills from solar. Store midday, sell surplus + peak.
    - Monsoon: Weak solar. MUST buy cheap at night. Sell at peak.
    - Winter: Solar fills battery alone. DON'T buy at night. Store solar, sell surplus midday, sell battery at peak.
    """
    hour = observation["hour"]
    price = observation["current_price"]
    solar = observation["solar_generation_kwh"]
    soc = observation["battery_soc"]
    consumption = observation["energy_consumed_kwh"]
    next_prices = observation["next_3h_prices"]
    hours_left = observation["hours_remaining"]
    season = observation.get("season", "summer")

    net_solar = solar - consumption
    max_future = max(next_prices) if next_prices and max(next_prices) > 0 else price
    battery_kwh = soc * 13.5

    if season == "monsoon":
        # Buy aggressively at night to fill battery for the day and peak arbitrage
        if hour <= 5 and price < 5.0 and soc < 0.95:
            return SolarGridAction(action_type=ActionType.BUY, amount_kwh=min(5.0, (1.0 - soc) * 13.5 / 0.92))
        # Store solar
        if 6 <= hour <= 17 and net_solar > 0.0 and soc < 1.0:
            return SolarGridAction(action_type=ActionType.STORE, amount_kwh=min(net_solar, 5.0))
        # Sell at peak, but leave 40% absolute minimum for night time consumption
        if 18 <= hour <= 21 and price > 5.5 and soc > 0.40:
            max_safe_sell = (soc - 0.40) * 13.5
            sell = min(5.0, battery_kwh * 0.5)
            # Push harder at absolute peak (Hr 19-20)
            if hour in [19, 20]:
                sell = min(5.0, battery_kwh * 0.8)
            return SolarGridAction(action_type=ActionType.SELL, amount_kwh=max(0.0, min(sell, max_safe_sell)))

    # === WINTER: Solar fills battery, don't buy at night ===
    elif season == "winter":
        if hour <= 5 and price < 2.8 and soc < 0.15:
            return SolarGridAction(action_type=ActionType.BUY, amount_kwh=max(0.5, min(3.0, (0.3 - soc) * 13.5 / 0.92)))
        if 7 <= hour <= 15 and net_solar > 0.3 and soc < 0.95:
            return SolarGridAction(action_type=ActionType.STORE, amount_kwh=min(net_solar, 5.0))
        if 10 <= hour <= 16 and net_solar > 0.5 and soc > 0.92:
            return SolarGridAction(action_type=ActionType.SELL, amount_kwh=min(net_solar, 5.0))
        if 18 <= hour <= 21 and price > 5.5 and soc > 0.1:
            if hour in [19, 20] and price > 7.0:
                sell = min(5.0, battery_kwh * 0.7)
            else:
                sell = min(5.0, battery_kwh * 0.4)
            return SolarGridAction(action_type=ActionType.SELL, amount_kwh=max(0.5, sell))

    # === SUMMER: Real IEX data — midday ~1 Rs, night Rs 3-9, evening Rs 4-10 ===
    elif season == "summer":
        day = observation.get("day_type", "weekday")
        # Estimate remaining consumption for night to determine reserve
        remaining_consumption = estimate_remaining_consumption(hour)
        # Reserve enough SOC to cover remaining consumption from battery
        # (avoids buying at expensive night prices Rs 3-9)
        # Weekend: higher reserve since consumption is higher and prices lower
        reserve_mult = 1.3 if day == "weekend" else 1.1
        reserve_kwh = min(remaining_consumption * reserve_mult, 13.5 * 0.65)
        reserve_soc = reserve_kwh / 13.5

        # Buy during solar glut when prices crash to ~1-2 Rs/kWh (best arbitrage)
        if 9 <= hour <= 14 and price < 2.0 and soc < 0.85:
            return SolarGridAction(action_type=ActionType.BUY, amount_kwh=min(5.0, (0.95 - soc) * 13.5 / 0.92))
        # Store solar surplus
        if 6 <= hour <= 16 and net_solar > 0.3 and soc < 0.95:
            return SolarGridAction(action_type=ActionType.STORE, amount_kwh=min(net_solar, 5.0))
        # Sell surplus when battery full during solar hours
        if 8 <= hour <= 16 and net_solar > 0.5 and soc > 0.92:
            return SolarGridAction(action_type=ActionType.SELL, amount_kwh=min(net_solar, 5.0))
        # Sell at evening peak — only sell what's above reserve
        # Weekend: higher price threshold since peaks are weaker
        sell_threshold = 5.5 if day == "weekend" else 4.5
        if 18 <= hour <= 20 and price > sell_threshold and soc > reserve_soc + 0.05:
            sellable_kwh = (soc - reserve_soc) * 13.5
            sell = min(5.0, sellable_kwh * 0.7)
            if hour in [19, 20] and price > 7.0:
                sell = min(5.0, sellable_kwh * 0.9)
            return SolarGridAction(action_type=ActionType.SELL, amount_kwh=max(0.5, sell))
        # Late evening: sell only if price is very high and we still have surplus over reserve
        if hour >= 21 and soc > reserve_soc + 0.1 and price > 6.0:
            return SolarGridAction(action_type=ActionType.SELL, amount_kwh=max(0.5, min(2.0, (soc - reserve_soc) * 13.5)))

    # Default for any other season
    return SolarGridAction(action_type=ActionType.HOLD, amount_kwh=min(solar, consumption, 5.0))


def run_inference(task_id: str):
    """Run a full episode and print in OpenEnv format."""
    env = SolarGridEnvironment()
    obs_obj = env.reset()

    # Apply task config
    task = next((t for t in TASKS if t["id"] == task_id), None)
    if task:
        config = task["config"]
        env.state.season = config["season"]
        env.state.day_type = config["day_type"]
        env.state.battery_soc = config["initial_soc"]
        env.state.price_profile = generate_price_profile(config["season"], config["day_type"], config.get("noise_level", 0.1))
        env.state.solar_profile = generate_solar_profile(config["season"], 5.0)
        env.state.consumption_profile = generate_consumption_profile(config["day_type"], config["season"])
        obs_obj = env._make_observation(f"Task: {task['name']}")

    obs = obs_obj.model_dump()
    print(f"[START] {task_id}")

    total_reward = 0.0
    while not env.state.done:
        action = smart_policy(obs)
        result = env.step(action)
        obs = result["observation"]
        reward = result["reward"]
        total_reward += reward

        hour = obs.get("hour", env.state.hour)
        soc = obs.get("battery_soc", env.state.battery_soc)
        price = env.state.price_profile[hour - 1] if hour > 0 else 0

        print(f"[STEP] {hour-1} | {action.action_type.value} | {action.amount_kwh:.1f} | Rs{price:.2f} | r={reward:.3f} | soc={soc:.2f}")

    # Grade
    grade = grade_episode(task_id, {
        "cumulative_revenue": env.state.cumulative_revenue,
        "cumulative_cost": env.state.cumulative_cost,
        "self_consumption_value": env.state.cumulative_self_consumption_value,
        "actions_taken": env.state.actions_taken,
        "final_soc": env.state.battery_soc,
    })

    print(f"[END] {grade['score']:.2f} | {'PASS' if grade['passed'] else 'FAIL'} | Rs{env.state.cumulative_revenue:.2f}")
    print(f"Feedback: {grade['feedback']}")
    print(f"Total shaped reward: {total_reward:.3f}")


if __name__ == "__main__":
    print("=" * 60)
    print("Solar Grid Arbitrage -- Inference")
    print("=" * 60)
    for task in TASKS:
        print(f"\n--- {task['name']} ({task['difficulty']}) ---")
        run_inference(task["id"])
        print()
