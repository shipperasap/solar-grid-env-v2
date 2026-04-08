"""
Inference script for Solar Grid Environment.
Uses OpenAI Client for LLM-based decision making.
Prints results in exact OpenEnv stdout format.

Env vars required:
  API_BASE_URL  - The API endpoint for the LLM
  MODEL_NAME    - The model identifier to use for inference
  HF_TOKEN      - Your Hugging Face / API key
"""

import os
import sys
import json

# Ensure server package is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from server.environment import SolarGridEnvironment
from server.models import SolarGridAction, ActionType
from server.tasks import TASKS, grade_episode
from server.price_engine import generate_price_profile, generate_solar_profile, generate_consumption_profile

from server.inference import smart_policy as fallback_policy_impl
from openai import OpenAI

# --- Configuration from environment ---
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN,
)

BENCHMARK = "solar_grid_env"

SYSTEM_PROMPT = """You are an energy trading agent for an Indian rooftop solar prosumer.
You manage a 5kW solar panel + 13.5kWh battery on the IEX Day-Ahead Market.

Each hour you must choose ONE action:
- sell: Sell solar surplus + battery power to grid at current IEX price
- store: Store solar surplus into battery for later
- buy: Buy cheap grid power to charge battery (for arbitrage)
- hold: Self-consume solar power, auto-store surplus

And specify amount_kwh (0.0 to 10.0).

Strategy tips:
- Night (0-5): Buy cheap if price < 3.5 Rs/kWh and battery < 70%
- Midday (6-16): Store solar surplus for evening peak
- Evening peak (18-21): SELL when prices are highest (6-9 Rs/kWh)
- Avoid selling during cheap hours when peak is coming

Respond with ONLY valid JSON: {"action_type": "sell|store|buy|hold", "amount_kwh": <float>}"""


def get_llm_action(observation: dict) -> SolarGridAction:
    """Use OpenAI-compatible LLM to decide action."""
    obs_summary = (
        f"Hour: {observation['hour']}, Price: Rs{observation['current_price']:.2f}/kWh, "
        f"Next 3h prices: {observation['next_3h_prices']}, "
        f"Solar: {observation['solar_generation_kwh']:.2f}kWh, "
        f"Battery SOC: {observation['battery_soc']:.0%}, "
        f"Consumption: {observation['energy_consumed_kwh']:.2f}kWh, "
        f"Revenue so far: Rs{observation['cumulative_revenue']:.2f}, "
        f"Hours left: {observation['hours_remaining']}, "
        f"Season: {observation['season']}, Day: {observation['day_type']}"
    )

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": obs_summary},
            ],
            temperature=0.2,
            max_tokens=100,
        )
        content = response.choices[0].message.content.strip()
        # Parse JSON from response (handle markdown code blocks)
        if "```" in content:
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
            content = content.strip()
        data = json.loads(content)
        return SolarGridAction(
            action_type=ActionType(data["action_type"]),
            amount_kwh=min(10.0, max(0.0, float(data["amount_kwh"]))),
        )
    except Exception:
        # Fallback to heuristic if LLM fails
        return fallback_policy_impl(observation)


def run_task(task_id: str):
    """Run a full episode for one task and print in OpenEnv format."""
    env = SolarGridEnvironment()
    all_rewards = []
    step_count = 0
    score = 0.01
    success = False

    print(f"[START] task={task_id} env={BENCHMARK} model={MODEL_NAME}", flush=True)

    try:
        obs_obj = env.reset()

        task = next((t for t in TASKS if t["id"] == task_id), None)
        if task:
            config = task["config"]
            env.state.season = config["season"]
            env.state.day_type = config["day_type"]
            env.state.battery_soc = config["initial_soc"]
            env.state.price_profile = generate_price_profile(
                config["season"], config["day_type"], config.get("noise_level", 0.1)
            )
            env.state.solar_profile = generate_solar_profile(config["season"], 5.0)
            env.state.consumption_profile = generate_consumption_profile(
                config["day_type"], config["season"]
            )
            obs_obj = env._make_observation(f"Task: {task['name']}")

        obs = obs_obj.model_dump()

        while not env.state.done:
            try:
                action = get_llm_action(obs)
                result = env.step(action)
                obs = result["observation"]
                reward = result["reward"]
                done = result["done"]

                all_rewards.append(reward)
                step_count += 1

                action_str = f"{action.action_type.value}:{action.amount_kwh:.1f}"
                print(
                    f"[STEP] step={step_count} action={action_str} "
                    f"reward={reward:.2f} done={str(done).lower()} error=null",
                    flush=True,
                )

            except Exception as e:
                step_count += 1
                all_rewards.append(0.0)
                error_msg = str(e).replace("\n", " ")
                print(
                    f"[STEP] step={step_count} action=error "
                    f"reward=0.00 done=false error={error_msg}",
                    flush=True,
                )

        # Grade
        grade = grade_episode(task_id, {
            "cumulative_revenue": env.state.cumulative_revenue,
            "cumulative_cost": env.state.cumulative_cost,
            "self_consumption_value": env.state.cumulative_self_consumption_value,
            "actions_taken": env.state.actions_taken,
            "final_soc": env.state.battery_soc,
        })

        score = grade["score"]
        success = grade["passed"]

    finally:
        rewards_str = ",".join(f"{r:.2f}" for r in all_rewards)
        print(
            f"[END] success={str(success).lower()} steps={step_count} "
            f"score={score:.2f} rewards={rewards_str}",
            flush=True,
        )


if __name__ == "__main__":
    for task in TASKS:
        run_task(task["id"])
