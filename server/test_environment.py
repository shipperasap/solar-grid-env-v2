"""Unit tests for the Solar Grid Environment."""

import sys
import os

if __package__ is None or __package__ == "":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from environment import SolarGridEnvironment
    from models import SolarGridAction, ActionType
    from tasks import TASKS, grade_episode
else:
    from .environment import SolarGridEnvironment
    from .models import SolarGridAction, ActionType
    from .tasks import TASKS, grade_episode


def test_reset():
    env = SolarGridEnvironment()
    obs = env.reset()
    assert obs.hour == 0
    assert 0.0 <= obs.battery_soc <= 1.0
    assert obs.hours_remaining == 23
    assert obs.current_price > 0
    assert len(obs.next_3h_prices) == 3
    print("test_reset passed")


def test_full_episode():
    env = SolarGridEnvironment()
    env.reset()
    steps = 0
    while not env.state.done:
        action = SolarGridAction(action_type=ActionType.HOLD, amount_kwh=1.0)
        result = env.step(action)
        steps += 1
        assert "observation" in result
        assert "reward" in result
        assert "done" in result
    assert steps == 24
    assert env.state.done
    print(f"test_full_episode passed ({steps} steps)")


def test_sell_action():
    env = SolarGridEnvironment()
    env.reset()
    env.state.battery_soc = 0.8
    env.state.hour = 19
    action = SolarGridAction(action_type=ActionType.SELL, amount_kwh=5.0)
    result = env.step(action)
    assert result["reward"] != 0
    assert env.state.battery_soc < 0.8
    print("test_sell_action passed")


def test_buy_action():
    env = SolarGridEnvironment()
    env.reset()
    env.state.battery_soc = 0.2
    env.state.hour = 2
    action = SolarGridAction(action_type=ActionType.BUY, amount_kwh=5.0)
    result = env.step(action)
    assert env.state.battery_soc > 0.2
    print("test_buy_action passed")


def test_store_action():
    env = SolarGridEnvironment()
    env.reset()
    env.state.battery_soc = 0.3
    env.state.hour = 12
    env.state.solar_profile[12] = 4.5
    env.state.consumption_profile[12] = 0.5
    action = SolarGridAction(action_type=ActionType.STORE, amount_kwh=3.0)
    result = env.step(action)
    assert env.state.battery_soc >= 0.3
    print("test_store_action passed")


def test_grader():
    for task in TASKS:
        result = grade_episode(task["id"], {
            "cumulative_revenue": 25.0,
            "cumulative_cost": 5.0,
            "self_consumption_value": 10.0,
            "actions_taken": [f"{h}:sell:2.0" for h in range(18, 22)] + [f"{h}:store:3.0" for h in range(8, 16)],
            "final_soc": 0.15,
        })
        assert "score" in result
        assert "passed" in result
        assert "feedback" in result
        print(f"test_grader [{task['id']}] score={result['score']:.2f} passed={result['passed']}")


def test_battery_constraints():
    env = SolarGridEnvironment()
    env.reset()
    env.state.battery_soc = 0.0
    env.state.hour = 19
    action = SolarGridAction(action_type=ActionType.SELL, amount_kwh=5.0)
    result = env.step(action)
    assert env.state.battery_soc >= 0.0
    print("test_battery_constraints passed")


if __name__ == "__main__":
    test_reset()
    test_full_episode()
    test_sell_action()
    test_buy_action()
    test_store_action()
    test_grader()
    test_battery_constraints()
    print("\nAll tests passed!")
