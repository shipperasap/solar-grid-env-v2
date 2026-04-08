import json
import uuid
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from contextlib import asynccontextmanager

from .environment import SolarGridEnvironment
from .models import SolarGridAction
from .tasks import TASKS, grade_episode

# Store environments per session
environments: dict = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    environments.clear()

app = FastAPI(
    title="Solar Grid Arbitrage Environment",
    description="Indian electricity grid arbitrage RL environment for IEX Day-Ahead Market",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
async def root():
    return {
        "name": "solar_grid_env",
        "version": "1.0.0",
        "description": "Indian rooftop solar prosumer with battery making hourly IEX DAM trading decisions",
        "tasks": [{"id": t["id"], "name": t["name"], "difficulty": t["difficulty"]} for t in TASKS],
    }


@app.get("/tasks")
async def get_tasks():
    return {"tasks": TASKS}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    session_id = str(uuid.uuid4())
    env = SolarGridEnvironment()
    environments[session_id] = env

    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            action_type = message.get("action", "")

            if action_type == "reset":
                observation = env.reset()
                task_id = message.get("task_id")

                if task_id:
                    task = next((t for t in TASKS if t["id"] == task_id), None)
                    if task:
                        config = task["config"]
                        from .price_engine import generate_price_profile, generate_solar_profile, generate_consumption_profile
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
                        observation = env._make_observation(
                            f"Task: {task['name']}. {task['description'][:100]}..."
                        )

                await websocket.send_text(json.dumps({
                    "observation": observation.model_dump(),
                    "reward": 0.0,
                    "done": False,
                    "state": {"episode_id": env.state.episode_id, "step_count": 0},
                }))

            elif action_type == "step":
                action_data = message.get("data", {})
                action = SolarGridAction(**action_data)
                result = env.step(action)

                if result["done"]:
                    task_id = message.get("task_id")
                    if task_id:
                        grade = grade_episode(task_id, {
                            "cumulative_revenue": env.state.cumulative_revenue,
                            "cumulative_cost": env.state.cumulative_cost,
                            "self_consumption_value": env.state.cumulative_self_consumption_value,
                            "actions_taken": env.state.actions_taken,
                            "final_soc": env.state.battery_soc,
                        })
                        result["grade"] = grade

                await websocket.send_text(json.dumps(result))

            elif action_type == "state":
                await websocket.send_text(json.dumps(env.get_state()))

            else:
                await websocket.send_text(json.dumps({
                    "error": f"Unknown action: {action_type}. Use 'reset', 'step', or 'state'."
                }))

    except WebSocketDisconnect:
        if session_id in environments:
            del environments[session_id]
    except Exception as e:
        await websocket.send_text(json.dumps({"error": str(e)}))
        if session_id in environments:
            del environments[session_id]


# === REST endpoints for non-WebSocket testing ===

@app.post("/reset")
async def rest_reset(task_id: str = None):
    env = SolarGridEnvironment()
    obs = env.reset()
    session_id = str(uuid.uuid4())
    environments[session_id] = env

    if task_id:
        task = next((t for t in TASKS if t["id"] == task_id), None)
        if task:
            config = task["config"]
            from .price_engine import generate_price_profile, generate_solar_profile, generate_consumption_profile
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
            obs = env._make_observation(
                f"Task: {task['name']}. {task['description'][:100]}..."
            )

    return {
        "session_id": session_id,
        "observation": obs.model_dump(),
        "reward": 0.0,
        "done": False,
    }


@app.post("/step/{session_id}")
async def rest_step(session_id: str, action: SolarGridAction):
    env = environments.get(session_id)
    if not env:
        return {"error": "Session not found. Call /reset first."}
    result = env.step(action)
    return result
