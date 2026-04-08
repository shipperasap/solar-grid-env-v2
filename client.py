"""
Typed client for Solar Grid Arbitrage Environment.

Usage:
    from client import SolarGridClient, SolarGridAction, ActionType

    client = SolarGridClient("https://shipperasap-solar-grid-env.hf.space")
    obs = client.reset()
    result = client.step(SolarGridAction(action_type=ActionType.HOLD, amount_kwh=1.0))
"""

import json
import requests
import asyncio
from typing import Optional, Dict, Any

try:
    import websockets
except ImportError:
    websockets = None

from server.models import SolarGridAction, SolarGridObservation, SolarGridState, ActionType


class SolarGridClient:
    """REST client for the Solar Grid Arbitrage Environment."""

    def __init__(self, base_url: str = "http://localhost:7860"):
        self.base_url = base_url.rstrip("/")
        self.session_id: Optional[str] = None

    def info(self) -> Dict[str, Any]:
        """Get environment info."""
        r = requests.get(f"{self.base_url}/")
        r.raise_for_status()
        return r.json()

    def tasks(self) -> list:
        """List available tasks."""
        r = requests.get(f"{self.base_url}/tasks")
        r.raise_for_status()
        return r.json()["tasks"]

    def reset(self, task_id: Optional[str] = None) -> SolarGridObservation:
        """Reset environment, optionally with a specific task."""
        params = {}
        if task_id:
            params["task_id"] = task_id
        r = requests.post(f"{self.base_url}/reset", params=params)
        r.raise_for_status()
        data = r.json()
        self.session_id = data["session_id"]
        return SolarGridObservation(**data["observation"])

    def step(self, action: SolarGridAction) -> Dict[str, Any]:
        """Take a step in the environment."""
        if not self.session_id:
            raise RuntimeError("Call reset() first.")
        r = requests.post(
            f"{self.base_url}/step/{self.session_id}",
            json=action.model_dump(),
        )
        r.raise_for_status()
        data = r.json()
        return {
            "observation": SolarGridObservation(**data["observation"]),
            "reward": data["reward"],
            "done": data["done"],
            "state": data.get("state", {}),
            "reward_breakdown": data.get("reward_breakdown", {}),
            "grade": data.get("grade"),
        }


class SolarGridWebSocketClient:
    """WebSocket client for the Solar Grid Arbitrage Environment."""

    def __init__(self, base_url: str = "ws://localhost:7860"):
        if websockets is None:
            raise ImportError("Install websockets: pip install websockets")
        self.ws_url = base_url.rstrip("/").replace("https://", "wss://").replace("http://", "ws://") + "/ws"
        self.ws = None

    async def connect(self):
        self.ws = await websockets.connect(self.ws_url)

    async def reset(self, task_id: Optional[str] = None) -> Dict[str, Any]:
        msg = {"action": "reset"}
        if task_id:
            msg["task_id"] = task_id
        await self.ws.send(json.dumps(msg))
        return json.loads(await self.ws.recv())

    async def step(self, action: SolarGridAction, task_id: Optional[str] = None) -> Dict[str, Any]:
        msg = {"action": "step", "data": action.model_dump()}
        if task_id:
            msg["task_id"] = task_id
        await self.ws.send(json.dumps(msg))
        return json.loads(await self.ws.recv())

    async def state(self) -> Dict[str, Any]:
        await self.ws.send(json.dumps({"action": "state"}))
        return json.loads(await self.ws.recv())

    async def close(self):
        if self.ws:
            await self.ws.close()


# Convenience exports
__all__ = [
    "SolarGridClient",
    "SolarGridWebSocketClient",
    "SolarGridAction",
    "SolarGridObservation",
    "SolarGridState",
    "ActionType",
]
