---
title: Solar Grid Env
emoji: 🔥
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# Solar Grid Arbitrage — Indian Electricity Market

An RL environment simulating a rooftop solar prosumer with battery storage making hourly energy trading decisions on India's **IEX Day-Ahead Market (DAM)**. The agent decides each hour whether to sell surplus power to the grid, store it in a battery, buy cheap grid power, or self-consume — maximizing daily revenue using real IEX price patterns.

## Domain Background

India's **Indian Energy Exchange (IEX)** operates the Day-Ahead Market where electricity is traded in hourly blocks. With India's push toward "One Nation One Grid" and upcoming P2P trading regulations from CERC, rooftop solar prosumers will increasingly participate in electricity markets. This environment models that future — a household with a 5kW rooftop solar system and 13.5kWh battery (Tesla Powerwall class) trading on IEX DAM.

Key dynamics:
- **Solar glut at midday** drives IEX prices down (Rs 2.5-3.5/kWh)
- **Evening peak (18:00-21:00)** drives prices to Rs 6-9/kWh
- **Night trough (00:00-05:00)** offers cheap buying opportunity
- Battery efficiency losses (8% charge, 5% discharge, 3% grid transmission)

## State Space (Observation)

| Field | Type | Range | Description |
|---|---|---|---|
| `hour` | int | 0-23 | Current hour of day |
| `current_price` | float | 0.5-10.0 | IEX DAM price Rs/kWh |
| `next_3h_prices` | list[float] | 3 values | Noisy forecast for next 3 hours |
| `solar_generation_kwh` | float | 0-5.0 | Solar output this hour |
| `battery_soc` | float | 0.0-1.0 | Battery state of charge |
| `energy_consumed_kwh` | float | 0.1-3.5 | Household consumption |
| `cumulative_revenue` | float | any | Net Rs earned so far |
| `cumulative_cost` | float | >= 0 | Total Rs spent |
| `hours_remaining` | int | 0-23 | Hours left in episode |
| `day_type` | str | weekday/weekend | Day type |
| `season` | str | summer/monsoon/winter | Season |
| `message` | str | - | Environment feedback |

## Action Space

| Action | Description |
|---|---|
| `sell` | Sell solar surplus + battery power to grid at IEX price |
| `store` | Store solar surplus into battery |
| `buy` | Buy grid power into battery (arbitrage) |
| `hold` | Self-consume solar, auto-store surplus |

Each action includes `amount_kwh` (0-10 kWh), clamped to physical constraints.

## Reward Function

4-component dense reward at every step, clamped to [-1, 1]:

| Component | Weight | Description |
|---|---|---|
| Profit | 50% | Immediate Rs gain/loss from action |
| Strategy | 25% | Shaping for peak-hour selling, trough buying |
| Efficiency | 15% | Battery health, solar utilization |
| Penalty | 10% | Constraint violations (impossible actions) |

End-of-episode bonus based on daily net profit.

## Episode Structure

- **24 steps** = 1 full day (hour 0 to hour 23)
- Randomized: season, day type, initial battery SOC, price/solar/consumption noise

## Tasks

### 1. Sunny Day Basics (Easy)
Summer weekday, high solar, real IEX prices. Goal: net revenue > Rs -5. Store solar midday when prices crash, sell at evening peak, conserve battery for expensive nights.
**Baseline score: PASS**

### 2. Monsoon Arbitrage (Medium)
Monsoon weekday, reduced solar. Goal: net revenue > Rs -20 via buy-low-sell-high. Less solar means less surplus to sell.
**Baseline score: challenging (room for LLM agent improvement)**

### 3. Weekend Summer Surplus (Medium)
Summer weekend, family home all day, real IEX weekend DAM prices. Goal: net revenue > Rs -15. Lower and less peaky prices than weekdays — requires adapting strategy to weekend dynamics.
**Baseline score: PASS**

### 4. Winter Peak Maximizer (Hard)
Winter weekday, short solar hours, extreme evening peaks (Rs 7-9/kWh). Goal: net revenue > Rs 12. Requires perfectly timed charge/discharge.
**Baseline score: 1.00 (PASS with V2 heuristic)**

## Physics Constraints

- Battery charge efficiency: 92%
- Battery discharge efficiency: 95%
- Grid sell efficiency: 97% (3% transmission loss)
- Max charge/discharge rate: 5 kW
- Battery capacity: 13.5 kWh

## Installation & Usage

### Local
```bash
pip install -r server/requirements.txt
uvicorn server.server:app --host 0.0.0.0 --port 7860
```

### Docker
```bash
docker build -t solar-grid-env .
docker run -p 7860:7860 solar-grid-env
```

### Run Inference (Heuristic Baseline)
```bash
python -m server.inference
```

### Run Inference (LLM Agent)
```bash
API_BASE_URL="https://api.openai.com/v1" MODEL_NAME="gpt-4o-mini" HF_TOKEN="your-key" python inference.py
```

### Run Tests
```bash
python -m server.test_environment
```

### WebSocket Example
```python
import asyncio, json, websockets

async def test():
    async with websockets.connect('ws://localhost:7860/ws') as ws:
        await ws.send(json.dumps({'action': 'reset', 'task_id': 'solar_grid_easy'}))
        obs = json.loads(await ws.recv())
        print(obs)

        for _ in range(24):
            await ws.send(json.dumps({
                'action': 'step',
                'task_id': 'solar_grid_easy',
                'data': {'action_type': 'hold', 'amount_kwh': 1.0}
            }))
            result = json.loads(await ws.recv())
            print(f"Hour {result['observation']['hour']}: reward={result['reward']:.3f}")

asyncio.run(test())
```

## Price Data Source

Summer price profiles are derived from **real IEX India Day-Ahead Market clearing prices** (MCP data from April 1-8, 2026). Winter and monsoon profiles are template-based, calibrated to real IEX price magnitudes.

## Team

**Team Interesting** — Madhuram Agarwal (Lead), Shreyansh Agrawal, Saksham Agrawal

## License

MIT
