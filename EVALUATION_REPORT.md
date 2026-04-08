# Solar Grid Arbitrage Environment — Evaluation Report

**Team Interesting** — Madhuram Agarwal (Lead), Shreyansh Agrawal, Saksham Agrawal

---

## 1. Environment Overview

A **rooftop solar prosumer** in India with a 5kW solar panel and 13.5kWh battery makes **hourly energy trading decisions** on the **IEX Day-Ahead Market (DAM)**.

Each hour the agent observes the current IEX price, a noisy 3-hour price forecast, solar generation, battery level, and household consumption — then picks one of 4 actions:

| Action | What it does |
|---|---|
| **sell** | Sell power to grid at IEX price (from solar surplus or battery) |
| **store** | Charge battery from solar surplus |
| **buy** | Buy grid power to charge battery (arbitrage) |
| **hold** | Self-consume solar, auto-store surplus |

**Episode = 24 steps = 1 full day.** Randomized conditions: season (summer/monsoon/winter), day type (weekday/weekend), initial battery SOC, and price/solar/consumption noise.

---

## 2. Why This Domain

| Factor | Detail |
|---|---|
| **Market scale** | India's IEX Day-Ahead Market processes 20+ GW daily |
| **User base** | 10M+ rooftop solar installations, growing rapidly |
| **Policy relevance** | CERC is rolling out P2P trading regulations for prosumers |
| **Real problem** | This is a live optimization challenge — not a toy or game |

The environment models what millions of Indian households will face as prosumer trading becomes mainstream.

---

## 3. Physics & Constraints

| Parameter | Value | Why it matters |
|---|---|---|
| Battery capacity | 13.5 kWh (Tesla Powerwall class) | Barely covers one evening's consumption (~13 kWh) |
| Charge efficiency | 92% | 8% energy lost when charging |
| Discharge efficiency | 95% | 5% energy lost when discharging |
| Grid transmission loss | 3% | Can't sell 100% of what you generate |
| Max charge/discharge rate | 5 kW | Physical rate limit |
| Solar panel capacity | 5 kW | Typical Indian rooftop system |

**Key tension:** The battery is too small to store everything and sell everything at peak. Agents that dump battery at peak and then pay Rs 7/kWh retail for nighttime consumption lose money. This creates genuine strategic depth.

---

## 4. Price Patterns (Based on Real IEX DAM Data)

| Time Period | Summer (Rs/kWh) | Winter | Monsoon |
|---|---|---|---|
| Night trough (0-5) | 2.2 - 3.0 | 2.5 - 3.5 | 2.8 - 3.5 |
| Morning ramp (6-11) | 3.8 - 5.8 | 3.8 - 5.5 | 4.0 - 5.0 |
| Solar dip midday (12-17) | 3.0 - 4.5 | 3.5 - 5.5 | 3.5 - 5.5 |
| Evening peak (18-21) | 6.0 - 8.2 | 7.0 - 9.0 | 6.5 - 8.0 |

The agent gets a **noisy** 3-hour forecast (+-10%). It does NOT see the full day's prices — it must reason about future prices from partial information.

---

## 5. Observation Space (13 fields)

| Field | Type | Range | Description |
|---|---|---|---|
| `hour` | int | 0-23 | Current hour |
| `current_price` | float | 0.5-10.0 | IEX DAM price Rs/kWh |
| `next_3h_prices` | list[float] | 3 values | Noisy price forecast |
| `solar_generation_kwh` | float | 0-5.0 | Solar output this hour |
| `battery_soc` | float | 0.0-1.0 | Battery state of charge |
| `battery_capacity_kwh` | float | 13.5 | Total battery capacity |
| `energy_consumed_kwh` | float | 0.1-3.5 | Household demand |
| `cumulative_revenue` | float | any | Net Rs earned so far |
| `cumulative_cost` | float | >= 0 | Total Rs spent |
| `hours_remaining` | int | 0-23 | Hours left |
| `day_type` | str | weekday/weekend | Day type |
| `season` | str | summer/monsoon/winter | Season |
| `message` | str | - | Step feedback |

---

## 6. Reward Function Design

**Dense, 4-component, clamped to [-1, 1], every step.**

```
total = profit(50%) + strategy(25%) + efficiency(15%) + penalty(10%)
```

### 6.1 Profit Component (50%)

| Action | Calculation |
|---|---|
| Sell | `revenue = amount x price` (full credit for genuine surplus, adjusted for opportunity cost if battery-sourced) |
| Buy | `-cost + 35% of future arbitrage potential` (looks ahead 8 hours for price spread) |
| Store | `expected_peak x amount x 0.4` when storing free solar; spread-based for other cases |
| Hold | `self_consumed x retail_tariff` (avoided retail purchase = real savings) |

### 6.2 Strategy Component (25%)

| Condition | Signal | Purpose |
|---|---|---|
| Store during solar hours (7-16), battery < 90% | +0.20 | Teach "save for peak" |
| Sell at peak (18-21), price > Rs 5.5 | +0.20 x (price/7) | Reward peak exploitation |
| Sell solar surplus when battery full | +0.12 | Don't waste free energy |
| Sell cheap when peak is coming | -0.12 | Punish impatience |
| Buy at night, monsoon, SOC < 40% | +0.15 | Reward smart arbitrage |
| Buy during solar hours when surplus available | -0.15 | Penalize waste |

### 6.3 Efficiency Component (15%)

| Condition | Signal |
|---|---|
| SOC < 5% | -0.08 (deep discharge) |
| SOC > 95% | -0.02 (overcharge) |
| SOC 20-80% | +0.02 (healthy range) |
| Solar wasted (surplus not stored/sold) | -0.10 to -0.12 |

### 6.4 Penalty Component (10%)

| Condition | Signal |
|---|---|
| Requested amount exceeds actual capacity | -0.05 |
| Action had zero effect | -0.10 |

### 6.5 End-of-Episode Bonus (season-scaled)

| Condition | Bonus |
|---|---|
| Net profit > season "great" threshold | +0.5 |
| Net profit > season "good" threshold | +0.25 |
| Net profit > Rs 0 | +0.1 |
| Lost money | -0.3 |

Season thresholds: Summer great=Rs 35, good=Rs 18 | Monsoon great=Rs 18, good=Rs 10 | Winter great=Rs 22, good=Rs 12

**Why this works:** Most hackathon envs use sparse end-of-episode rewards. Ours gives signal every step — the agent knows immediately whether selling at hour 12 was smart or dumb. The strategy shaping means even a random agent gets pushed toward "store during day, sell at peak."

---

## 7. Three Tasks

### Task 1: Sunny Day Basics (Easy)
- **Scenario:** Summer weekday, high solar output, 5% noise
- **Initial battery:** 50% charged
- **Pass threshold:** Net revenue > Rs 0
- **What it tests:** Can the agent learn the basic solar arbitrage cycle?
- **Optimal play:** Store solar midday, sell surplus when battery full, sell at evening peak

### Task 2: Monsoon Arbitrage (Medium)
- **Scenario:** Monsoon weekday, cloud cover cuts solar ~50%, 10% noise
- **Initial battery:** 80% charged
- **Pass threshold:** Net revenue > Rs -20
- **What it tests:** Can the agent adapt when solar is scarce? Must manage a near-full battery strategically — sell at peak without draining reserves needed for consumption
- **Key challenge:** High initial SOC is a trap: selling too aggressively at peak leaves nothing for nighttime, forcing expensive grid purchases

### Task 3: Winter Peak Maximizer (Hard)
- **Scenario:** Winter weekday, short solar hours, extreme evening peaks (Rs 7-9), 12% noise
- **Initial battery:** 20% charged
- **Pass threshold:** Net revenue > Rs 12
- **What it tests:** Can the agent combine solar storage + precisely timed peak selling?
- **Key challenge:** Low initial battery, short solar window, high consumption. Night buying is a trap in winter — solar surplus alone fills the battery for free

---

## 8. Test Results

### 8.1 Automated Validation

| Check | Result |
|---|---|
| HF Space returns HTTP 200 | PASS |
| Root endpoint returns env info + 3 tasks | PASS |
| `reset()` returns valid observation (13 fields) | PASS |
| `step()` returns reward in [-1,1] and done flag | PASS |
| Full 24-step episode completes correctly | PASS |
| `openenv.yaml` present at root | PASS |
| `Dockerfile` builds and runs | PASS |
| `inference.py` at root, prints [START]/[STEP]/[END] | PASS |
| All grader scores strictly in (0, 1) | PASS |
| inference.py completes in <20 min on 2 vCPU / 8 GB | PASS |
| WebSocket endpoint `/ws` functional | PASS |
| REST endpoints `/reset`, `/step/{id}` functional | PASS |
| 7 unit tests pass | PASS |

### 8.2 Heuristic Baseline Results (V2 Season-Aware Policy)

| Task | Difficulty | Net Revenue | Score | Status |
|---|---|---|---|---|
| Sunny Day Basics | Easy | Rs 48.09 | **0.99** | **PASS** |
| Monsoon Arbitrage | Medium | Rs -9.41 | **0.80** | **PASS** |
| Winter Peak Maximizer | Hard | Rs 21.46 | **0.99** | **PASS** |

**All 3 tasks pass with the heuristic baseline.**

### 8.3 LLM Agent Results (Gemini 2.5 Flash)

The LLM receives each observation as a text prompt with strategy tips and returns JSON action decisions.

| Task | Difficulty | Net Revenue | Score | Status |
|---|---|---|---|---|
| Sunny Day Basics | Easy | Rs 59.75 | **1.00** | **PASS** |
| Monsoon Arbitrage | Medium | Rs -18.07 | **0.00** | **FAIL** |
| Winter Peak Maximizer | Hard | Rs 23.21 | **1.00** | **PASS** |

### 8.4 Analysis

| Observation | Insight |
|---|---|
| **Easy task trivial** | Both heuristic and LLM achieve perfect scores — validates the baseline difficulty |
| **Medium task differentiates agents** | Heuristic passes (0.80), LLM fails (0.00) — the LLM over-sells at peak, draining reserves needed for nighttime consumption. This proves the environment rewards *careful* trading, not just aggressive selling |
| **Hard task rewards planning** | Both pass, but require precise solar storage timing + peak selling coordination |
| **Clear seasonality** | Different strategies work in different seasons. Summer = store and sell surplus. Monsoon = manage high initial SOC carefully. Winter = store all solar, sell at extreme peaks. This proves the environment effectively measures adaptation |

---

## 9. Sample Agent Trace (Heuristic, Easy Task)

```
Hour  0: hold    — night, no solar, conserve battery (SOC 47%)
Hour  1: hold    — night, minimal consumption
Hour  5: hold    — sunrise, tiny solar starts
Hour  6: hold    — solar just covers consumption
Hour  7: store 0.7 — solar surplus begins, charge battery
Hour  8: store 2.1 — strong solar, battery filling (SOC 56%)
Hour  9: store 3.1 — peak solar hours (SOC 77%)
Hour 10: store 4.1 — battery nearly full (SOC 100%)
Hour 11: sell 4.2 — battery full! Sell surplus at Rs 4.23
Hour 12: sell 4.8 — continue selling free solar surplus
Hour 13: sell 4.0 — afternoon surplus
Hour 17: hold    — transition to evening, self-consume
Hour 18: sell 5.0 — PEAK! Rs 5.31, sell from battery
Hour 19: sell 1.8 — Rs 8.00! Peak of peak, drain remaining battery
Hour 20: hold    — battery empty, ride out the night
Result: Rs 48.09 revenue, Score 1.00 PASS
```

---

## 10. What Makes This Environment Good for LLM Agents

1. **Rich text observations** — Every field is named and described, making it natural for LLMs to reason about
2. **Dense reward signal** — Feedback every step, not just end-of-episode
3. **Requires reasoning, not memorization** — Noisy prices + seasonal variation mean no fixed strategy works everywhere
4. **Real strategic depth** — The battery capacity constraint forces genuine trade-offs (sell vs. save for self-consumption)
5. **Difficulty progression** — Easy task validates basic competence, hard task requires multi-step planning
6. **Typed API** — Pydantic models make the action/observation contract unambiguous

---

## 11. Deployment & Access

| Resource | URL |
|---|---|
| HuggingFace Space | https://huggingface.co/spaces/shipperasap/solar-grid-env |
| GitHub Repository | https://github.com/shipperasap/solar-grid-env |
| Interactive API Docs | https://shipperasap-solar-grid-env.hf.space/docs |

### Quick Start
```bash
# Clone and install
git clone https://github.com/shipperasap/solar-grid-env.git
cd solar-grid-env
pip install -r server/requirements.txt

# Run tests
python -m server.test_environment

# Run heuristic baseline
python -m server.inference

# Run with LLM agent
API_BASE_URL="https://generativelanguage.googleapis.com/v1beta/openai/" \
MODEL_NAME="gemini-2.5-flash" \
HF_TOKEN="your_api_key" \
python inference.py

# Start server
uvicorn server.server:app --host 0.0.0.0 --port 7860

# Docker
docker build -t solar-grid-env .
docker run -p 7860:7860 solar-grid-env
```

---

## 12. Project Structure

```
solar_grid_env/
├── Dockerfile              # HF Spaces deployment
├── inference.py            # OpenEnv entry point (LLM + heuristic fallback)
├── client.py               # Typed REST + WebSocket client
├── openenv.yaml            # OpenEnv spec metadata
├── pyproject.toml          # Python package config
├── README.md               # Full documentation
├── EVALUATION_REPORT.md    # This document
├── validate-submission.sh  # Submission validation script
├── .gitignore
└── server/
    ├── __init__.py
    ├── models.py           # Pydantic: SolarGridAction, Observation, State
    ├── price_engine.py     # IEX DAM price profiles (3 seasons x 2 day types)
    ├── reward.py           # 4-component dense reward function
    ├── environment.py      # Core: reset(), step(), 24-hour episodes
    ├── tasks.py            # 3 tasks + grading rubrics
    ├── server.py           # FastAPI + WebSocket server
    ├── inference.py        # Heuristic policy (season-aware V2)
    ├── test_environment.py # 7 unit tests
    └── requirements.txt
```

---

## 13. Known Limitations & Future Work

| Limitation | Potential Improvement |
|---|---|
| Monsoon task is hardest for LLMs — high initial SOC creates a non-obvious trap | Could add hint in observation message about SOC management |
| Price profiles are template-based with Gaussian noise, not historical data | Could integrate real IEX API data for more realistic variation |
| No multi-day episodes | Extending to week-long trading would test long-horizon planning |
| Single prosumer only | Multi-agent P2P trading would model CERC regulations more fully |

---

*Built by Team Interesting for the OpenEnv Hackathon, April 2026.*
