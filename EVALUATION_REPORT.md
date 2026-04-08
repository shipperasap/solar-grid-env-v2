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

## 4. Price Patterns (Real IEX DAM Data for Summer, Templates for Winter/Monsoon)

Summer prices are derived from **actual IEX DAM MCP data** (April 1-8, 2026), covering 6 weekdays and 2 weekend days. Each episode randomly picks one of these real historical days and applies noise.

| Time Period | Summer Weekday (Rs/kWh) | Summer Weekend | Winter | Monsoon |
|---|---|---|---|---|
| Night (0-5) | 3.2 - 10.0 | 4.3 - 9.3 | 2.5 - 3.5 | 2.8 - 3.5 |
| Morning ramp (6-11) | 0.7 - 10.0 | 0.5 - 5.8 | 3.8 - 5.5 | 4.0 - 5.0 |
| Solar dip midday (12-17) | 0.7 - 4.1 | 1.0 - 3.6 | 3.5 - 5.5 | 3.5 - 5.5 |
| Evening peak (18-21) | 3.6 - 10.0 | 4.5 - 10.0 | 7.0 - 9.0 | 6.5 - 8.0 |

**Key real-data insights:** Solar RE penetration drives midday prices to ~Rs 1/kWh (much lower than templates). Night prices are surprisingly high (Rs 3-9). The biggest arbitrage opportunity is buy-at-midday, sell-at-evening.

The agent gets a **noisy** 3-hour forecast (+-10%). It does NOT see the full day's prices — it must reason about future prices from partial information.

---

## 5. Observation Space (12 fields)

| Field | Type | Range | Description |
|---|---|---|---|
| `hour` | int | 0-23 | Current hour |
| `current_price` | float | 0.5-10.0 | IEX DAM price Rs/kWh |
| `next_3h_prices` | list[float] | 3 values | Noisy price forecast |
| `solar_generation_kwh` | float | 0-5.0 | Solar output this hour |
| `battery_soc` | float | 0.0-1.0 | Battery state of charge |
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
| Sell at peak (18-21), price > Rs 4.5 | +0.20 x (price/6) | Reward peak exploitation |
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

## 7. Four Tasks

### Task 1: Sunny Day Basics (Easy)
- **Scenario:** Summer weekday, real IEX prices, 5% noise
- **Initial battery:** 50% charged
- **Pass threshold:** Net revenue > Rs -5 (lowered from Rs 0 because real IEX data has expensive nights at Rs 3-9/kWh — household consumption costs are unavoidable and eat into revenue. Rs -5 still requires smart trading; a hold-only policy scores ~Rs -15)
- **What it tests:** Can the agent learn the basic solar arbitrage cycle?
- **Optimal play:** Store/buy at midday solar glut (~Rs 1/kWh), sell at evening peak, conserve battery for expensive nights

### Task 2: Monsoon Arbitrage (Medium)
- **Scenario:** Monsoon weekday, cloud cover cuts solar ~50%, 10% noise
- **Initial battery:** 80% charged
- **Pass threshold:** Net revenue > Rs -20
- **What it tests:** Can the agent adapt when solar is scarce? Must manage a near-full battery strategically — sell at peak without draining reserves needed for consumption
- **Key challenge:** High initial SOC is a trap: selling too aggressively at peak leaves nothing for nighttime, forcing expensive grid purchases

### Task 3: Weekend Summer Surplus (Medium)
- **Scenario:** Summer weekend, real IEX weekend DAM prices, 8% noise
- **Initial battery:** 50% charged
- **Pass threshold:** Net revenue > Rs -15
- **What it tests:** Can the agent adapt to weekend price dynamics? Lower and less peaky prices, higher daytime consumption (family home all day)
- **Key challenge:** Weekend evening peaks are weaker (~Rs 5-7.5 vs weekday Rs 7-10), so the agent must be more selective about when to sell

### Task 4: Winter Peak Maximizer (Hard)
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
| Root endpoint returns env info + 4 tasks | PASS |
| `reset()` returns valid observation (12 fields) | PASS |
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

### 8.2 Heuristic Baseline Results (V2 Season-Aware Policy, Real IEX Prices)

| Task | Difficulty | Net Revenue | Score | Status |
|---|---|---|---|---|
| Sunny Day Basics | Easy | Rs 4.02 | **0.75** | **PASS** |
| Monsoon Arbitrage | Medium | Rs -5.21 | **0.80** | **PASS** |
| Weekend Summer Surplus | Medium | Rs -5.14 | **0.95** | **PASS** |
| Winter Peak Maximizer | Hard | Rs 26.59 | **0.99** | **PASS** |

**All 4 tasks pass with the heuristic baseline.**

Note: Summer tasks show lower absolute revenue than V1 templates because real IEX night prices (Rs 3-9/kWh) make nighttime household consumption expensive. This is realistic — the environment now accurately reflects the cost structure prosumers face.

### 8.3 Analysis

| Observation | Insight |
|---|---|
| **Easy task requires real strategy** | Real IEX data eliminated the "trivial" easy task — even the heuristic earns only Rs 4 because night consumption costs are real. A hold-only policy loses ~Rs 15. This better differentiates agents |
| **Monsoon remains the LLM trap** | High initial SOC (80%) tempts aggressive selling, but draining reserves for nighttime forces expensive grid purchases. Heuristic passes (0.80) by conserving battery |
| **Weekend adds new dynamics** | Lower weekend peaks (Rs 5-7.5 vs weekday Rs 7-10) and higher daytime consumption change the optimal strategy. The agent must be more conservative about selling |
| **Winter has the highest ceiling** | Rs 26+ possible because evening peaks hit Rs 7-9 while solar is free. Requires precise timing but rewards it heavily |
| **Real price data matters** | The biggest change from V1: midday solar glut crashes prices to ~Rs 1/kWh, making midday-to-evening arbitrage the dominant strategy (not the textbook night-to-peak play) |

---

## 9. Sample Agent Trace (Heuristic, Easy Task — Real IEX Prices)

```
Hour  0: hold    — night Rs 3.39/kWh (expensive!), conserve battery (SOC 46%)
Hour  5: hold    — sunrise, tiny solar starts
Hour  7: store 0.7 — solar surplus begins, charge battery
Hour  8: store 2.0 — prices dropping to Rs 2.17 as solar floods grid
Hour  9: buy  5.0 — solar glut! Price crashed to Rs 1.51, fill battery (SOC 90%)
Hour 10: store 3.7 — Rs 0.77/kWh, cheapest hour, top off battery (SOC 100%)
Hour 11: sell 4.6 — battery full, sell surplus at Rs 0.85 (low but free solar)
Hour 12-16: sell  — continue selling free solar surplus at Rs 0.85-2.10
Hour 17: hold    — self-consume, save battery for evening
Hour 18: hold    — evening, battery powering consumption (SOC 71%)
Hour 19: sell 1.7 — Rs 4.60 peak, sell above reserve
Hour 20-22: hold — battery powers night consumption, avoids Rs 3.3-3.6 grid
Result: Rs 4.02 net revenue, Score 0.75 PASS
```

**Key insight:** Real IEX data completely changes the strategy. Night buying is a trap (Rs 3-9/kWh). The winning play is midday buying/storing when solar crashes prices to ~Rs 1/kWh, then selling at evening peak while keeping enough battery to avoid expensive night grid purchases.

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
| GitHub Repository | https://github.com/shipperasap/solar-grid-env-v2 |

### Quick Start
```bash
# Clone and install
git clone https://github.com/shipperasap/solar-grid-env-v2.git
cd solar-grid-env-v2
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
    ├── price_engine.py     # Real IEX DAM prices (summer) + template profiles (winter/monsoon)
    ├── reward.py           # 4-component dense reward function
    ├── environment.py      # Core: reset(), step(), 24-hour episodes
    ├── tasks.py            # 4 tasks + grading rubrics
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
| Winter/monsoon price profiles are template-based (no real data available yet) | Could integrate real IEX API data for winter/monsoon seasons |
| No multi-day episodes | Extending to week-long trading would test long-horizon planning |
| Single prosumer only | Multi-agent P2P trading would model CERC regulations more fully |

---

*Built by Team Interesting for the OpenEnv Hackathon, April 2026.*
