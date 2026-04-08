import random
import math
from typing import List

# === IEX DAM PRICE PROFILES (Rs/kWh) ===
# Derived from actual IEX Day-Ahead Market clearing prices (MCP)
# Source: IEX DAM Market Snapshot, 01-Apr-2026 to 08-Apr-2026
# Original data in Rs/MWh (15-min blocks), averaged to hourly and converted to Rs/kWh

# --- Real historical daily profiles (summer, April 2026) ---
# Each sub-list is one real day, hours 0-23

IEX_SUMMER_WEEKDAY_DAYS = [
    # 01-Apr-2026 (Wednesday)
    [6.0, 5.33, 4.77, 4.54, 5.02, 8.12, 7.62, 4.33, 2.89, 2.1, 1.73, 1.36, 1.0, 0.98, 1.7, 2.41, 3.06, 3.62, 6.54, 10.0, 9.75, 9.87, 10.0, 10.0],
    # 02-Apr-2026 (Thursday)
    [9.0, 5.56, 4.85, 4.56, 5.47, 10.0, 7.84, 4.87, 3.36, 2.86, 2.38, 2.5, 2.34, 2.11, 2.68, 3.08, 3.43, 4.05, 6.76, 10.0, 8.12, 8.69, 10.0, 10.0],
    # 03-Apr-2026 (Friday)
    [5.29, 4.7, 4.65, 4.56, 4.5, 6.29, 9.96, 5.57, 2.9, 2.28, 1.96, 1.91, 1.71, 1.55, 1.93, 2.52, 2.99, 3.5, 4.55, 9.01, 5.53, 5.47, 7.53, 6.05],
    # 06-Apr-2026 (Monday)
    [4.39, 4.17, 3.92, 3.85, 3.9, 4.16, 5.13, 3.81, 2.17, 1.31, 0.98, 0.98, 0.98, 0.98, 1.26, 1.63, 2.3, 2.92, 4.32, 5.25, 4.23, 3.9, 3.86, 4.18],
    # 07-Apr-2026 (Tuesday)
    [3.52, 3.49, 3.5, 3.49, 3.61, 3.7, 4.23, 3.38, 2.16, 1.45, 0.74, 0.86, 0.98, 0.89, 0.99, 1.7, 2.08, 2.33, 3.63, 4.39, 3.7, 3.55, 3.32, 3.39],
    # 08-Apr-2026 (Wednesday)
    [3.2, 3.14, 3.35, 3.4, 3.49, 3.45, 4.02, 2.94, 1.49, 1.41, 0.96, 0.98, 0.98, 0.73, 0.98, 1.57, 1.99, 2.28, 3.83, 4.4, 3.85, 3.59, 3.59, 3.85],
]

IEX_SUMMER_WEEKEND_DAYS = [
    # 04-Apr-2026 (Saturday)
    [9.26, 5.91, 4.88, 4.49, 4.39, 5.83, 5.81, 4.44, 2.73, 2.2, 1.94, 1.65, 1.55, 1.55, 1.68, 2.13, 2.86, 3.59, 7.33, 10.0, 5.72, 5.01, 5.44, 6.82],
    # 05-Apr-2026 (Sunday)
    [4.89, 4.63, 4.5, 4.31, 4.25, 4.78, 4.89, 3.67, 1.95, 1.26, 0.53, 0.75, 0.98, 0.98, 1.24, 1.64, 2.25, 3.06, 4.57, 5.04, 4.49, 4.39, 4.16, 4.48],
]

# --- Averaged profiles (used as base for noise) ---

SUMMER_WEEKDAY_PRICES = [
    5.23, 4.40, 4.17, 4.07, 4.33, 5.95,     # 00-05: night (higher than templates — real IEX)
    6.47, 4.15, 2.50, 1.90, 1.46, 1.43,     # 06-11: morning ramp then solar crash
    1.33, 1.21, 1.59, 2.15, 2.64, 3.12,     # 12-17: deep solar dip (real RE impact)
    4.94, 7.17, 5.86, 5.84, 6.38, 6.25,     # 18-23: evening peak
]

SUMMER_WEEKEND_PRICES = [
    7.07, 5.27, 4.69, 4.40, 4.32, 5.30,     # 00-05: higher night
    5.35, 4.05, 2.34, 1.73, 1.23, 1.20,     # 06-11: morning then solar crash
    1.27, 1.27, 1.46, 1.88, 2.55, 3.33,     # 12-17: deep solar dip
    5.95, 7.52, 5.11, 4.70, 4.80, 5.65,     # 18-23: evening peak (lower than weekday)
]

# Winter and monsoon — no real data available, template-based
# Scaled to be consistent with real summer data magnitudes

WINTER_WEEKDAY_PRICES = [
    3.0, 2.8, 2.5, 2.5, 2.8, 3.5,           # slightly higher night (heating)
    4.5, 5.5, 5.0, 4.5, 4.0, 3.8,           # morning peak (heating)
    3.5, 3.5, 3.5, 3.8, 4.5, 5.5,           # less solar dip in winter
    7.0, 8.5, 9.0, 7.5, 6.0, 4.5,           # higher evening peak (early dark + heating)
]

MONSOON_WEEKDAY_PRICES = [
    3.2, 3.0, 2.8, 2.8, 3.0, 3.5,           # monsoon base higher
    4.0, 4.8, 5.0, 4.5, 4.2, 4.0,           # moderate morning
    4.0, 3.8, 3.5, 3.8, 4.5, 5.5,           # less solar -> higher midday
    6.5, 7.8, 8.0, 7.0, 5.5, 4.0,           # evening peak
]

# Solar generation profiles (fraction of panel capacity)
SUMMER_SOLAR = [
    0.0, 0.0, 0.0, 0.0, 0.0, 0.05,          # sunrise ~5:30
    0.15, 0.35, 0.55, 0.75, 0.90, 0.95,     # ramp up
    1.0, 0.95, 0.85, 0.70, 0.50, 0.25,      # afternoon decline
    0.05, 0.0, 0.0, 0.0, 0.0, 0.0,          # sunset ~18:30
]

WINTER_SOLAR = [
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,           # sunrise ~6:30
    0.05, 0.20, 0.40, 0.60, 0.75, 0.80,     # slower ramp
    0.80, 0.75, 0.60, 0.40, 0.20, 0.05,     # earlier decline
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,           # sunset ~17:30
]

MONSOON_SOLAR = [
    0.0, 0.0, 0.0, 0.0, 0.0, 0.02,
    0.08, 0.18, 0.30, 0.40, 0.45, 0.50,     # cloud cover reduces output
    0.55, 0.50, 0.40, 0.30, 0.18, 0.08,
    0.02, 0.0, 0.0, 0.0, 0.0, 0.0,
]

# Household consumption profile (kWh) - Indian middle class home
WEEKDAY_CONSUMPTION = [
    0.3, 0.2, 0.2, 0.2, 0.2, 0.3,           # night: minimal
    0.5, 0.8, 0.6, 0.4, 0.4, 0.5,           # morning routine
    0.5, 0.6, 0.8, 1.0, 1.2, 1.5,           # afternoon (AC in summer)
    2.0, 2.5, 2.0, 1.5, 1.0, 0.5,           # evening peak (cooking, TV, AC)
]

WEEKEND_CONSUMPTION = [
    0.3, 0.2, 0.2, 0.2, 0.3, 0.3,
    0.5, 0.8, 1.0, 1.2, 1.5, 1.5,           # home all day
    1.5, 1.5, 1.2, 1.0, 1.2, 1.5,
    2.0, 2.5, 2.2, 1.8, 1.2, 0.5,
]


def generate_price_profile(season: str, day_type: str, noise_pct: float = 0.1, adversarial_mode: bool = False) -> List[float]:
    """Generate a 24-hour price profile.

    For summer: randomly picks a real historical IEX day and adds noise.
    For winter/monsoon: uses template profiles with noise (no real data available).
    When adversarial_mode is True for winter, adds a deceptive price spike at hour 15
    to punish agents who sell too early instead of waiting for evening peak.
    """
    if season == "summer":
        if day_type == "weekend":
            # Pick a random real weekend day
            base = random.choice(IEX_SUMMER_WEEKEND_DAYS)
        else:
            # Pick a random real weekday
            base = random.choice(IEX_SUMMER_WEEKDAY_DAYS)
    elif season == "winter":
        base = WINTER_WEEKDAY_PRICES.copy()
        # Adversarial: add deceptive spike at hour 15 that lures early sellers
        if adversarial_mode:
            base[15] = 7.5  # Fake spike - looks like peak but evening will be higher (8.5-9)
    else:
        base = MONSOON_WEEKDAY_PRICES

    profile = []
    for p in base:
        noise = random.gauss(0, p * noise_pct)
        profile.append(max(0.5, round(p + noise, 2)))
    return profile


def generate_solar_profile(season: str, panel_kw: float, noise_pct: float = 0.08) -> List[float]:
    """Generate hourly solar output in kWh."""
    if season == "summer":
        base = SUMMER_SOLAR
    elif season == "winter":
        base = WINTER_SOLAR
    else:
        base = MONSOON_SOLAR

    profile = []
    for frac in base:
        output = frac * panel_kw
        noise = random.gauss(0, output * noise_pct) if output > 0 else 0
        profile.append(max(0.0, round(output + noise, 2)))
    return profile


def generate_consumption_profile(day_type: str, season: str, noise_pct: float = 0.15) -> List[float]:
    """Generate hourly household consumption in kWh."""
    base = WEEKEND_CONSUMPTION if day_type == "weekend" else WEEKDAY_CONSUMPTION

    season_mult = {"summer": 1.4, "monsoon": 1.1, "winter": 1.2}
    mult = season_mult.get(season, 1.0)

    profile = []
    for c in base:
        consumption = c * mult
        noise = random.gauss(0, consumption * noise_pct)
        profile.append(max(0.1, round(consumption + noise, 2)))
    return profile


def get_retail_tariff(hour: int, season: str) -> float:
    """Get the retail electricity tariff (Rs/kWh) for self-consumption value."""
    if 22 <= hour or hour < 6:
        base = 4.5   # off-peak
    elif 6 <= hour < 10 or 17 <= hour < 22:
        base = 7.0   # peak
    else:
        base = 5.5   # mid-peak

    season_adj = {"summer": 1.1, "monsoon": 1.0, "winter": 0.95}
    return round(base * season_adj.get(season, 1.0), 2)
