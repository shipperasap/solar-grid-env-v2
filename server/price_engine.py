import random
import math
from typing import List

# === IEX DAM PRICE PROFILES (Rs/kWh) ===
# Based on real IEX Day-Ahead Market clearing price patterns

SUMMER_WEEKDAY_PRICES = [
    2.8, 2.5, 2.3, 2.2, 2.5, 3.0,     # 00-05: night trough
    3.8, 4.5, 5.2, 5.8, 5.0, 4.2,     # 06-11: morning ramp + peak
    3.5, 3.2, 3.0, 3.3, 3.8, 4.5,     # 12-17: solar dip midday
    6.0, 7.5, 8.2, 7.0, 5.5, 4.0,     # 18-23: evening SUPER peak
]

SUMMER_WEEKEND_PRICES = [
    2.5, 2.3, 2.0, 2.0, 2.2, 2.5,     # lower night
    3.2, 3.8, 4.2, 4.5, 4.0, 3.5,     # moderate morning
    3.0, 2.8, 2.5, 2.8, 3.2, 4.0,     # deeper solar dip
    5.5, 6.8, 7.0, 6.0, 4.5, 3.5,     # lower evening peak
]

WINTER_WEEKDAY_PRICES = [
    3.0, 2.8, 2.5, 2.5, 2.8, 3.5,     # slightly higher night (heating)
    4.5, 5.5, 5.0, 4.5, 4.0, 3.8,     # morning peak (heating)
    3.5, 3.5, 3.5, 3.8, 4.5, 5.5,     # less solar dip in winter
    7.0, 8.5, 9.0, 7.5, 6.0, 4.5,     # higher evening peak (early dark + heating)
]

MONSOON_WEEKDAY_PRICES = [
    3.2, 3.0, 2.8, 2.8, 3.0, 3.5,     # monsoon base higher
    4.0, 4.8, 5.0, 4.5, 4.2, 4.0,     # moderate morning
    4.0, 3.8, 3.5, 3.8, 4.5, 5.5,     # less solar -> higher midday
    6.5, 7.8, 8.0, 7.0, 5.5, 4.0,     # evening peak
]

# Solar generation profiles (fraction of panel capacity)
SUMMER_SOLAR = [
    0.0, 0.0, 0.0, 0.0, 0.0, 0.05,    # sunrise ~5:30
    0.15, 0.35, 0.55, 0.75, 0.90, 0.95, # ramp up
    1.0, 0.95, 0.85, 0.70, 0.50, 0.25, # afternoon decline
    0.05, 0.0, 0.0, 0.0, 0.0, 0.0,     # sunset ~18:30
]

WINTER_SOLAR = [
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,     # sunrise ~6:30
    0.05, 0.20, 0.40, 0.60, 0.75, 0.80, # slower ramp
    0.80, 0.75, 0.60, 0.40, 0.20, 0.05, # earlier decline
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,      # sunset ~17:30
]

MONSOON_SOLAR = [
    0.0, 0.0, 0.0, 0.0, 0.0, 0.02,
    0.08, 0.18, 0.30, 0.40, 0.45, 0.50, # cloud cover reduces output
    0.55, 0.50, 0.40, 0.30, 0.18, 0.08,
    0.02, 0.0, 0.0, 0.0, 0.0, 0.0,
]

# Household consumption profile (kWh) - Indian middle class home
WEEKDAY_CONSUMPTION = [
    0.3, 0.2, 0.2, 0.2, 0.2, 0.3,     # night: minimal
    0.5, 0.8, 0.6, 0.4, 0.4, 0.5,     # morning routine
    0.5, 0.6, 0.8, 1.0, 1.2, 1.5,     # afternoon (AC in summer)
    2.0, 2.5, 2.0, 1.5, 1.0, 0.5,     # evening peak (cooking, TV, AC)
]

WEEKEND_CONSUMPTION = [
    0.3, 0.2, 0.2, 0.2, 0.3, 0.3,
    0.5, 0.8, 1.0, 1.2, 1.5, 1.5,     # home all day
    1.5, 1.5, 1.2, 1.0, 1.2, 1.5,
    2.0, 2.5, 2.2, 1.8, 1.2, 0.5,
]


def generate_price_profile(season: str, day_type: str, noise_pct: float = 0.1) -> List[float]:
    """Generate a 24-hour price profile with realistic noise."""
    if season == "summer":
        base = SUMMER_WEEKEND_PRICES if day_type == "weekend" else SUMMER_WEEKDAY_PRICES
    elif season == "winter":
        base = WINTER_WEEKDAY_PRICES
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
