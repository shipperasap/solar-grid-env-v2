from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum

class ActionType(str, Enum):
    SELL = "sell"           # sell solar/battery power to grid
    STORE = "store"         # store solar power in battery
    BUY = "buy"            # buy from grid (charge battery or use)
    HOLD = "hold"          # do nothing (use solar for self-consumption)

class SolarGridAction(BaseModel):
    """Agent's hourly decision."""
    action_type: ActionType = Field(..., description="What to do this hour")
    amount_kwh: float = Field(
        ..., ge=0.0, le=10.0,
        description="Energy amount in kWh (0-10). Clamped to available capacity."
    )

class SolarGridObservation(BaseModel):
    """What the agent sees each hour."""
    hour: int = Field(..., ge=0, le=23, description="Current hour (0-23)")
    current_price: float = Field(..., description="IEX DAM price Rs/kWh this hour")
    next_3h_prices: List[float] = Field(..., description="Forecast prices for next 3 hours")
    solar_generation_kwh: float = Field(..., ge=0, description="Solar output this hour in kWh")
    battery_soc: float = Field(..., ge=0.0, le=1.0, description="Battery state of charge (0-1)")
    battery_capacity_kwh: float = Field(..., description="Total battery capacity in kWh")
    energy_consumed_kwh: float = Field(..., description="Household consumption this hour")
    cumulative_revenue: float = Field(..., description="Total Rs earned so far today")
    cumulative_cost: float = Field(..., description="Total Rs spent so far today")
    hours_remaining: int = Field(..., description="Hours left in episode")
    day_type: str = Field(..., description="'weekday' or 'weekend'")
    season: str = Field(..., description="'summer', 'monsoon', 'winter'")
    message: str = Field(default="", description="Environment feedback message")

class SolarGridState(BaseModel):
    """Internal environment state."""
    episode_id: str
    step_count: int = 0
    hour: int = 0
    battery_soc: float = 0.5  # start half charged
    battery_capacity_kwh: float = 13.5  # typical home battery (Tesla Powerwall size)
    solar_panel_kw: float = 5.0  # 5kW rooftop system
    cumulative_revenue: float = 0.0
    cumulative_cost: float = 0.0
    cumulative_self_consumption_value: float = 0.0
    price_profile: List[float] = Field(default_factory=list)
    solar_profile: List[float] = Field(default_factory=list)
    consumption_profile: List[float] = Field(default_factory=list)
    actions_taken: List[str] = Field(default_factory=list)
    day_type: str = "weekday"
    season: str = "summer"
    done: bool = False
