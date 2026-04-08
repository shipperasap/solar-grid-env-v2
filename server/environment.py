import uuid
import random
from typing import Optional

from .models import SolarGridAction, SolarGridObservation, SolarGridState, ActionType
from .price_engine import (
    generate_price_profile, generate_solar_profile,
    generate_consumption_profile, get_retail_tariff
)
from .reward import compute_reward


BATTERY_CHARGE_EFFICIENCY = 0.92    # 92% round-trip efficiency
BATTERY_DISCHARGE_EFFICIENCY = 0.95  # 95% discharge efficiency
BATTERY_MAX_CHARGE_RATE = 5.0       # kW max charge rate
BATTERY_MAX_DISCHARGE_RATE = 5.0    # kW max discharge rate
GRID_SELL_EFFICIENCY = 0.97         # 3% transmission loss


class SolarGridEnvironment:
    """
    Indian Electricity Grid Arbitrage Environment.

    A rooftop solar prosumer with battery storage makes hourly decisions
    on the IEX Day-Ahead Market to maximize daily revenue.

    Episode: 24 steps (one full day, hour 0 to hour 23)
    """

    def __init__(self):
        self.state: Optional[SolarGridState] = None

    def reset(self) -> SolarGridObservation:
        """Reset environment for a new day."""
        season = random.choice(["summer", "summer", "winter", "monsoon"])
        day_type = random.choice(["weekday", "weekday", "weekday", "weekday", "weekend"])
        initial_soc = random.uniform(0.3, 0.7)

        price_profile = generate_price_profile(season, day_type)
        solar_profile = generate_solar_profile(season, panel_kw=5.0)
        consumption_profile = generate_consumption_profile(day_type, season)

        self.state = SolarGridState(
            episode_id=str(uuid.uuid4()),
            step_count=0,
            hour=0,
            battery_soc=initial_soc,
            battery_capacity_kwh=13.5,
            solar_panel_kw=5.0,
            cumulative_revenue=0.0,
            cumulative_cost=0.0,
            cumulative_self_consumption_value=0.0,
            price_profile=price_profile,
            solar_profile=solar_profile,
            consumption_profile=consumption_profile,
            actions_taken=[],
            day_type=day_type,
            season=season,
            done=False,
        )

        return self._make_observation("New day started. Battery at {:.0%}. Season: {}. Day: {}.".format(
            initial_soc, season, day_type
        ))

    def step(self, action: SolarGridAction) -> dict:
        """Execute one hourly step."""
        if self.state is None:
            raise ValueError("Environment not initialized. Call reset() first.")
        if self.state.done:
            raise ValueError("Episode is done. Call reset() to start a new episode.")

        hour = self.state.hour
        solar_kwh = self.state.solar_profile[hour]
        consumption_kwh = self.state.consumption_profile[hour]
        current_price = self.state.price_profile[hour]
        retail_tariff = get_retail_tariff(hour, self.state.season)
        battery_energy = self.state.battery_soc * self.state.battery_capacity_kwh

        soc_before = self.state.battery_soc
        actual_amount = 0.0
        message_parts = []

        net_solar = max(0, solar_kwh - consumption_kwh)
        solar_deficit = max(0, consumption_kwh - solar_kwh)

        # --- Execute action ---
        if action.action_type == ActionType.SELL:
            max_from_battery = min(
                battery_energy * BATTERY_DISCHARGE_EFFICIENCY,
                BATTERY_MAX_DISCHARGE_RATE
            )
            max_sellable = net_solar * GRID_SELL_EFFICIENCY + max_from_battery
            actual_amount = min(action.amount_kwh, max_sellable)

            if actual_amount > 0:
                sold_from_solar = min(net_solar * GRID_SELL_EFFICIENCY, actual_amount)
                sold_from_battery = actual_amount - sold_from_solar

                if sold_from_battery > 0:
                    battery_drain = sold_from_battery / BATTERY_DISCHARGE_EFFICIENCY
                    self.state.battery_soc -= battery_drain / self.state.battery_capacity_kwh

                revenue = actual_amount * current_price
                self.state.cumulative_revenue += revenue
                message_parts.append(f"Sold {actual_amount:.1f}kWh at Rs{current_price:.2f} = Rs{revenue:.2f}")
            else:
                message_parts.append("Nothing to sell (no solar surplus, battery empty)")

        elif action.action_type == ActionType.STORE:
            available_capacity = (1.0 - self.state.battery_soc) * self.state.battery_capacity_kwh
            max_storable = min(
                net_solar,
                available_capacity / BATTERY_CHARGE_EFFICIENCY,
                BATTERY_MAX_CHARGE_RATE
            )
            actual_amount = min(action.amount_kwh, max_storable)

            if actual_amount > 0:
                stored = actual_amount * BATTERY_CHARGE_EFFICIENCY
                self.state.battery_soc += stored / self.state.battery_capacity_kwh
                message_parts.append(f"Stored {actual_amount:.1f}kWh (->{stored:.1f}kWh after losses). SOC: {self.state.battery_soc:.0%}")
            else:
                message_parts.append("Cannot store: no surplus solar or battery full")

        elif action.action_type == ActionType.BUY:
            available_capacity = (1.0 - self.state.battery_soc) * self.state.battery_capacity_kwh
            max_buyable = min(
                available_capacity / BATTERY_CHARGE_EFFICIENCY,
                BATTERY_MAX_CHARGE_RATE,
                action.amount_kwh
            )
            actual_amount = min(action.amount_kwh, max_buyable)

            if actual_amount > 0:
                cost = actual_amount * current_price
                stored = actual_amount * BATTERY_CHARGE_EFFICIENCY
                self.state.battery_soc += stored / self.state.battery_capacity_kwh
                self.state.cumulative_cost += cost
                self.state.cumulative_revenue -= cost
                message_parts.append(f"Bought {actual_amount:.1f}kWh at Rs{current_price:.2f} = -Rs{cost:.2f}. SOC: {self.state.battery_soc:.0%}")
            else:
                message_parts.append("Cannot buy: battery full")

        elif action.action_type == ActionType.HOLD:
            self_consumed = min(solar_kwh, consumption_kwh)
            savings = self_consumed * retail_tariff
            self.state.cumulative_self_consumption_value += savings
            actual_amount = self_consumed
            message_parts.append(f"Self-consumed {self_consumed:.1f}kWh, saved Rs{savings:.2f} vs grid")

            # Store any surplus automatically
            surplus = net_solar
            if surplus > 0:
                available_cap = (1.0 - self.state.battery_soc) * self.state.battery_capacity_kwh
                auto_store = min(surplus, available_cap / BATTERY_CHARGE_EFFICIENCY)
                if auto_store > 0:
                    self.state.battery_soc += (auto_store * BATTERY_CHARGE_EFFICIENCY) / self.state.battery_capacity_kwh
                    message_parts.append(f"Auto-stored {auto_store:.1f}kWh surplus")

        # Clamp SOC (round to avoid float precision issues with Pydantic validation)
        self.state.battery_soc = round(max(0.0, min(1.0, self.state.battery_soc)), 10)

        # Handle consumption deficit
        if solar_deficit > 0 and action.action_type != ActionType.BUY:
            battery_available = self.state.battery_soc * self.state.battery_capacity_kwh * BATTERY_DISCHARGE_EFFICIENCY
            from_battery = min(solar_deficit, battery_available)
            if from_battery > 0:
                self.state.battery_soc -= (from_battery / BATTERY_DISCHARGE_EFFICIENCY) / self.state.battery_capacity_kwh
                solar_deficit -= from_battery

            if solar_deficit > 0:
                forced_cost = solar_deficit * retail_tariff
                self.state.cumulative_cost += forced_cost
                self.state.cumulative_revenue -= forced_cost
                message_parts.append(f"Grid purchase for deficit: {solar_deficit:.1f}kWh = -Rs{forced_cost:.2f}")

        # Re-clamp SOC after deficit handling
        self.state.battery_soc = round(max(0.0, min(1.0, self.state.battery_soc)), 10)

        # Compute reward
        hours_remaining = 23 - hour
        reward_dict = compute_reward(
            action_type=action.action_type.value,
            amount_kwh=action.amount_kwh,
            actual_amount=actual_amount,
            current_price=current_price,
            retail_tariff=retail_tariff,
            solar_available=solar_kwh,
            consumption=consumption_kwh,
            battery_soc_before=soc_before,
            battery_soc_after=self.state.battery_soc,
            battery_capacity=self.state.battery_capacity_kwh,
            hour=hour,
            hours_remaining=hours_remaining,
            season=self.state.season,
            cumulative_revenue=self.state.cumulative_revenue,
            price_profile=self.state.price_profile,
        )

        # Advance state
        self.state.step_count += 1
        self.state.actions_taken.append(f"{hour}:{action.action_type.value}:{actual_amount:.1f}")
        self.state.hour += 1

        if self.state.hour >= 24:
            self.state.done = True
            message_parts.append(f"DAY COMPLETE! Net revenue: Rs{self.state.cumulative_revenue:.2f}, Self-consumption value: Rs{self.state.cumulative_self_consumption_value:.2f}")

        observation = self._make_observation(" | ".join(message_parts))

        return {
            "observation": observation.model_dump(),
            "reward": reward_dict["total"],
            "done": self.state.done,
            "state": {
                "episode_id": self.state.episode_id,
                "step_count": self.state.step_count,
            },
            "reward_breakdown": reward_dict,
        }

    def _make_observation(self, message: str) -> SolarGridObservation:
        """Build observation from current state."""
        hour = min(self.state.hour, 23)
        prices = self.state.price_profile

        next_3h = []
        for h in range(hour + 1, min(hour + 4, 24)):
            noisy_price = prices[h] * random.uniform(0.9, 1.1)
            next_3h.append(round(noisy_price, 2))
        while len(next_3h) < 3:
            next_3h.append(0.0)

        return SolarGridObservation(
            hour=hour,
            current_price=prices[hour] if hour < 24 else 0.0,
            next_3h_prices=next_3h,
            solar_generation_kwh=self.state.solar_profile[hour] if hour < 24 else 0.0,
            battery_soc=self.state.battery_soc,
            battery_capacity_kwh=self.state.battery_capacity_kwh,
            energy_consumed_kwh=self.state.consumption_profile[hour] if hour < 24 else 0.0,
            cumulative_revenue=round(self.state.cumulative_revenue, 2),
            cumulative_cost=round(self.state.cumulative_cost, 2),
            hours_remaining=max(0, 23 - hour),
            day_type=self.state.day_type,
            season=self.state.season,
            message=message,
        )

    def get_state(self) -> dict:
        """Return full state for debugging."""
        if self.state:
            return self.state.model_dump()
        return {}
