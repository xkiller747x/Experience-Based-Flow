"""Predefined simulation scenario configurations."""

from __future__ import annotations

from .config import SimulationConfig


SCENARIOS: dict[str, SimulationConfig] = {
    "small": SimulationConfig(num_vehicles=5, num_orders=20, area_size_km=30.0),
    "medium": SimulationConfig(num_vehicles=10, num_orders=50, area_size_km=50.0),
    "large": SimulationConfig(num_vehicles=20, num_orders=100, area_size_km=80.0),
    "stress": SimulationConfig(
        num_vehicles=30,
        num_orders=200,
        area_size_km=100.0,
        eventfrequency=0.3,
        severeevent_prob=0.4,
    ),
}


def build_scenario(scenario_type: str) -> SimulationConfig:
    """Return a SimulationConfig for small, medium, large, or stress."""

    try:
        scenario = SCENARIOS[scenario_type.lower()]
    except KeyError as error:
        available = ", ".join(sorted(SCENARIOS))
        raise ValueError(f"Unknown scenario '{scenario_type}'. Available: {available}") from error

    return SimulationConfig(**scenario.__dict__)


def small() -> SimulationConfig:
    return build_scenario("small")


def medium() -> SimulationConfig:
    return build_scenario("medium")


def large() -> SimulationConfig:
    return build_scenario("large")


def stress() -> SimulationConfig:
    return build_scenario("stress")
