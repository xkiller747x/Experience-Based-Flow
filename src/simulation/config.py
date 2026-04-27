"""Configuration for the logistics scheduling simulation environment."""

from dataclasses import dataclass


@dataclass(init=False)
class SimulationConfig:
    """Configurable parameters controlling simulation scale and randomness."""

    num_vehicles: int = 10
    num_orders: int = 50
    area_size_km: float = 50.0

    min_capacity_weight: float = 500.0
    max_capacity_weight: float = 2000.0
    min_order_weight: float = 10.0
    max_order_weight: float = 200.0

    time_window_width: float = 1.0
    start_hour: float = 8.0
    order_window_min_offset: float = 1.0
    order_window_max_offset: float = 4.0

    eventfrequency: float = 0.1
    severeevent_prob: float = 0.2
    simulation_hours: int = 8
    num_intermediate_nodes: int = 10

    min_vehicle_speed: float = 35.0
    max_vehicle_speed: float = 60.0
    min_costper_km: float = 0.5
    max_costper_km: float = 2.0

    seed: int | None = 42

    def __init__(
        self,
        num_vehicles: int = 10,
        num_orders: int = 50,
        area_size_km: float = 50.0,
        min_capacity_weight: float = 500.0,
        max_capacity_weight: float = 2000.0,
        min_order_weight: float = 10.0,
        max_order_weight: float = 200.0,
        time_window_width: float = 1.0,
        start_hour: float = 8.0,
        order_window_min_offset: float = 1.0,
        order_window_max_offset: float = 4.0,
        eventfrequency: float | None = None,
        severeevent_prob: float | None = None,
        event_frequency: float | None = None,
        severe_event_prob: float | None = None,
        simulation_hours: int = 8,
        num_intermediate_nodes: int = 10,
        min_vehicle_speed: float = 35.0,
        max_vehicle_speed: float = 60.0,
        min_costper_km: float = 0.5,
        max_costper_km: float = 2.0,
        seed: int | None = 42,
    ) -> None:
        self.num_vehicles = num_vehicles
        self.num_orders = num_orders
        self.area_size_km = area_size_km
        self.min_capacity_weight = min_capacity_weight
        self.max_capacity_weight = max_capacity_weight
        self.min_order_weight = min_order_weight
        self.max_order_weight = max_order_weight
        self.time_window_width = time_window_width
        self.start_hour = start_hour
        self.order_window_min_offset = order_window_min_offset
        self.order_window_max_offset = order_window_max_offset
        self.eventfrequency = (
            event_frequency
            if event_frequency is not None
            else 0.1
            if eventfrequency is None
            else eventfrequency
        )
        self.severeevent_prob = (
            severe_event_prob
            if severe_event_prob is not None
            else 0.2
            if severeevent_prob is None
            else severeevent_prob
        )
        self.simulation_hours = simulation_hours
        self.num_intermediate_nodes = num_intermediate_nodes
        self.min_vehicle_speed = min_vehicle_speed
        self.max_vehicle_speed = max_vehicle_speed
        self.min_costper_km = min_costper_km
        self.max_costper_km = max_costper_km
        self.seed = seed
        self.__post_init__()

    @property
    def event_frequency(self) -> float:
        return self.eventfrequency

    @event_frequency.setter
    def event_frequency(self, value: float) -> None:
        self.eventfrequency = value

    @property
    def severe_event_prob(self) -> float:
        return self.severeevent_prob

    @severe_event_prob.setter
    def severe_event_prob(self, value: float) -> None:
        self.severeevent_prob = value

    def __post_init__(self) -> None:
        if self.num_vehicles < 0:
            raise ValueError("num_vehicles must be non-negative")
        if self.num_orders < 0:
            raise ValueError("num_orders must be non-negative")
        if self.area_size_km <= 0:
            raise ValueError("area_size_km must be positive")
        if self.min_capacity_weight <= 0:
            raise ValueError("min_capacity_weight must be positive")
        if self.max_capacity_weight < self.min_capacity_weight:
            raise ValueError("max_capacity_weight must be >= min_capacity_weight")
        if self.min_order_weight <= 0:
            raise ValueError("min_order_weight must be positive")
        if self.max_order_weight < self.min_order_weight:
            raise ValueError("max_order_weight must be >= min_order_weight")
        if self.time_window_width <= 0:
            raise ValueError("time_window_width must be positive")
        if self.order_window_min_offset < 0:
            raise ValueError("order_window_min_offset must be non-negative")
        if self.order_window_max_offset <= self.order_window_min_offset:
            raise ValueError("order_window_max_offset must exceed order_window_min_offset")
        if self.time_window_width > (
            self.order_window_max_offset - self.order_window_min_offset
        ):
            raise ValueError("time_window_width must fit inside the order offset range")
        if not 0 <= self.eventfrequency <= 1:
            raise ValueError("eventfrequency must be between 0 and 1")
        if not 0 <= self.severeevent_prob <= 1:
            raise ValueError("severeevent_prob must be between 0 and 1")
        if self.simulation_hours < 0:
            raise ValueError("simulation_hours must be non-negative")
        if self.num_intermediate_nodes < 0:
            raise ValueError("num_intermediate_nodes must be non-negative")
