"""Main simulator interface for the logistics scheduling environment."""

from __future__ import annotations

from dataclasses import replace
from typing import List

from .config import SimulationConfig
from .generator import generate_events, generate_orders, generate_road_network, generate_vehicles
from .models import Event, EventType, Order, RoadNetwork, Vehicle


class LogisticsSimulator:
    """Coordinates generation and state transitions for the simulation."""

    def __init__(self, config: SimulationConfig | None = None) -> None:
        self.config = config or SimulationConfig()
        self.road_network: RoadNetwork | None = None
        self.vehicles: List[Vehicle] = []
        self.orders: List[Order] = []
        self.events: List[Event] = []
        self.current_time: float = self.config.start_hour

    def generate(self) -> "LogisticsSimulator":
        """Generate road network, vehicles, orders, and initial events."""

        self.road_network = generate_road_network(self.config)
        self.vehicles = generate_vehicles(self.config, self.road_network)
        self.orders = generate_orders(self.config, self.road_network)
        self.events = generate_events(self.config, self.orders, self.road_network)
        self.current_time = self.config.start_hour
        return self

    def injectevent(self, event: Event) -> None:
        """Apply an event to the in-memory simulation state."""

        self.events.append(event)
        self.events.sort(key=lambda item: item.timestamp)
        self.current_time = max(self.current_time, event.timestamp)

        if event.type is EventType.ORDERCANCEL:
            affected_order_ids = set(event.affected_orders)
            for order in self.orders:
                if order.id in affected_order_ids:
                    order.cancelled = True

        elif event.type is EventType.VEHICLEBREAKDOWN:
            for vehicle in self.vehicles:
                if vehicle.id == event.location or vehicle.currentlocation == event.location:
                    vehicle.available = False

        elif event.type is EventType.TRAFFICACCIDENT:
            self._update_edge_congestion(event.location, increment=0.1 * event.severity)

        elif event.type is EventType.ROADCLOSED:
            self._set_edge_closed(event.location)

    def inject_event(self, event: Event) -> None:
        self.injectevent(event)

    def getactiveorders(self) -> List[Order]:
        """Return orders that are not cancelled and not expired."""

        return [
            order
            for order in self.orders
            if not order.cancelled
            and order.created_at <= self.current_time <= order.deliverywindow.endhour
        ]

    def get_active_orders(self) -> List[Order]:
        return self.getactiveorders()

    def getvehicle_states(self) -> List[Vehicle]:
        """Return shallow copies of all vehicle states."""

        return [replace(vehicle) for vehicle in self.vehicles]

    def get_vehicle_states(self) -> List[Vehicle]:
        return self.getvehicle_states()

    def reset(self) -> None:
        """Clear all generated state and reset the clock."""

        self.road_network = None
        self.vehicles = []
        self.orders = []
        self.events = []
        self.current_time = self.config.start_hour

    def _update_edge_congestion(self, location: str, increment: float) -> None:
        if self.road_network is None:
            return
        for edge in self.road_network.edges:
            if f"{edge.fromnode}->{edge.tonode}" == location:
                edge.congestion = min(1.0, edge.congestion + increment)
                return

    def _set_edge_closed(self, location: str) -> None:
        if self.road_network is None:
            return
        for edge in self.road_network.edges:
            if f"{edge.fromnode}->{edge.tonode}" == location:
                edge.congestion = 1.0
                return
