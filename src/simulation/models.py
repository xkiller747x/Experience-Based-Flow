"""Data models for the logistics scheduling simulation environment."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List


class EventType(str, Enum):
    """Supported exceptional event types."""

    # 原有 4 种
    VEHICLEBREAKDOWN = "vehicle_breakdown"
    TRAFFICACCIDENT = "traffic_accident"
    ROADCLOSED = "road_closed"
    ORDERCANCEL = "order_cancel"
    # 新增 16 种
    VEHICLEMAINTENANCE = "vehicle_maintenance"
    FUELSHORTAGE = "fuel_shortage"
    DRIVERUNAVAILABLE = "driver_unavailable"
    TRAFFICCONGESTION = "traffic_congestion"
    ROADNARROWING = "road_narrowing"
    BRIDGEWEIGHTLIMIT = "bridge_weight_limit"
    ORDERMODIFY = "order_modify"
    ORDERUPDATE = "order_update"
    PRIORITYORDERURGENT = "priority_order_urgent"
    DELIVERYFAILURE = "delivery_failure"
    DEMANDSURGE = "demand_surge"
    DEMANDDROP = "demand_drop"
    WEATHERDELAY = "weather_delay"
    NATURALDISASTER = "natural_disaster"
    PUBLICEVENT = "public_event"
    WAREHOUSEDELAY = "warehouse_delay"
    INVENTORYSTOCKOUT = "inventory_stockout"


@dataclass
class TimeWindow:
    """Pickup or delivery time window measured in simulation hours."""

    starthour: float
    endhour: float

    def __init__(
        self,
        starthour: float | None = None,
        endhour: float | None = None,
        start_hour: float | None = None,
        end_hour: float | None = None,
    ) -> None:
        if start_hour is not None:
            starthour = start_hour
        if end_hour is not None:
            endhour = end_hour
        if starthour is None or endhour is None:
            raise TypeError("starthour/endhour are required")

        self.starthour = starthour
        self.endhour = endhour
        self.__post_init__()

    def __post_init__(self) -> None:
        if self.starthour < 0:
            raise ValueError("starthour must be non-negative")
        if self.endhour <= self.starthour:
            raise ValueError("endhour must be greater than starthour")

    @property
    def start_hour(self) -> float:
        return self.starthour

    @start_hour.setter
    def start_hour(self, value: float) -> None:
        self.starthour = value

    @property
    def end_hour(self) -> float:
        return self.endhour

    @end_hour.setter
    def end_hour(self, value: float) -> None:
        self.endhour = value

    def contains(self, hour: float) -> bool:
        return self.starthour <= hour <= self.endhour


@dataclass
class Node:
    """A graph node in the road network."""

    id: str
    lat: float
    lon: float
    type: str

    def __post_init__(self) -> None:
        valid_types = {"depot", "customer", "intermediate"}
        if self.type not in valid_types:
            raise ValueError(f"type must be one of {sorted(valid_types)}")


@dataclass
class Edge:
    """A directed road segment between two nodes."""

    fromnode: str
    tonode: str
    distance: float
    travel_time: float
    congestion: float

    def __init__(
        self,
        fromnode: str | None = None,
        tonode: str | None = None,
        distance: float = 0.0,
        travel_time: float = 0.0,
        congestion: float = 0.0,
        from_node: str | None = None,
        to_node: str | None = None,
    ) -> None:
        if from_node is not None:
            fromnode = from_node
        if to_node is not None:
            tonode = to_node
        if fromnode is None or tonode is None:
            raise TypeError("fromnode/tonode are required")

        self.fromnode = fromnode
        self.tonode = tonode
        self.distance = distance
        self.travel_time = travel_time
        self.congestion = congestion
        self.__post_init__()

    def __post_init__(self) -> None:
        if self.distance < 0:
            raise ValueError("distance must be non-negative")
        if self.travel_time < 0:
            raise ValueError("travel_time must be non-negative")
        if not 0 <= self.congestion <= 1:
            raise ValueError("congestion must be between 0 and 1")

    @property
    def from_node(self) -> str:
        return self.fromnode

    @from_node.setter
    def from_node(self, value: str) -> None:
        self.fromnode = value

    @property
    def to_node(self) -> str:
        return self.tonode

    @to_node.setter
    def to_node(self, value: str) -> None:
        self.tonode = value

    def effective_travel_time(self) -> float:
        return self.travel_time * (1 + self.congestion)


@dataclass
class RoadNetwork:
    """Fully connected road network used by the simulator."""

    nodes: List[Node] = field(default_factory=list)
    edges: List[Edge] = field(default_factory=list)
    depot_id: str = ""

    def get_node(self, node_id: str) -> Node | None:
        return next((node for node in self.nodes if node.id == node_id), None)

    def get_neighbors(self, node_id: str) -> List[str]:
        return [edge.tonode for edge in self.edges if edge.fromnode == node_id]

    def get_edge(self, fromnode: str, tonode: str) -> Edge | None:
        return next(
            (
                edge
                for edge in self.edges
                if edge.fromnode == fromnode and edge.tonode == tonode
            ),
            None,
        )


@dataclass
class Vehicle:
    """Vehicle state and capacity constraints."""

    id: str
    capacityweight: float
    capacityvolume: float
    currentlocation: str
    speed: float
    costper_km: float
    available: bool = True

    def __init__(
        self,
        id: str,
        capacityweight: float | None = None,
        capacityvolume: float | None = None,
        currentlocation: str | None = None,
        speed: float = 40.0,
        costper_km: float | None = None,
        available: bool = True,
        capacity_weight: float | None = None,
        capacity_volume: float | None = None,
        current_location: str | None = None,
        cost_per_km: float | None = None,
    ) -> None:
        if capacity_weight is not None:
            capacityweight = capacity_weight
        if capacity_volume is not None:
            capacityvolume = capacity_volume
        if current_location is not None:
            currentlocation = current_location
        if cost_per_km is not None:
            costper_km = cost_per_km
        if capacityweight is None or capacityvolume is None:
            raise TypeError("capacityweight/capacityvolume are required")
        if currentlocation is None:
            raise TypeError("currentlocation is required")
        if costper_km is None:
            raise TypeError("costper_km is required")

        self.id = id
        self.capacityweight = capacityweight
        self.capacityvolume = capacityvolume
        self.currentlocation = currentlocation
        self.speed = speed
        self.costper_km = costper_km
        self.available = available
        self.__post_init__()

    def __post_init__(self) -> None:
        if self.capacityweight <= 0:
            raise ValueError("capacityweight must be positive")
        if self.capacityvolume <= 0:
            raise ValueError("capacityvolume must be positive")
        if self.speed <= 0:
            raise ValueError("speed must be positive")
        if self.costper_km < 0:
            raise ValueError("costper_km must be non-negative")

    @property
    def capacity_weight(self) -> float:
        return self.capacityweight

    @capacity_weight.setter
    def capacity_weight(self, value: float) -> None:
        self.capacityweight = value

    @property
    def capacity_volume(self) -> float:
        return self.capacityvolume

    @capacity_volume.setter
    def capacity_volume(self, value: float) -> None:
        self.capacityvolume = value

    @property
    def current_location(self) -> str:
        return self.currentlocation

    @current_location.setter
    def current_location(self, value: str) -> None:
        self.currentlocation = value

    @property
    def cost_per_km(self) -> float:
        return self.costper_km

    @cost_per_km.setter
    def cost_per_km(self, value: float) -> None:
        self.costper_km = value

    def can_carry(self, weight: float, volume: float) -> bool:
        return weight <= self.capacityweight and volume <= self.capacityvolume


@dataclass
class Order:
    """Pickup and delivery order request."""

    id: str
    weight: float
    volume: float
    pickup: str
    delivery: str
    pickupwindow: TimeWindow
    deliverywindow: TimeWindow
    priority: int
    created_at: float
    cancelled: bool = False
    assignedvehicle: str | None = None

    def __init__(
        self,
        id: str,
        weight: float,
        volume: float,
        pickup: str,
        delivery: str,
        pickupwindow: TimeWindow | None = None,
        deliverywindow: TimeWindow | None = None,
        priority: int = 1,
        created_at: float = 0.0,
        cancelled: bool = False,
        assignedvehicle: str | None = None,
        pickup_window: TimeWindow | None = None,
        delivery_window: TimeWindow | None = None,
        assigned_vehicle: str | None = None,
    ) -> None:
        if pickup_window is not None:
            pickupwindow = pickup_window
        if delivery_window is not None:
            deliverywindow = delivery_window
        if assigned_vehicle is not None:
            assignedvehicle = assigned_vehicle
        if pickupwindow is None or deliverywindow is None:
            raise TypeError("pickupwindow/deliverywindow are required")

        self.id = id
        self.weight = weight
        self.volume = volume
        self.pickup = pickup
        self.delivery = delivery
        self.pickupwindow = pickupwindow
        self.deliverywindow = deliverywindow
        self.priority = priority
        self.created_at = created_at
        self.cancelled = cancelled
        self.assignedvehicle = assignedvehicle
        self.__post_init__()

    def __post_init__(self) -> None:
        if self.weight <= 0:
            raise ValueError("weight must be positive")
        if self.volume <= 0:
            raise ValueError("volume must be positive")
        if self.pickup == self.delivery:
            raise ValueError("pickup and delivery must be different nodes")
        if not 1 <= self.priority <= 5:
            raise ValueError("priority must be between 1 and 5")

    @property
    def pickup_window(self) -> TimeWindow:
        return self.pickupwindow

    @pickup_window.setter
    def pickup_window(self, value: TimeWindow) -> None:
        self.pickupwindow = value

    @property
    def delivery_window(self) -> TimeWindow:
        return self.deliverywindow

    @delivery_window.setter
    def delivery_window(self, value: TimeWindow) -> None:
        self.deliverywindow = value

    @property
    def assigned_vehicle(self) -> str | None:
        return self.assignedvehicle

    @assigned_vehicle.setter
    def assigned_vehicle(self, value: str | None) -> None:
        self.assignedvehicle = value


@dataclass
class Event:
    """Exceptional event that can affect routes, orders, or vehicles."""

    timestamp: float
    type: EventType
    location: str
    severity: int
    affected_orders: List[str] = field(default_factory=list)
    description: str = ""

    def __post_init__(self) -> None:
        if self.timestamp < 0:
            raise ValueError("timestamp must be non-negative")
        if not 1 <= self.severity <= 5:
            raise ValueError("severity must be between 1 and 5")
