"""Synthetic data generators for the logistics simulation environment."""

from __future__ import annotations

import math
import random
from typing import List

from .config import SimulationConfig
from .models import Edge, Event, EventType, Node, Order, RoadNetwork, TimeWindow, Vehicle


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate the great-circle distance between two coordinates in km."""

    earth_radius_km = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    haversine = (
        math.sin(delta_phi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    )
    return 2 * earth_radius_km * math.asin(math.sqrt(haversine))


def _rng(config: SimulationConfig, offset: int = 0) -> random.Random:
    seed = None if config.seed is None else config.seed + offset
    return random.Random(seed)


def _random_node_position(
    rng: random.Random,
    center_lat: float,
    center_lon: float,
    area_size_km: float,
) -> tuple[float, float]:
    angle = rng.uniform(0, 2 * math.pi)
    radius_km = area_size_km * math.sqrt(rng.random())
    lat = center_lat + (radius_km / 111.0) * math.cos(angle)
    lon = center_lon + (radius_km / (111.0 * math.cos(math.radians(center_lat)))) * math.sin(angle)
    return lat, lon


def generate_road_network(config: SimulationConfig) -> RoadNetwork:
    """Generate one depot, customer nodes, intermediate nodes, and full edges."""

    rng = _rng(config)
    center_lat = 31.2304
    center_lon = 121.4737
    depot_id = "depot_0"

    nodes: List[Node] = [
        Node(id=depot_id, lat=center_lat, lon=center_lon, type="depot")
    ]

    for index in range(config.num_orders):
        lat, lon = _random_node_position(
            rng,
            center_lat,
            center_lon,
            config.area_size_km,
        )
        nodes.append(Node(id=f"customer_{index}", lat=lat, lon=lon, type="customer"))

    for index in range(config.num_intermediate_nodes):
        lat, lon = _random_node_position(
            rng,
            center_lat,
            center_lon,
            config.area_size_km,
        )
        nodes.append(
            Node(id=f"intermediate_{index}", lat=lat, lon=lon, type="intermediate")
        )

    edges: List[Edge] = []
    for from_node in nodes:
        for to_node in nodes:
            if from_node.id == to_node.id:
                continue
            distance = haversine_distance(
                from_node.lat,
                from_node.lon,
                to_node.lat,
                to_node.lon,
            )
            congestion = rng.uniform(0.0, 0.4)
            travel_time = (distance / 40.0) * 60.0
            edges.append(
                Edge(
                    fromnode=from_node.id,
                    tonode=to_node.id,
                    distance=distance,
                    travel_time=travel_time,
                    congestion=congestion,
                )
            )

    return RoadNetwork(nodes=nodes, edges=edges, depot_id=depot_id)


def generate_vehicles(config: SimulationConfig, road_network: RoadNetwork) -> List[Vehicle]:
    """Generate vehicles starting from the depot with 1:1 weight-volume capacity."""

    rng = _rng(config, offset=1)
    vehicles: List[Vehicle] = []
    for index in range(config.num_vehicles):
        capacityweight = rng.uniform(
            config.min_capacity_weight,
            config.max_capacity_weight,
        )
        vehicles.append(
            Vehicle(
                id=f"vehicle_{index}",
                capacityweight=capacityweight,
                capacityvolume=capacityweight,
                currentlocation=road_network.depot_id,
                speed=rng.uniform(config.min_vehicle_speed, config.max_vehicle_speed),
                costper_km=rng.uniform(config.min_costper_km, config.max_costper_km),
                available=True,
            )
        )
    return vehicles


def generate_orders(config: SimulationConfig, road_network: RoadNetwork) -> List[Order]:
    """Generate orders between different customer nodes."""

    rng = _rng(config, offset=2)
    customer_ids = [node.id for node in road_network.nodes if node.type == "customer"]
    if config.num_orders and len(customer_ids) < 2:
        raise ValueError("at least two customer nodes are required to generate orders")

    orders: List[Order] = []
    offset_min = config.order_window_min_offset
    offset_max = config.order_window_max_offset
    latest_start = config.start_hour + offset_max - config.time_window_width

    for index in range(config.num_orders):
        pickup = rng.choice(customer_ids)
        delivery = rng.choice([node_id for node_id in customer_ids if node_id != pickup])
        pickup_start = rng.uniform(config.start_hour + offset_min, latest_start)
        delivery_start = rng.uniform(pickup_start, latest_start)
        weight = rng.uniform(config.min_order_weight, config.max_order_weight)

        orders.append(
            Order(
                id=f"order_{index}",
                weight=weight,
                volume=weight,
                pickup=pickup,
                delivery=delivery,
                pickupwindow=TimeWindow(
                    starthour=pickup_start,
                    endhour=pickup_start + config.time_window_width,
                ),
                deliverywindow=TimeWindow(
                    starthour=delivery_start,
                    endhour=delivery_start + config.time_window_width,
                ),
                priority=rng.randint(1, 5),
                created_at=config.start_hour,
            )
        )
    return orders


def generate_events(
    config: SimulationConfig,
    orders: List[Order],
    road_network: RoadNetwork,
    num_hours: int | None = None,
) -> List[Event]:
    """Generate random exceptional events using config.eventfrequency."""

    rng = _rng(config, offset=3)
    hours = config.simulation_hours if num_hours is None else num_hours
    events: List[Event] = []
    event_types = list(EventType)

    for hour in range(hours):
        if rng.random() >= config.eventfrequency:
            continue

        # 按照真实物流行业比例加权选择事件类型
        event_weights = {
            # 交通类 30%
            EventType.TRAFFICACCIDENT: 7,
            EventType.TRAFFICCONGESTION: 10,
            EventType.ROADCLOSED: 6,
            EventType.ROADNARROWING: 4,
            EventType.BRIDGEWEIGHTLIMIT: 3,
            # 订单类 25%
            EventType.ORDERCANCEL: 7,
            EventType.ORDERMODIFY: 6,
            EventType.ORDERUPDATE: 5,
            EventType.PRIORITYORDERURGENT: 4,
            EventType.DELIVERYFAILURE: 3,
            # 车辆类 20%
            EventType.VEHICLEBREAKDOWN: 8,
            EventType.VEHICLEMAINTENANCE: 4,
            EventType.DRIVERUNAVAILABLE: 5,
            EventType.FUELSHORTAGE: 3,
            # 外部环境类 10%
            EventType.WEATHERDELAY: 5,
            EventType.NATURALDISASTER: 2,
            EventType.PUBLICEVENT: 3,
            # 仓配类 10%
            EventType.WAREHOUSEDELAY: 6,
            EventType.INVENTORYSTOCKOUT: 4,
            # 其他 5%
            EventType.DEMANDSURGE: 2,
            EventType.DEMANDDROP: 3,
        }
        event_type = rng.choices(list(event_weights.keys()), weights=list(event_weights.values()), k=1)[0]
        affected_orders: List[str] = []

        if (
            event_type in (EventType.TRAFFICACCIDENT, EventType.TRAFFICCONGESTION,
                           EventType.ROADCLOSED, EventType.ROADNARROWING,
                           EventType.BRIDGEWEIGHTLIMIT)
            and road_network.edges
        ):
            edge = rng.choice(road_network.edges)
            location = f"{edge.fromnode}->{edge.tonode}"
        elif event_type in (EventType.ORDERCANCEL, EventType.ORDERMODIFY,
                            EventType.ORDERUPDATE, EventType.PRIORITYORDERURGENT,
                            EventType.DELIVERYFAILURE) and orders:
            order = rng.choice(orders)
            affected_orders = [order.id]
            location = order.pickup
        elif event_type in (EventType.VEHICLEBREAKDOWN, EventType.VEHICLEMAINTENANCE,
                            EventType.DRIVERUNAVAILABLE) and config.num_vehicles > 0:
            location = f"vehicle_{rng.randint(0, config.num_vehicles - 1)}"
        elif event_type in (EventType.WAREHOUSEDELAY, EventType.INVENTORYSTOCKOUT):
            location = "depot_0"
        else:
            location = rng.choice(road_network.nodes).id

        severity = (
            rng.randint(4, 5)
            if rng.random() < config.severeevent_prob
            else rng.randint(1, 3)
        )
        descriptions = {
            EventType.TRAFFICACCIDENT: "Traffic accident reported",
            EventType.ROADCLOSED: "Road closed",
            EventType.ORDERCANCEL: "Order cancelled by customer",
            EventType.VEHICLEBREAKDOWN: "Vehicle breakdown reported",
            EventType.VEHICLEMAINTENANCE: "Vehicle scheduled maintenance",
            EventType.FUELSHORTAGE: "Fuel shortage reported",
            EventType.DRIVERUNAVAILABLE: "Driver unavailable",
            EventType.TRAFFICCONGESTION: "Traffic congestion reported",
            EventType.ROADNARROWING: "Road narrowing reported",
            EventType.BRIDGEWEIGHTLIMIT: "Bridge weight limit imposed",
            EventType.ORDERMODIFY: "Order modified by customer",
            EventType.ORDERUPDATE: "Order updated",
            EventType.PRIORITYORDERURGENT: "Priority order flagged urgent",
            EventType.DELIVERYFAILURE: "Delivery failure reported",
            EventType.DEMANDSURGE: "Demand surge detected",
            EventType.DEMANDDROP: "Demand drop detected",
            EventType.WEATHERDELAY: "Weather delay reported",
            EventType.NATURALDISASTER: "Natural disaster reported",
            EventType.PUBLICEVENT: "Public event causing disruption",
            EventType.WAREHOUSEDELAY: "Warehouse processing delay",
            EventType.INVENTORYSTOCKOUT: "Inventory stockout",
        }

        events.append(
            Event(
                timestamp=config.start_hour + float(hour),
                type=event_type,
                location=location,
                severity=severity,
                affected_orders=affected_orders,
                description=descriptions.get(event_type, f"Event: {event_type.value}"),
            )
        )

    events.sort(key=lambda event: event.timestamp)
    return events
