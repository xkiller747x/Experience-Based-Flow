"""VRP solver abstractions backed by Google OR-Tools."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Literal, Sequence

from src.simulation.models import Order, RoadNetwork, TimeWindow, Vehicle


@dataclass(slots=True)
class RouteStop:
    """A single pickup or delivery visit in a vehicle route."""

    node_id: str
    stop_type: Literal["pickup", "delivery"]
    order_id: str | None = None
    arrival_time: float = 0.0
    departure_time: float = 0.0
    travel_distance: float = 0.0
    travel_time: float = 0.0


@dataclass(slots=True)
class RoutePlan:
    """Aggregated routing result for all vehicles."""

    routes: dict[str, list[RouteStop]] = field(default_factory=dict)
    total_distance: float = 0.0
    total_travel_time: float = 0.0
    unassigned_order_ids: list[str] = field(default_factory=list)


class VRPSolver(ABC):
    """Base interface for VRP solvers."""

    @abstractmethod
    def solve(
        self,
        vehicles: Sequence[Vehicle],
        orders: Sequence[Order],
        road_network: RoadNetwork,
    ) -> RoutePlan:
        """Compute a route plan for the given vehicles and orders."""


@dataclass(slots=True)
class _VisitNode:
    node_id: str
    stop_type: Literal["depot", "pickup", "delivery"]
    order_id: str | None = None
    weight_demand: int = 0
    volume_demand: int = 0
    time_window: TimeWindow | None = None


class ORToolsSolver(VRPSolver):
    """Vehicle routing solver with depot starts, capacity, and time windows."""

    def __init__(
        self,
        time_limit_seconds: float = 10,
        distance_scale: int = 1000,
        capacity_scale: int = 1000,
        drop_penalty: int = 1_000_000_000,
    ) -> None:
        self.time_limit_seconds = max(0.001, float(time_limit_seconds))
        self.distance_scale = distance_scale
        self.capacity_scale = capacity_scale
        self.drop_penalty = drop_penalty

    def solve(
        self,
        vehicles: Sequence[Vehicle],
        orders: Sequence[Order],
        road_network: RoadNetwork,
    ) -> RoutePlan:
        """Solve a pickup-delivery VRP from the depot with OR-Tools."""

        try:
            from ortools.constraint_solver import pywrapcp, routing_enums_pb2
        except ImportError as error:
            raise ImportError(
                "ORToolsSolver requires the 'ortools' package. "
                "Install it with: pip install ortools"
            ) from error

        available_vehicles = [vehicle for vehicle in vehicles if vehicle.available]
        active_orders = [order for order in orders if not order.cancelled]
        routes = {vehicle.id: [] for vehicle in available_vehicles}

        if not available_vehicles:
            return RoutePlan(
                routes={},
                total_distance=0.0,
                total_travel_time=0.0,
                unassigned_order_ids=[order.id for order in active_orders],
            )

        if not active_orders:
            return RoutePlan(routes=routes)

        depot_id = self._depot_id(road_network)
        edge_lookup = {
            (edge.fromnode, edge.tonode): edge for edge in road_network.edges
        }
        visit_nodes, order_visit_indices = self._build_visit_nodes(
            depot_id,
            active_orders,
        )

        manager = pywrapcp.RoutingIndexManager(
            len(visit_nodes),
            len(available_vehicles),
            0,
        )
        routing = pywrapcp.RoutingModel(manager)

        def distance_callback(from_index: int, to_index: int) -> int:
            from_visit = visit_nodes[manager.IndexToNode(from_index)]
            to_visit = visit_nodes[manager.IndexToNode(to_index)]
            distance = self._distance(edge_lookup, from_visit.node_id, to_visit.node_id)
            return self._scale_distance(distance)

        distance_callback_index = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(distance_callback_index)

        def time_callback(from_index: int, to_index: int) -> int:
            from_visit = visit_nodes[manager.IndexToNode(from_index)]
            to_visit = visit_nodes[manager.IndexToNode(to_index)]
            travel_time = self._travel_time(
                edge_lookup,
                from_visit.node_id,
                to_visit.node_id,
            )
            return self._minutes_to_time_units(travel_time)

        time_callback_index = routing.RegisterTransitCallback(time_callback)
        planning_start = self._planning_start(active_orders)
        horizon = self._time_horizon(active_orders, edge_lookup, visit_nodes)
        routing.AddDimension(
            time_callback_index,
            horizon,
            horizon,
            False,
            "Time",
        )
        time_dimension = routing.GetDimensionOrDie("Time")
        self._apply_time_windows(
            manager,
            routing,
            time_dimension,
            visit_nodes,
            planning_start,
            horizon,
        )

        def weight_demand_callback(from_index: int) -> int:
            return visit_nodes[manager.IndexToNode(from_index)].weight_demand

        def volume_demand_callback(from_index: int) -> int:
            return visit_nodes[manager.IndexToNode(from_index)].volume_demand

        weight_callback_index = routing.RegisterUnaryTransitCallback(
            weight_demand_callback,
        )
        volume_callback_index = routing.RegisterUnaryTransitCallback(
            volume_demand_callback,
        )
        routing.AddDimensionWithVehicleCapacity(
            weight_callback_index,
            0,
            [self._scale_capacity(vehicle.capacityweight) for vehicle in available_vehicles],
            True,
            "Weight",
        )
        routing.AddDimensionWithVehicleCapacity(
            volume_callback_index,
            0,
            [self._scale_capacity(vehicle.capacityvolume) for vehicle in available_vehicles],
            True,
            "Volume",
        )

        for pickup_node, delivery_node in order_visit_indices.values():
            pickup_index = manager.NodeToIndex(pickup_node)
            delivery_index = manager.NodeToIndex(delivery_node)
            routing.AddPickupAndDelivery(pickup_index, delivery_index)
            routing.solver().Add(
                routing.VehicleVar(pickup_index)
                == routing.VehicleVar(delivery_index),
            )
            routing.solver().Add(
                time_dimension.CumulVar(pickup_index)
                <= time_dimension.CumulVar(delivery_index),
            )
            routing.solver().Add(
                routing.ActiveVar(pickup_index) == routing.ActiveVar(delivery_index),
            )
            routing.AddDisjunction([pickup_index], self.drop_penalty)
            routing.AddDisjunction([delivery_index], self.drop_penalty)

        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        )
        search_parameters.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        )
        time_limit_ms = max(1, int(round(self.time_limit_seconds * 1000)))
        search_parameters.time_limit.seconds = time_limit_ms // 1000
        search_parameters.time_limit.nanos = (time_limit_ms % 1000) * 1_000_000

        solution = routing.SolveWithParameters(search_parameters)
        if solution is None:
            return RoutePlan(
                routes=routes,
                total_distance=0.0,
                total_travel_time=0.0,
                unassigned_order_ids=[order.id for order in active_orders],
            )

        return self._build_route_plan(
            manager,
            routing,
            solution,
            time_dimension,
            available_vehicles,
            active_orders,
            visit_nodes,
            order_visit_indices,
            edge_lookup,
        )

    def _build_route_plan(
        self,
        manager,
        routing,
        solution,
        time_dimension,
        vehicles: Sequence[Vehicle],
        orders: Sequence[Order],
        visit_nodes: Sequence[_VisitNode],
        order_visit_indices: dict[str, tuple[int, int]],
        edge_lookup,
    ) -> RoutePlan:
        routes: dict[str, list[RouteStop]] = {vehicle.id: [] for vehicle in vehicles}
        total_distance = 0.0
        total_travel_time = 0.0

        for vehicle_index, vehicle in enumerate(vehicles):
            index = routing.Start(vehicle_index)
            while not routing.IsEnd(index):
                next_index = solution.Value(routing.NextVar(index))
                from_visit = visit_nodes[manager.IndexToNode(index)]
                to_visit = visit_nodes[manager.IndexToNode(next_index)]

                travel_distance = self._distance(
                    edge_lookup,
                    from_visit.node_id,
                    to_visit.node_id,
                )
                travel_time = self._travel_time(
                    edge_lookup,
                    from_visit.node_id,
                    to_visit.node_id,
                )
                total_distance += travel_distance
                total_travel_time += travel_time

                if not routing.IsEnd(next_index) and to_visit.stop_type != "depot":
                    arrival_time = (
                        solution.Value(time_dimension.CumulVar(next_index)) / 60.0
                    )
                    routes[vehicle.id].append(
                        RouteStop(
                            node_id=to_visit.node_id,
                            stop_type=to_visit.stop_type,
                            order_id=to_visit.order_id,
                            arrival_time=arrival_time,
                            departure_time=arrival_time,
                            travel_distance=travel_distance,
                            travel_time=travel_time,
                        )
                    )

                index = next_index

        unassigned_order_ids: list[str] = []
        for order in orders:
            pickup_node, _ = order_visit_indices[order.id]
            pickup_index = manager.NodeToIndex(pickup_node)
            if solution.Value(routing.ActiveVar(pickup_index)) == 0:
                unassigned_order_ids.append(order.id)

        return RoutePlan(
            routes=routes,
            total_distance=total_distance,
            total_travel_time=total_travel_time,
            unassigned_order_ids=unassigned_order_ids,
        )

    def _build_visit_nodes(
        self,
        depot_id: str,
        orders: Sequence[Order],
    ) -> tuple[list[_VisitNode], dict[str, tuple[int, int]]]:
        visit_nodes = [_VisitNode(node_id=depot_id, stop_type="depot")]
        order_visit_indices: dict[str, tuple[int, int]] = {}

        for order in orders:
            pickup_index = len(visit_nodes)
            visit_nodes.append(
                _VisitNode(
                    node_id=order.pickup,
                    stop_type="pickup",
                    order_id=order.id,
                    weight_demand=self._scale_capacity(order.weight),
                    volume_demand=self._scale_capacity(order.volume),
                    time_window=order.pickupwindow,
                )
            )

            delivery_index = len(visit_nodes)
            visit_nodes.append(
                _VisitNode(
                    node_id=order.delivery,
                    stop_type="delivery",
                    order_id=order.id,
                    weight_demand=-self._scale_capacity(order.weight),
                    volume_demand=-self._scale_capacity(order.volume),
                    time_window=order.deliverywindow,
                )
            )
            order_visit_indices[order.id] = (pickup_index, delivery_index)

        return visit_nodes, order_visit_indices

    def _apply_time_windows(
        self,
        manager,
        routing,
        time_dimension,
        visit_nodes: Sequence[_VisitNode],
        planning_start: int,
        horizon: int,
    ) -> None:
        for node_index, visit_node in enumerate(visit_nodes):
            if visit_node.stop_type == "depot":
                continue

            routing_index = manager.NodeToIndex(node_index)
            if routing_index < 0 or visit_node.time_window is None:
                continue

            time_dimension.CumulVar(routing_index).SetRange(
                self._hour_to_time_units(visit_node.time_window.starthour),
                self._hour_to_time_units(visit_node.time_window.endhour),
            )

        for vehicle_index in range(routing.vehicles()):
            time_dimension.CumulVar(routing.Start(vehicle_index)).SetRange(
                planning_start,
                horizon,
            )
            time_dimension.CumulVar(routing.End(vehicle_index)).SetRange(
                planning_start,
                horizon,
            )

    def _time_horizon(
        self,
        orders: Sequence[Order],
        edge_lookup,
        visit_nodes: Sequence[_VisitNode],
    ) -> int:
        latest_window_end = max(
            max(
                self._hour_to_time_units(order.pickupwindow.endhour),
                self._hour_to_time_units(order.deliverywindow.endhour),
            )
            for order in orders
        )
        max_arc_minutes = 0.0
        for from_visit in visit_nodes:
            for to_visit in visit_nodes:
                max_arc_minutes = max(
                    max_arc_minutes,
                    self._travel_time(edge_lookup, from_visit.node_id, to_visit.node_id),
                )
        route_margin = self._minutes_to_time_units(max_arc_minutes * len(visit_nodes))
        return max(latest_window_end + route_margin, latest_window_end + 24 * 60)

    def _planning_start(self, orders: Sequence[Order]) -> int:
        earliest = min(
            min(
                order.created_at,
                order.pickupwindow.starthour,
                order.deliverywindow.starthour,
            )
            for order in orders
        )
        return self._hour_to_time_units(earliest)

    def _depot_id(self, road_network: RoadNetwork) -> str:
        if road_network.depot_id:
            return road_network.depot_id

        for node in road_network.nodes:
            if node.type == "depot":
                return node.id

        raise ValueError("road_network must define a depot_id or a depot node")

    def _distance(self, edge_lookup, from_node: str, to_node: str) -> float:
        if from_node == to_node:
            return 0.0

        edge = edge_lookup.get((from_node, to_node))
        if edge is None:
            return 1_000_000.0

        return edge.distance

    def _travel_time(self, edge_lookup, from_node: str, to_node: str) -> float:
        if from_node == to_node:
            return 0.0

        edge = edge_lookup.get((from_node, to_node))
        if edge is None:
            return 1_000_000.0

        return edge.effective_travel_time()

    def _scale_distance(self, distance: float) -> int:
        return max(0, int(math.ceil(distance * self.distance_scale)))

    def _scale_capacity(self, capacity: float) -> int:
        return max(0, int(math.ceil(capacity * self.capacity_scale)))

    def _hour_to_time_units(self, hour: float) -> int:
        return max(0, int(round(hour * 60.0)))

    def _minutes_to_time_units(self, minutes: float) -> int:
        return max(0, int(math.ceil(minutes)))
