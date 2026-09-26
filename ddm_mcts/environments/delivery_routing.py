from __future__ import annotations

from dataclasses import dataclass

from .base import Environment


@dataclass(frozen=True, slots=True)
class RoutingState:
    current: str
    unvisited: tuple[str, ...]
    cost: float = 0.0
    route: tuple[str, ...] = ("Depot",)


class DeliveryRouting(Environment[RoutingState, str]):
    def __init__(self, distances: dict[tuple[str, str], float] | None = None) -> None:
        self.distances = distances or _default_distances()
        self.destinations = tuple(sorted({node for pair in self.distances for node in pair if node != "Depot"}))

    def _distance(self, source: str, target: str) -> float:
        try:
            return self.distances[(source, target)]
        except KeyError:
            return self.distances[(target, source)]

    def initial_state(self) -> RoutingState:
        return RoutingState("Depot", self.destinations)

    def legal_actions(self, state: RoutingState) -> tuple[str, ...]:
        if self.is_terminal(state):
            return ()
        return state.unvisited if state.unvisited else ("Depot",)

    def step(self, state: RoutingState, action: str) -> RoutingState:
        if action not in self.legal_actions(state):
            raise ValueError(f"invalid next destination: {action}")
        remaining = tuple(node for node in state.unvisited if node != action)
        return RoutingState(action, remaining, state.cost + self._distance(state.current, action), state.route + (action,))

    def is_terminal(self, state: RoutingState) -> bool:
        return not state.unvisited and state.current == "Depot"

    def get_reward(self, state: RoutingState, player: int) -> float:
        del player
        return -state.cost if self.is_terminal(state) else 0.0

    def current_player(self, state: RoutingState) -> int:
        del state
        return 1

    def render(self, state: RoutingState) -> str:
        return f"At {state.current}; unvisited: {', '.join(state.unvisited) or 'none'}; travel cost: {state.cost}; route: {' -> '.join(state.route)}"


def _default_distances() -> dict[tuple[str, str], float]:
    return {("Depot", "A"): 2, ("Depot", "B"): 3, ("Depot", "C"): 8, ("A", "B"): 7, ("A", "C"): 3, ("B", "C"): 2}


def routing_scenarios() -> dict[str, dict[tuple[str, str], float]]:
    scenarios = {
        "nearest_is_not_global": _default_distances(),
        "balanced": {("Depot", "A"): 4, ("Depot", "B"): 5, ("Depot", "C"): 6, ("A", "B"): 2, ("A", "C"): 5, ("B", "C"): 2},
    }
    for index in range(8):
        nodes = ("Depot", "A", "B", "C")
        coords = {node: ((position * 3 + index) % 7, (position * position + 2 * index) % 9) for position, node in enumerate(nodes)}
        distances = {}
        for left_index, left in enumerate(nodes):
            for right in nodes[left_index + 1 :]:
                dx, dy = coords[left][0] - coords[right][0], coords[left][1] - coords[right][1]
                distances[(left, right)] = round((dx * dx + dy * dy) ** 0.5 + 1, 2)
        scenarios[f"graph_{index + 1:02d}"] = distances
    return scenarios
