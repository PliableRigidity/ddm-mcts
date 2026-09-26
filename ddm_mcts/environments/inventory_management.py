from __future__ import annotations

import random
from dataclasses import dataclass

from .base import Environment

InventoryAction = int


@dataclass(frozen=True, slots=True)
class InventoryState:
    inventory: int = 2
    timestep: int = 0
    horizon: int = 5
    total_cost: float = 0.0
    seed: int = 0
    demands: tuple[int, ...] = ()


class InventoryManagement(Environment[InventoryState, InventoryAction]):
    actions = (0, 2, 4, 6)

    def __init__(self, seed: int = 0, horizon: int = 5) -> None:
        self.seed, self.horizon = seed, horizon

    def initial_state(self) -> InventoryState:
        return InventoryState(horizon=self.horizon, seed=self.seed)

    def legal_actions(self, state: InventoryState) -> tuple[int, ...]:
        return () if self.is_terminal(state) else self.actions

    @staticmethod
    def _demand(state: InventoryState) -> int:
        return random.Random(state.seed * 10_007 + state.timestep * 997).choice((0, 1, 2, 3, 4, 5))

    def step(self, state: InventoryState, action: int) -> InventoryState:
        if action not in self.legal_actions(state):
            raise ValueError(f"invalid order quantity: {action}")
        demand = self._demand(state)
        available = state.inventory + action
        ending = max(0, available - demand)
        shortage = max(0, demand - available)
        cost = 0.5 * action + 0.25 * ending + 3.0 * shortage
        return InventoryState(ending, state.timestep + 1, state.horizon, state.total_cost + cost, state.seed, state.demands + (demand,))

    def is_terminal(self, state: InventoryState) -> bool:
        return state.timestep >= state.horizon

    def get_reward(self, state: InventoryState, player: int) -> float:
        del player
        return -state.total_cost if self.is_terminal(state) else 0.0

    def current_player(self, state: InventoryState) -> int:
        del state
        return 1

    def render(self, state: InventoryState) -> str:
        return f"Period {state.timestep}/{state.horizon}; inventory={state.inventory}; cost={state.total_cost:.2f}; observed demand={state.demands}"
