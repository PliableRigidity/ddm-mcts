from __future__ import annotations

from dataclasses import dataclass

from .base import Environment

GridAction = str
MOVES = {"UP": (-1, 0), "DOWN": (1, 0), "LEFT": (0, -1), "RIGHT": (0, 1)}


@dataclass(frozen=True, slots=True)
class GridState:
    position: tuple[int, int]
    goal: tuple[int, int]
    obstacles: frozenset[tuple[int, int]]
    height: int
    width: int
    steps: int = 0
    max_steps: int = 30


class GridNavigation(Environment[GridState, GridAction]):
    def __init__(self, initial: GridState | None = None) -> None:
        self._initial = initial or GridState((0, 0), (4, 4), frozenset({(0, 3), (1, 1), (1, 3), (2, 1), (3, 2)}), 5, 5)

    def initial_state(self) -> GridState:
        return self._initial

    def legal_actions(self, state: GridState) -> tuple[GridAction, ...]:
        if self.is_terminal(state):
            return ()
        legal = []
        for action, (dr, dc) in MOVES.items():
            target = (state.position[0] + dr, state.position[1] + dc)
            if 0 <= target[0] < state.height and 0 <= target[1] < state.width and target not in state.obstacles:
                legal.append(action)
        return tuple(legal)

    def step(self, state: GridState, action: GridAction) -> GridState:
        if action not in self.legal_actions(state):
            raise ValueError(f"illegal movement: {action}")
        dr, dc = MOVES[action]
        return GridState((state.position[0] + dr, state.position[1] + dc), state.goal, state.obstacles, state.height, state.width, state.steps + 1, state.max_steps)

    def is_terminal(self, state: GridState) -> bool:
        return state.position == state.goal or state.steps >= state.max_steps

    def get_reward(self, state: GridState, player: int) -> float:
        del player
        if not self.is_terminal(state):
            return 0.0
        return 1.0 - 0.02 * state.steps if state.position == state.goal else -1.0

    def current_player(self, state: GridState) -> int:
        del state
        return 1

    def render(self, state: GridState) -> str:
        rows = []
        for row in range(state.height):
            cells = []
            for col in range(state.width):
                point = (row, col)
                cells.append("A" if point == state.position else "G" if point == state.goal else "#" if point in state.obstacles else ".")
            rows.append(" ".join(cells))
        return "\n".join(rows) + f"\nSteps: {state.steps}/{state.max_steps}"


def grid_scenarios() -> dict[str, GridState]:
    return {
        "simple": GridState((0, 0), (2, 2), frozenset(), 3, 3, max_steps=10),
        "obstacles": GridNavigation().initial_state(),
        "misleading_path": GridState((2, 0), (2, 4), frozenset({(1, 1), (2, 1), (2, 3), (3, 3)}), 5, 5, max_steps=20),
        "move_away_first": GridState((1, 1), (1, 4), frozenset({(1, 2), (0, 2), (0, 3)}), 4, 5, max_steps=20),
        "narrow_gate": GridState((0, 0), (4, 4), frozenset({(0, 2), (1, 0), (1, 2), (3, 2), (4, 2)}), 5, 5, max_steps=24),
        "bottom_detour": GridState((0, 0), (0, 4), frozenset({(0, 1), (1, 1), (1, 2), (1, 3)}), 4, 5, max_steps=20),
        "two_corridors": GridState((2, 0), (2, 5), frozenset({(1, 2), (2, 2), (3, 2), (1, 4), (3, 4)}), 5, 6, max_steps=24),
        "spiral": GridState((0, 0), (2, 2), frozenset({(0, 1), (1, 1), (1, 3), (2, 3), (3, 1), (3, 2), (3, 3)}), 5, 5, max_steps=24),
        "open_rectangle": GridState((3, 0), (0, 5), frozenset(), 4, 6, max_steps=16),
        "central_wall": GridState((2, 0), (2, 4), frozenset({(1, 2), (2, 2), (3, 2)}), 5, 5, max_steps=18),
    }
