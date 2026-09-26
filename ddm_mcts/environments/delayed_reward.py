from __future__ import annotations

from dataclasses import dataclass

from .base import Environment


@dataclass(frozen=True, slots=True)
class DelayedRewardState:
    node: str = "START"
    cumulative_reward: float = 0.0
    path: tuple[str, ...] = ()


class DelayedReward(Environment[DelayedRewardState, str]):
    """Small exact tree where attractive immediate rewards can be globally poor."""

    default_transitions = {
        "START": {"A_IMMEDIATE": ("A2", 10), "B_DELAYED": ("B2", 1), "C_SAFE": ("C_END", 30)},
        "A2": {"CONTINUE_A": ("A_END", -50)},
        "B2": {"CONTINUE_B": ("B3", 1)},
        "B3": {"FINISH_B": ("B_END", 100)},
    }
    def __init__(self, transitions: dict[str, dict[str, tuple[str, float]]] | None = None) -> None:
        self.transitions = transitions or self.default_transitions

    def initial_state(self) -> DelayedRewardState:
        return DelayedRewardState()

    def legal_actions(self, state: DelayedRewardState) -> tuple[str, ...]:
        return tuple(self.transitions.get(state.node, {}))

    def step(self, state: DelayedRewardState, action: str) -> DelayedRewardState:
        try:
            target, reward = self.transitions[state.node][action]
        except KeyError as exc:
            raise ValueError(f"invalid action {action} at {state.node}") from exc
        return DelayedRewardState(target, state.cumulative_reward + reward, state.path + (action,))

    def is_terminal(self, state: DelayedRewardState) -> bool:
        return state.node.endswith("_END")

    def get_reward(self, state: DelayedRewardState, player: int) -> float:
        del player
        return state.cumulative_reward if self.is_terminal(state) else 0.0

    def current_player(self, state: DelayedRewardState) -> int:
        del state
        return 1

    def render(self, state: DelayedRewardState) -> str:
        return f"Decision node: {state.node}; accumulated reward: {state.cumulative_reward}; path: {state.path or 'none'}"


def delayed_reward_scenarios() -> dict[str, dict[str, dict[str, tuple[str, float]]]]:
    scenarios = {}
    for index in range(10):
        immediate = 8 + index
        delayed = 70 + index * 5
        safe = 20 + (index % 4) * 5
        scenarios[f"delayed_{index + 1:02d}"] = {
            "START": {"A_IMMEDIATE": ("A2", immediate), "B_DELAYED": ("B2", -2 + index % 3), "C_SAFE": ("C_END", safe)},
            "A2": {"CONTINUE_A": ("A_END", -35 - index * 2)},
            "B2": {"CONTINUE_B": ("B3", 1)},
            "B3": {"FINISH_B": ("B_END", delayed)},
        }
    return scenarios
