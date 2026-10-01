"""Composable perception, including an optional model boundary (no model download)."""

from collections.abc import Callable
from typing import Protocol

from .observation import Observation, SemanticGoal
from .representation import WorldState


class PerceptionProvider(Protocol):
    def perceive(self, observation: Observation, goal: SemanticGoal | None = None) -> WorldState: ...


class GroundTruthPerception:
    def perceive(self, observation: Observation, goal: SemanticGoal | None = None) -> WorldState:
        if observation.source != "ground-truth":
            raise ValueError("ground-truth perception requires a ground-truth observation")
        state = WorldState(observation.robot, observation.entities, goal, observation.source)
        state.target()  # Resolve explicitly; fail instead of guessing an absent target.
        return state


class ModelPerceptionAdapter:
    """Future VLM boundary: image + goal -> structured state, via injected callable.

    The callable owns inference and schema conversion. No backend/network is implied.
    """

    def __init__(self, predictor: Callable[[Observation, SemanticGoal | None], WorldState]):
        self.predictor = predictor

    def perceive(self, observation: Observation, goal: SemanticGoal | None = None) -> WorldState:
        state = self.predictor(observation, goal)
        if not isinstance(state, WorldState):
            raise TypeError("model perception must return WorldState")
        if state.semantic_goal != goal:
            raise ValueError("model returned a mismatched goal")
        state.target()
        return state
