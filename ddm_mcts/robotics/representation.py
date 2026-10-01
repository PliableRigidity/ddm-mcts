"""Small immutable decision representation, separate from simulator snapshots."""

from dataclasses import dataclass, replace

from .observation import Entity, SemanticGoal
from .reach import ReachState, ReachTask


@dataclass(frozen=True)
class WorldState:
    robot: ReachState
    entities: tuple[Entity, ...]
    semantic_goal: SemanticGoal | None = None
    source: str = "ground-truth"
    resolved_target_label: str | None = None
    planning_goal: tuple[float, float, float] | None = None

    def target(self) -> Entity:
        if self.semantic_goal is None:
            return Entity("goal", self.robot.goal)
        label = self.resolved_target_label or self.semantic_goal.label
        matches = [entity for entity in self.entities if entity.label == label]
        if len(matches) != 1:
            raise ValueError(f"expected one target for {self.semantic_goal.label!r}, found {len(matches)}")
        return matches[0]

    @property
    def goal(self):
        return self.planning_goal if self.planning_goal is not None else self.target().position

    @property
    def position(self):
        return self.robot.position

    @property
    def steps(self):
        return self.robot.steps

    @property
    def time(self):
        return self.robot.time

    def predict_observation(self, simulator_state):
        """Keep perceived entities fixed; predict robot motion with the world model.

        This simulation implementation explicitly uses privileged robot dynamics.
        It does not render or re-perceive speculative futures.
        """
        return replace(self, robot=replace(simulator_state, goal=self.goal))


@dataclass(frozen=True)
class SemanticReachTask:
    goal: SemanticGoal
    epsilon: float = 0.012
    max_steps: int = 30

    def resolve(self, state: WorldState) -> ReachTask:
        return ReachTask(state.target().position, self.epsilon, self.max_steps)


@dataclass(frozen=True)
class CoordinateReachTask:
    """Explicit ground-truth observation path for the original XYZ reach task."""

    goal: None = None
    epsilon: float = 0.012
    max_steps: int = 30

    def resolve(self, state: WorldState) -> ReachTask:
        return ReachTask(state.goal, self.epsilon, self.max_steps)
