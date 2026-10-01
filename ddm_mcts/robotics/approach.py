"""Perceived object center -> outside-object pre-contact reaching objective."""

from dataclasses import dataclass, replace
from math import isfinite, sqrt

from .observation import SemanticGoal
from .reach import ReachTask


@dataclass(frozen=True)
class ApproachReachTask:
    goal: SemanticGoal
    epsilon: float = 0.012
    max_steps: int = 30
    standoff: float = 0.05  # beyond conservative object bounding sphere
    object_radius: float = 0.04  # configured extent prior, NOT simulator truth
    approach_vector: tuple[float, float, float] = (0.0, 0.0, 1.0)

    def __post_init__(self):
        if not all(isfinite(v) for v in (*self.approach_vector, self.standoff, self.object_radius)):
            raise ValueError("approach parameters must be finite")
        if self.standoff <= self.epsilon or self.object_radius <= 0 or len(self.approach_vector) != 3:
            raise ValueError("positive object radius and standoff greater than reach tolerance required")
        if sum(v * v for v in self.approach_vector) < 1e-12:
            raise ValueError("nonzero approach vector required")

    def prepare_state(self, state):
        center = state.target().position
        norm = sqrt(sum(v * v for v in self.approach_vector))
        point = tuple(c + v / norm * (self.object_radius + self.standoff) for c, v in zip(center, self.approach_vector, strict=True))
        # Clear the object-height band vertically before translating over it.
        # Stateless staging: speculative futures retain the current waypoint;
        # the live observe/plan loop resolves the next waypoint after execution.
        if self.approach_vector == (0.0, 0.0, 1.0) and state.position[2] < point[2] - self.epsilon:
            point = (state.position[0], state.position[1], point[2])
        return replace(state, planning_goal=point)

    def resolve(self, state):
        return ReachTask(self.prepare_state(state).goal, self.epsilon, self.max_steps)

    def diagnostics(self, state):
        norm = sqrt(sum(v * v for v in self.approach_vector))
        return {
            "object_center": state.target().position,
            "object_extent_source": "configured conservative bounding-radius prior",
            "object_bounding_radius_m": self.object_radius,
            "object_estimated_diameter_m": 2 * self.object_radius,
            "approach_vector": tuple(v / norm for v in self.approach_vector),
            "standoff_m": self.standoff,
            "pre_contact_target": tuple(
                c + v / norm * (self.object_radius + self.standoff)
                for c, v in zip(state.target().position, self.approach_vector, strict=True)
            ),
            "requested_waypoint": self.prepare_state(state).goal,
        }
