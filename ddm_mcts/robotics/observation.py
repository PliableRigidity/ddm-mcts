"""Raw observation contracts; ground truth remains an explicit input path."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class Entity:
    label: str
    position: tuple[float, float, float]
    pixel: tuple[float, float] | None = None
    pixels: int = 0
    observed: bool = True
    missed_frames: int = 0


@dataclass(frozen=True)
class SemanticGoal:
    label: str

    def __post_init__(self):
        if not self.label.strip():
            raise ValueError("goal label must be nonempty")


@dataclass(frozen=True)
class Observation:
    timestamp: float
    sequence: int
    source: str
    robot: Any
    entities: tuple[Entity, ...] = ()
    rgb: Any = None
    camera: Any = None
    metadata: tuple[tuple[str, Any], ...] = ()


class ObservationProvider(Protocol):
    def observe(self) -> Observation: ...


class GroundTruthObservationProvider:
    def __init__(self, world, entities=None):
        self.world = world
        self.entities = entities or (lambda: ())
        self.sequence = 0

    def observe(self) -> Observation:
        state = self.world.get_state()
        self.sequence += 1
        return Observation(state.time, self.sequence, "ground-truth", state, tuple(self.entities()))

    def close(self):
        pass
