"""Bounded language interpretation, not unrestricted natural-language planning."""

import re
from dataclasses import dataclass

from .placement import NextTo, PlacementTarget


@dataclass(frozen=True)
class ManipulationOperation:
    target: str
    destination: NextTo | PlacementTarget | None = None

    def __post_init__(self):
        if self.target not in ("cube", "cylinder"):
            raise ValueError("only cube and cylinder are manipulable")
        if self.destination is not None and not isinstance(self.destination, (NextTo, PlacementTarget)):
            raise ValueError("unsupported destination")
        if isinstance(self.destination, NextTo) and self.target == self.destination.reference:
            raise ValueError("cannot place an object next to itself")


@dataclass(frozen=True)
class PhysicalManipulationTask:
    instruction: str
    operations: tuple[ManipulationOperation, ...]

    def __post_init__(self):
        if (
            not self.instruction.strip()
            or not self.operations
            or len(self.operations) > 4
            or not all(isinstance(op, ManipulationOperation) for op in self.operations)
        ):
            raise ValueError("one to four supported manipulation operations required")
        object.__setattr__(self, "operations", tuple(self.operations))


def parse_manipulation_task(instruction, *, target_position=(0.46, 0.10, 0.37)):
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("nonempty manipulation instruction required")
    text = re.sub(r"\s+", " ", instruction.lower().strip().rstrip("."))
    clauses = re.split(r",? then ", text)
    operations = []
    for clause in clauses:
        picked = re.fullmatch(r"pick up (?:the )?(cube|cylinder)", clause)
        if picked:
            operations.append(ManipulationOperation(picked[1]))
            continue
        match = re.fullmatch(
            r"(?:move|place|pick up) (?:the )?(cube|cylinder)(?: and place (?:it|the \1))? "
            r"(?:next to (?:the )?(cube|cylinder)|to (?:the )?(?:other )?target location)",
            clause,
        )
        if not match:
            raise ValueError(f"unsupported bounded manipulation clause: {clause!r}")
        destination = NextTo(match[2]) if match[2] else PlacementTarget(tuple(target_position))
        operations.append(ManipulationOperation(match[1], destination))
    if len(operations) > 1 and any(op.destination is None for op in operations):
        raise ValueError("multi-operation tasks require complete pick-and-place operations")
    return PhysicalManipulationTask(instruction, tuple(operations))
