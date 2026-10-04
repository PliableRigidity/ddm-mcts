"""Bounded language semantics expressed as physical goals, never task handlers."""

import re
from dataclasses import dataclass
from enum import StrEnum


class Predicate(StrEnum):
    HELD = "held"
    SUPPORTED_BY = "supported_by"
    ON_TOP_OF = "on_top_of"
    NEXT_TO = "next_to"
    CONTACTING = "contacting"
    UPRIGHT = "upright"
    STABLE = "stable"
    TOPPLED = "toppled"
    RELEASED = "released"
    AT_LOCATION = "at_location"


@dataclass(frozen=True)
class PhysicalGoal:
    predicate: Predicate
    subject: str
    reference: str | None = None
    location: tuple[float, float, float] | None = None

    def __post_init__(self):
        object.__setattr__(self, "predicate", Predicate(self.predicate))
        if (
            self.predicate in (Predicate.ON_TOP_OF, Predicate.SUPPORTED_BY, Predicate.CONTACTING, Predicate.NEXT_TO)
            and self.reference is None
        ):
            raise ValueError("relational predicate requires a reference")
        if self.predicate == Predicate.AT_LOCATION:
            from math import isfinite

            if self.location is None or len(self.location) != 3 or not all(isfinite(v) for v in self.location):
                raise ValueError("AT_LOCATION requires finite XYZ")
        if self.subject not in ("cube", "cylinder", "tower"):
            raise ValueError("unsupported physical entity")
        if self.reference is not None and self.reference not in ("cube", "cylinder", "table"):
            raise ValueError("unsupported reference")
        if self.subject == self.reference:
            raise ValueError("self relation is invalid")


@dataclass(frozen=True)
class TemporalGoal:
    duration: float

    def __post_init__(self):
        if not 0 < self.duration <= 10:
            raise ValueError("wait duration must be in (0, 10] seconds")


@dataclass(frozen=True)
class GoalConjunction:
    predicates: tuple[PhysicalGoal, ...]


@dataclass(frozen=True)
class GoalSequence:
    instruction: str
    goals: tuple[PhysicalGoal | TemporalGoal | GoalConjunction, ...]


def parse_physical_goals(instruction):
    text = instruction.lower().strip()
    text = re.sub(r"\bonce that is done\b", "", text)
    text = re.sub(r"\bthen\b", "", text)
    text = re.sub(r"\b(?:the|a)\b", "", text)
    text = re.sub(r"\s+", " ", text)
    # "and place it" is an anaphoric placement clause, not a goal delimiter.
    text = re.sub(r"pick up (cube|cylinder) and (?:place|put) it", r"place \1", text)
    clauses = [c.strip() for c in re.split(r"(?<!\d)\.|\.(?!\d)|[,;]|\band\b", text) if c.strip()]
    goals = []
    for clause in clauses:
        clause = re.sub(r"\s+", " ", clause)
        if match := re.fullmatch(r"(?:put|place|move) (cube|cylinder) (?:on|on top of) (cube|cylinder)", clause):
            obj, support = match.groups()
            goals.append(
                GoalConjunction(
                    tuple(
                        PhysicalGoal(p, obj, support if p != Predicate.STABLE else None)
                        for p in (Predicate.ON_TOP_OF, Predicate.SUPPORTED_BY, Predicate.STABLE)
                    )
                )
            )
        elif match := re.fullmatch(r"(?:put|place|move) (cube|cylinder) next to (cube|cylinder)", clause):
            goals.append(PhysicalGoal(Predicate.NEXT_TO, *match.groups()))
        elif match := re.fullmatch(r"pick up (cube|cylinder)", clause):
            goals.append(PhysicalGoal(Predicate.HELD, match[1]))
        elif match := re.fullmatch(r"(?:topple (cube|cylinder|tower)|push (cube|cylinder) over)", clause):
            goals.append(PhysicalGoal(Predicate.TOPPLED, match[1] or match[2]))
        elif match := re.fullmatch(r"wait (?:for )?(\d+(?:\.\d+)?) seconds?", clause):
            goals.append(TemporalGoal(float(match[1])))
        elif match := re.fullmatch(r"move (cube|cylinder) to (?:other )?target location", clause):
            goals.append(PhysicalGoal(Predicate.AT_LOCATION, match[1], location=(0.46, 0.10, 0.37 if match[1] == "cube" else 0.375)))
        else:
            raise ValueError(f"unsupported physical goal clause: {clause!r}")
    if not goals or len(goals) > 5:
        raise ValueError("one to five bounded goals required")
    return GoalSequence(instruction, tuple(goals))
