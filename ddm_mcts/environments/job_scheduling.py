from __future__ import annotations

from dataclasses import dataclass

from .base import Environment


@dataclass(frozen=True, slots=True)
class Job:
    name: str
    duration: int
    deadline: int
    value: float


@dataclass(frozen=True, slots=True)
class SchedulingState:
    remaining: tuple[Job, ...]
    time: int = 0
    reward: float = 0.0
    completed: tuple[str, ...] = ()


class JobScheduling(Environment[SchedulingState, str]):
    def __init__(self, jobs: tuple[Job, ...] | None = None, lateness_penalty: float = 2.0) -> None:
        self.jobs = jobs or (Job("A", 2, 3, 8), Job("B", 1, 6, 3), Job("C", 3, 5, 10))
        self.lateness_penalty = lateness_penalty

    def initial_state(self) -> SchedulingState:
        return SchedulingState(self.jobs)

    def legal_actions(self, state: SchedulingState) -> tuple[str, ...]:
        return tuple(job.name for job in state.remaining)

    def step(self, state: SchedulingState, action: str) -> SchedulingState:
        try:
            job = next(job for job in state.remaining if job.name == action)
        except StopIteration as exc:
            raise ValueError(f"unknown or completed job: {action}") from exc
        finish = state.time + job.duration
        earned = job.value - self.lateness_penalty * max(0, finish - job.deadline)
        return SchedulingState(tuple(item for item in state.remaining if item.name != action), finish, state.reward + earned, state.completed + (action,))

    def is_terminal(self, state: SchedulingState) -> bool:
        return not state.remaining

    def get_reward(self, state: SchedulingState, player: int) -> float:
        del player
        return state.reward if self.is_terminal(state) else 0.0

    def current_player(self, state: SchedulingState) -> int:
        del state
        return 1

    def render(self, state: SchedulingState) -> str:
        jobs = ", ".join(f"{j.name}(duration={j.duration}, deadline={j.deadline}, value={j.value})" for j in state.remaining)
        return f"Time: {state.time}; reward so far: {state.reward}; remaining: {jobs or 'none'}"


def scheduling_scenarios() -> dict[str, tuple[Job, ...]]:
    return {
        "deadline_tradeoff": (Job("A", 2, 3, 8), Job("B", 1, 6, 3), Job("C", 3, 5, 10)),
        "short_urgent": (Job("urgent", 1, 1, 5), Job("valuable", 4, 5, 14), Job("small", 2, 3, 4)),
        "lateness_penalty": (Job("long", 4, 4, 13), Job("early", 2, 2, 8), Job("flexible", 1, 8, 2)),
        "two_urgent": (Job("A", 2, 2, 7), Job("B", 2, 3, 8), Job("C", 4, 9, 11)),
        "value_vs_deadline": (Job("A", 5, 5, 18), Job("B", 1, 2, 5), Job("C", 2, 4, 7)),
        "equal_values": (Job("A", 1, 2, 6), Job("B", 2, 3, 6), Job("C", 3, 7, 6)),
        "four_jobs": (Job("A", 1, 1, 4), Job("B", 2, 4, 7), Job("C", 2, 5, 6), Job("D", 3, 8, 10)),
        "late_high_value": (Job("A", 4, 3, 16), Job("B", 1, 1, 5), Job("C", 2, 5, 8)),
        "slack": (Job("A", 2, 8, 5), Job("B", 3, 9, 9), Job("C", 1, 7, 3)),
        "tight_cluster": (Job("A", 2, 2, 8), Job("B", 2, 3, 9), Job("C", 2, 4, 10)),
    }
