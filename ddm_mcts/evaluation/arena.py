from __future__ import annotations

import csv
import json
import logging
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Generic, TypeVar

from ddm_mcts.agents.base import Agent
from ddm_mcts.environments.base import Environment

from .metrics import AgentMetrics

LOGGER = logging.getLogger(__name__)
StateT = TypeVar("StateT")
ActionT = TypeVar("ActionT")


@dataclass(slots=True)
class BenchmarkResult:
    metadata: dict[str, object]
    agent_a: AgentMetrics
    agent_b: AgentMetrics
    average_game_length: float

    def to_dict(self) -> dict[str, object]:
        return {
            "metadata": self.metadata,
            "agent_a": self.agent_a.to_dict(),
            "agent_b": self.agent_b.to_dict(),
            "average_game_length": self.average_game_length,
        }

    def export(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.suffix.lower() == ".json":
            destination.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        elif destination.suffix.lower() == ".csv":
            rows = []
            for label, metrics in (("agent_a", self.agent_a), ("agent_b", self.agent_b)):
                rows.append({"agent": label, **self.metadata, **metrics.to_dict(), "average_game_length": self.average_game_length})
            with destination.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        else:
            raise ValueError("result path must end in .json or .csv")


class Arena(Generic[StateT, ActionT]):
    def __init__(self, environment: Environment[StateT, ActionT]) -> None:
        self.environment = environment

    def run(
        self,
        agent_a: Agent[StateT, ActionT],
        agent_b: Agent[StateT, ActionT],
        games: int,
        *,
        seed: int | None = None,
        verbose: bool = False,
        metadata: dict[str, object] | None = None,
    ) -> BenchmarkResult:
        if games < 1:
            raise ValueError("games must be at least 1")
        metrics = [AgentMetrics(), AgentMetrics()]
        lengths: list[int] = []
        for game_index in range(games):
            state = self.environment.initial_state()
            # Agent A starts even games, B starts odd games.
            seats = (agent_a, agent_b) if game_index % 2 == 0 else (agent_b, agent_a)
            seat_to_metric = (0, 1) if game_index % 2 == 0 else (1, 0)
            moves = 0
            while not self.environment.is_terminal(state):
                seat = 0 if self.environment.current_player(state) == 1 else 1
                started = time.perf_counter()
                decision = seats[seat].decide(state)
                elapsed = time.perf_counter() - started
                target = metrics[seat_to_metric[seat]]
                target.decisions += 1
                target.latency_seconds += elapsed
                target.simulations += decision.simulations
                if verbose:
                    LOGGER.info("game=%d move=%d agent=%s stats=%s", game_index, moves, seats[seat].name, decision.statistics)
                state = self.environment.step(state, decision.action)
                moves += 1
            lengths.append(moves)
            if self.environment.get_reward(state, 1) == 0:
                metrics[0].draws += 1
                metrics[1].draws += 1
            else:
                winning_seat = 0 if self.environment.get_reward(state, 1) == 1 else 1
                winner_metric = seat_to_metric[winning_seat]
                metrics[winner_metric].wins += 1
                metrics[1 - winner_metric].losses += 1
        info = {
            "agent_a": agent_a.name,
            "agent_b": agent_b.name,
            "games": games,
            "seed": seed,
            "timestamp": datetime.now(UTC).isoformat(),
            **(metadata or {}),
        }
        return BenchmarkResult(info, metrics[0], metrics[1], sum(lengths) / len(lengths))
