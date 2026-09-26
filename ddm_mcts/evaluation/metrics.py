from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(slots=True)
class AgentMetrics:
    wins: int = 0
    losses: int = 0
    draws: int = 0
    decisions: int = 0
    latency_seconds: float = 0.0
    simulations: int = 0

    @property
    def games(self) -> int:
        return self.wins + self.losses + self.draws

    def to_dict(self) -> dict[str, int | float]:
        result = asdict(self)
        result.update(
            win_rate=self.wins / self.games if self.games else 0.0,
            average_decision_latency_ms=(1000 * self.latency_seconds / self.decisions if self.decisions else 0.0),
            average_simulations_per_move=(self.simulations / self.decisions if self.decisions else 0.0),
        )
        return result
