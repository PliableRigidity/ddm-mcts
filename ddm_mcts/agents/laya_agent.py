from __future__ import annotations

import random
from typing import Generic, TypeVar

from ddm_mcts.agents.base import Agent, Decision
from ddm_mcts.environments.base import Environment
from ddm_mcts.policies.base import Policy

StateT = TypeVar("StateT")
ActionT = TypeVar("ActionT")


class PolicyAgent(Agent[StateT, ActionT], Generic[StateT, ActionT]):
    def __init__(
        self,
        environment: Environment[StateT, ActionT],
        policy: Policy[StateT, ActionT],
        *,
        name: str = "policy",
        seed: int | None = None,
    ) -> None:
        self.environment, self.policy, self.name = environment, policy, name
        self._rng = random.Random(seed)

    def decide(self, state: StateT) -> Decision[ActionT]:
        legal = self.environment.legal_actions(state)
        probs = self.policy.probabilities(state, legal)
        best = max(probs.values())
        candidates = [action for action, probability in probs.items() if probability == best]
        return Decision(self._rng.choice(candidates), statistics=[{"action": a, "prior": p} for a, p in probs.items()])


class LayaAgent(PolicyAgent[StateT, ActionT]):
    def __init__(self, environment: Environment[StateT, ActionT], policy: Policy[StateT, ActionT], seed: int | None = None) -> None:
        super().__init__(environment, policy, name="laya", seed=seed)
