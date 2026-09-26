from __future__ import annotations

import math
from typing import Generic, TypeVar

from ddm_mcts.agents.base import Agent, Decision
from ddm_mcts.environments.base import Environment
from ddm_mcts.policies.base import Policy
from ddm_mcts.policies.mixed_policy import MixedPolicy
from ddm_mcts.search.mcts import MCTS, MCTSConfig

StateT = TypeVar("StateT")
ActionT = TypeVar("ActionT")


class LayaMCTSAgent(Agent[StateT, ActionT], Generic[StateT, ActionT]):
    def __init__(
        self,
        environment: Environment[StateT, ActionT],
        policy: Policy[StateT, ActionT],
        simulations: int = 100,
        seed: int | None = None,
        *,
        adaptive: bool = False,
        min_simulations: int = 10,
        max_simulations: int = 500,
        alpha: float = 1.0,
    ) -> None:
        self.name = "laya-mcts"
        self.environment, self.policy = environment, policy
        self.alpha = alpha
        self.adaptive = adaptive
        self.min_simulations, self.max_simulations = min_simulations, max_simulations
        self.searcher = MCTS(environment, MixedPolicy(policy, alpha), MCTSConfig(simulations=simulations, seed=seed))

    def _budget(self, state: StateT) -> int:
        if not self.adaptive:
            return self.searcher.config.simulations
        legal = self.environment.legal_actions(state)
        probs = self.policy.probabilities(state, legal)
        if len(probs) <= 1:
            return self.min_simulations
        entropy = -sum(p * math.log(p) for p in probs.values() if p > 0)
        normalized = entropy / math.log(len(probs))
        return round(self.min_simulations + normalized * (self.max_simulations - self.min_simulations))

    def decide(self, state: StateT) -> Decision[ActionT]:
        result = self.searcher.search(state, self._budget(state))
        return Decision(result.action, result.simulations, result.root_statistics())
