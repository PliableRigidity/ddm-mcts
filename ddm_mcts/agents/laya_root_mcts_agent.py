from __future__ import annotations

from typing import Generic, TypeVar

from ddm_mcts.agents.base import Agent, Decision
from ddm_mcts.environments.base import Environment
from ddm_mcts.policies.base import Policy
from ddm_mcts.policies.mixed_policy import MixedPolicy
from ddm_mcts.policies.random_policy import UniformPolicy
from ddm_mcts.search.mcts import MCTS, MCTSConfig

StateT = TypeVar("StateT")
ActionT = TypeVar("ActionT")


class LayaRootMCTSAgent(Agent[StateT, ActionT], Generic[StateT, ActionT]):
    """Use a DDM prior at the real root only; descendants receive uniform priors."""

    def __init__(self, environment: Environment[StateT, ActionT], policy: Policy[StateT, ActionT], simulations: int = 100, seed: int | None = None, *, alpha: float = 1.0) -> None:
        self.name = "laya-root-mcts"
        self.alpha = alpha
        self.searcher = MCTS(environment, UniformPolicy(), MCTSConfig(simulations=simulations, seed=seed), root_policy=MixedPolicy(policy, alpha))

    def decide(self, state: StateT) -> Decision[ActionT]:
        result = self.searcher.search(state)
        return Decision(result.action, result.simulations, result.root_statistics())
