from __future__ import annotations

from typing import Generic, TypeVar

from ddm_mcts.agents.base import Agent, Decision
from ddm_mcts.environments.base import Environment
from ddm_mcts.policies.random_policy import RandomPolicy, UniformPolicy
from ddm_mcts.search.mcts import MCTS, MCTSConfig

from .laya_agent import PolicyAgent

StateT = TypeVar("StateT")
ActionT = TypeVar("ActionT")


class MCTSAgent(Agent[StateT, ActionT], Generic[StateT, ActionT]):
    def __init__(self, environment: Environment[StateT, ActionT], simulations: int = 100, seed: int | None = None) -> None:
        self.name = "mcts"
        self.searcher = MCTS(environment, UniformPolicy(), MCTSConfig(simulations=simulations, seed=seed))

    def decide(self, state: StateT) -> Decision[ActionT]:
        result = self.searcher.search(state)
        return Decision(result.action, result.simulations, result.root_statistics())


class RandomAgent(PolicyAgent[StateT, ActionT]):
    def __init__(self, environment: Environment[StateT, ActionT], seed: int | None = None) -> None:
        super().__init__(environment, RandomPolicy(seed), name="random", seed=seed)
