from __future__ import annotations

import math
import random
from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, TypeVar

from ddm_mcts.environments.base import Environment
from ddm_mcts.policies.base import Policy, normalize_probabilities
from ddm_mcts.policies.random_policy import UniformPolicy

from .node import Node

StateT = TypeVar("StateT")
ActionT = TypeVar("ActionT")
Evaluator = Callable[[StateT, int, random.Random], float]


@dataclass(frozen=True, slots=True)
class MCTSConfig:
    simulations: int = 100
    c_puct: float = 1.4
    seed: int | None = None


@dataclass(slots=True)
class SearchResult(Generic[ActionT, StateT]):
    action: ActionT
    root: Node[StateT, ActionT]
    simulations: int

    def root_statistics(self) -> list[dict[str, float | int | ActionT]]:
        return [
            {"action": action, "prior": child.prior, "visits": child.visit_count, "q": child.mean_value}
            for action, child in sorted(self.root.children.items(), key=lambda item: str(item[0]))
        ]


class MCTS(Generic[StateT, ActionT]):
    """Adversarial PUCT search with values stored from the root player's perspective."""

    def __init__(
        self,
        environment: Environment[StateT, ActionT],
        policy: Policy[StateT, ActionT] | None = None,
        config: MCTSConfig | None = None,
        evaluator: Evaluator[StateT] | None = None,
        root_policy: Policy[StateT, ActionT] | None = None,
    ) -> None:
        self.environment = environment
        self.policy = policy or UniformPolicy()
        self.config = config or MCTSConfig()
        self._rng = random.Random(self.config.seed)
        self.evaluator = evaluator or self._random_rollout
        self.root_policy = root_policy

    def _expand(self, node: Node[StateT, ActionT], policy: Policy[StateT, ActionT] | None = None) -> None:
        legal = self.environment.legal_actions(node.state)
        priors = normalize_probabilities((policy or self.policy).probabilities(node.state, legal), legal)
        node.children = {
            action: Node(
                state=self.environment.step(node.state, action),
                parent=node,
                action=action,
                prior=priors[action],
            )
            for action in legal
        }
        node.expanded = True

    def _select_child(self, node: Node[StateT, ActionT], root_player: int) -> Node[StateT, ActionT]:
        maximizing = self.environment.current_player(node.state) == root_player
        parent_scale = math.sqrt(max(1, node.visit_count))

        def score(child: Node[StateT, ActionT]) -> float:
            exploitation = child.mean_value if maximizing else -child.mean_value
            exploration = self.config.c_puct * child.prior * parent_scale / (1 + child.visit_count)
            return exploitation + exploration

        best = max(score(child) for child in node.children.values())
        candidates = [child for child in node.children.values() if abs(score(child) - best) < 1e-12]
        return self._rng.choice(candidates)

    def _random_rollout(self, state: StateT, root_player: int, rng: random.Random) -> float:
        current = state
        while not self.environment.is_terminal(current):
            current = self.environment.step(current, rng.choice(self.environment.legal_actions(current)))
        return self.environment.get_reward(current, root_player)

    def search(self, state: StateT, simulations: int | None = None) -> SearchResult[ActionT, StateT]:
        if self.environment.is_terminal(state):
            raise ValueError("cannot search a terminal state")
        budget = self.config.simulations if simulations is None else simulations
        if budget < 1:
            raise ValueError("simulations must be at least 1")
        root_player = self.environment.current_player(state)
        root: Node[StateT, ActionT] = Node(state=state)
        self._expand(root, self.root_policy)
        for _ in range(budget):
            node = root
            path = [node]
            while node.expanded and node.children:
                node = self._select_child(node, root_player)
                path.append(node)
            if self.environment.is_terminal(node.state):
                value = self.environment.get_reward(node.state, root_player)
            else:
                self._expand(node)
                value = self.evaluator(node.state, root_player, self._rng)
            for ancestor in path:
                ancestor.visit_count += 1
                ancestor.value_sum += value
        most_visits = max(child.visit_count for child in root.children.values())
        candidates = [a for a, child in root.children.items() if child.visit_count == most_visits]
        action = self._rng.choice(candidates)
        return SearchResult(action, root, budget)
