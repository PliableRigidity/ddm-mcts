from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from ddm_mcts.agents.base import Agent, Decision
from ddm_mcts.environments.base import Environment
from ddm_mcts.policies.base import Policy
from ddm_mcts.policies.mixed_policy import MixedPolicy
from ddm_mcts.policies.random_policy import UniformPolicy
from ddm_mcts.search.mcts import MCTS, MCTSConfig

StateT = TypeVar("StateT")
ActionT = TypeVar("ActionT")


@dataclass(frozen=True, slots=True)
class AdaptiveSearchConfig:
    minimum_simulations: int = 10
    maximum_simulations: int = 50
    alpha_choices: tuple[float, ...] = (0.0, 0.25, 0.5)
    order_stability_check: bool = True
    order_samples: int = 2
    probability_change_threshold: float = 0.10
    rank_correlation_threshold: float = 0.50
    seed: int = 0

    def __post_init__(self) -> None:
        if self.minimum_simulations < 1 or self.maximum_simulations < self.minimum_simulations:
            raise ValueError("adaptive simulation bounds are invalid")
        if self.order_samples < 2 and self.order_stability_check:
            raise ValueError("order stability checking requires at least 2 total samples")
        if not self.alpha_choices or any(not 0 <= alpha <= 1 for alpha in self.alpha_choices):
            raise ValueError("alpha choices must be non-empty and within [0, 1]")


def _rank_correlation(original: dict[Any, float], measured: dict[Any, float]) -> float:
    actions = tuple(original)
    if len(actions) < 2:
        return 1.0
    first = {action: rank for rank, action in enumerate(sorted(actions, key=original.get, reverse=True), 1)}
    second = {action: rank for rank, action in enumerate(sorted(actions, key=measured.get, reverse=True), 1)}
    n = len(actions)
    return 1 - 6 * sum((first[action] - second[action]) ** 2 for action in actions) / (n * (n * n - 1))


class AdaptiveLayaMCTSAgent(Agent[StateT, ActionT], Generic[StateT, ActionT]):
    """Conservative root-guided MCTS with optional order-stability gating."""

    name = "adaptive-laya-mcts"

    def __init__(self, environment: Environment[StateT, ActionT], policy: Policy[StateT, ActionT], config: AdaptiveSearchConfig | None = None) -> None:
        self.environment, self.policy = environment, policy
        self.config = config or AdaptiveSearchConfig()

    def decide(self, state: StateT) -> Decision[ActionT]:
        legal = tuple(self.environment.legal_actions(state))
        original = self.policy.probabilities(state, legal)
        ranking = sorted(original, key=original.get, reverse=True)
        laya_top = ranking[0]
        entropy = -sum(probability * math.log(probability) for probability in original.values() if probability > 0)
        normalized_entropy = entropy / math.log(len(original)) if len(original) > 1 else 0.0
        choices = tuple(sorted(self.config.alpha_choices))
        alpha = choices[-1] if normalized_entropy <= 0.35 else choices[len(choices) // 2] if normalized_entropy <= 0.70 else choices[0]
        comparisons: list[dict[str, Any]] = []
        rng = random.Random(self.config.seed)
        if self.config.order_stability_check:
            for sample in range(1, self.config.order_samples):
                order = list(legal)
                rng.shuffle(order)
                permuted = self.policy.probabilities(state, order)
                comparisons.append({
                    "sample": sample,
                    "order": order,
                    "probabilities": permuted,
                    "top_agrees": max(permuted, key=permuted.get) == laya_top,
                    "mean_absolute_change": sum(abs(permuted[action] - original[action]) for action in legal) / len(legal),
                    "rank_correlation": _rank_correlation(original, permuted),
                })
        stable = not comparisons or all(
            item["top_agrees"]
            and item["mean_absolute_change"] <= self.config.probability_change_threshold
            and item["rank_correlation"] >= self.config.rank_correlation_threshold
            for item in comparisons
        )
        alpha_before_stability = alpha
        original_minimum = self.config.minimum_simulations
        minimum = original_minimum
        if not stable:
            alpha = min(alpha, 0.25)
            minimum = max(minimum, min(10, self.config.maximum_simulations))
        initial = MCTS(
            self.environment,
            UniformPolicy(),
            MCTSConfig(simulations=minimum, seed=self.config.seed),
            root_policy=MixedPolicy(self.policy, alpha),
        ).search(state)
        escalate = (not stable or initial.action != laya_top) and minimum < self.config.maximum_simulations
        final = initial
        if escalate:
            final = MCTS(
                self.environment,
                UniformPolicy(),
                MCTSConfig(simulations=self.config.maximum_simulations, seed=self.config.seed),
                root_policy=MixedPolicy(self.policy, alpha),
            ).search(state)
        trace = {
            "laya_original": original,
            "laya_top_action": laya_top,
            "policy_entropy": entropy,
            "normalized_entropy": normalized_entropy,
            "order_comparisons": comparisons,
            "order_stability": "HIGH" if stable else "LOW",
            "alpha_before_stability": alpha_before_stability,
            "alpha": alpha,
            "stability_changed_alpha": alpha != alpha_before_stability,
            "stability_increased_minimum": minimum > original_minimum,
            "minimum_search": minimum,
            "initial_action": initial.action,
            "escalated": escalate,
            "triggered_additional_search": minimum > original_minimum or escalate,
            "final_simulations": self.config.maximum_simulations if escalate else minimum,
            "final_action": final.action,
        }
        return Decision(final.action, trace["final_simulations"], final.root_statistics(), metadata=trace)
