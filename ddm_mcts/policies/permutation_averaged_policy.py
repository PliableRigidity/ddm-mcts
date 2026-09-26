from __future__ import annotations

import random
from collections.abc import Hashable, Sequence
from typing import Generic, TypeVar

from .base import Policy, normalize_probabilities

StateT = TypeVar("StateT", bound=Hashable)
ActionT = TypeVar("ActionT", bound=Hashable)


class PermutationAveragedPolicy(Policy[StateT, ActionT], Generic[StateT, ActionT]):
    """Average semantic-action probabilities over deterministic option orders."""

    def __init__(self, policy: Policy[StateT, ActionT], samples: int = 2, seed: int = 0) -> None:
        if not 1 <= samples <= 3:
            raise ValueError("samples must be between 1 and 3")
        self.policy, self.samples, self.seed = policy, samples, seed
        self._cache: dict[tuple[StateT, tuple[ActionT, ...], int, int], dict[ActionT, float]] = {}
        self.requests = self.policy_evaluations = self.hits = 0
        self.last_orders: tuple[tuple[ActionT, ...], ...] = ()

    def _orders(self, state: StateT, legal: tuple[ActionT, ...]) -> tuple[tuple[ActionT, ...], ...]:
        orders = [legal]
        if len(legal) < 2:
            return tuple(orders)
        # repr makes the seed stable across interpreter hash randomization.
        rng = random.Random(f"{self.seed}:{state!r}:{legal!r}")
        attempts = 0
        while len(orders) < min(self.samples, 3) and attempts < 50:
            candidate = list(legal)
            rng.shuffle(candidate)
            ordered = tuple(candidate)
            if ordered not in orders:
                orders.append(ordered)
            attempts += 1
        if len(orders) < self.samples:
            reversed_order = tuple(reversed(legal))
            if reversed_order not in orders:
                orders.append(reversed_order)
        return tuple(orders[: self.samples])

    def probabilities(self, state: StateT, legal_actions: Sequence[ActionT]) -> dict[ActionT, float]:
        self.requests += 1
        legal = tuple(legal_actions)
        if not legal:
            return {}
        key = (state, legal, self.samples, self.seed)
        if key in self._cache:
            self.hits += 1
            return dict(self._cache[key])
        orders = self._orders(state, legal)
        self.last_orders = orders
        totals = {action: 0.0 for action in legal}
        for order in orders:
            distribution = self.policy.probabilities(state, order)
            self.policy_evaluations += 1
            for action in legal:
                totals[action] += float(distribution.get(action, 0.0))
        averaged = normalize_probabilities({action: value / len(orders) for action, value in totals.items()}, legal)
        self._cache[key] = dict(averaged)
        return averaged

    def diagnostics(self) -> dict[str, object]:
        base = getattr(self.policy, "diagnostics", lambda: {})()
        return {
            **base,
            "permutation_samples": self.samples,
            "permutation_requests": self.requests,
            "permutation_policy_evaluations": self.policy_evaluations,
            "permutation_cache_hits": self.hits,
        }
