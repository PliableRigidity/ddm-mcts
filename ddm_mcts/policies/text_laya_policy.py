from __future__ import annotations

import logging
import time
from collections.abc import Callable, Hashable, Sequence
from typing import Any, Generic, TypeVar

from .base import Policy, normalize_probabilities
from .laya_policy import LayaUnavailableError

LOGGER = logging.getLogger(__name__)
StateT = TypeVar("StateT", bound=Hashable)
ActionT = TypeVar("ActionT", bound=Hashable)


class TextLayaPolicy(Policy[StateT, ActionT], Generic[StateT, ActionT]):
    """Generic official-Laya adapter for text-renderable decision environments."""

    def __init__(
        self,
        state_text: Callable[[StateT], str],
        *,
        objective: str,
        action_text: Callable[[ActionT], str] = str,
        device: str | None = None,
        model: str | None = None,
        predictor: Callable[[Any, dict[str, Any]], dict[str, Any]] | None = None,
    ) -> None:
        self.state_text, self.objective, self.action_text = state_text, objective, action_text
        self.device, self.model, self._predictor = device, model, predictor
        self._router: Any = None
        self._cache: dict[tuple[StateT, tuple[ActionT, ...]], dict[ActionT, float]] = {}
        self.requests = self.calls = self.hits = self.misses = self.errors = 0
        self.inference_seconds = self.cache_seconds = 0.0
        self.last_raw_output: dict[str, Any] | None = None

    def _ensure_predictor(self):
        if self._predictor is not None:
            return self._predictor
        try:
            from laya import Router
        except ImportError as exc:
            raise LayaUnavailableError('Laya is not installed. Run: pip install -e ".[laya]"') from exc
        try:
            kwargs: dict[str, Any] = {"preload": False}
            if self.device:
                kwargs["device"] = self.device
            self._router = Router(**kwargs)
            self._predictor = self._router.predict
            return self._predictor
        except Exception as exc:
            raise LayaUnavailableError(f"Laya could not initialize: {exc}") from exc

    def probabilities(self, state: StateT, legal_actions: Sequence[ActionT]) -> dict[ActionT, float]:
        self.requests += 1
        legal = tuple(legal_actions)
        if not legal:
            return {}
        key = (state, legal)
        started = time.perf_counter()
        if key in self._cache:
            self.hits += 1
            self.cache_seconds += time.perf_counter() - started
            return dict(self._cache[key])
        self.misses += 1
        labels = {f"option_{index}": self.action_text(action) for index, action in enumerate(legal)}
        question = {"action": {"type": "choice", "instructions": self.objective, "criteria": labels}}
        try:
            predictor = self._ensure_predictor()
            inference_started = time.perf_counter()
            self.calls += 1
            if self.model and self._router is not None:
                result = self._router.predict(self.state_text(state), question, model=self.model)
            else:
                result = predictor(self.state_text(state), question)
            self.inference_seconds += time.perf_counter() - inference_started
            self.last_raw_output = result
            raw = result["answers"]["action"]["probabilities"]
            probabilities = normalize_probabilities({action: raw.get(f"option_{index}", 0.0) for index, action in enumerate(legal)}, legal)
        except LayaUnavailableError:
            raise
        except Exception as exc:
            self.errors += 1
            LOGGER.warning("Laya evaluation failed; using an explicit uniform prior: %s", exc)
            probabilities = normalize_probabilities({}, legal)
        self._cache[key] = dict(probabilities)
        return probabilities

    def diagnostics(self) -> dict[str, int | float]:
        return {
            "total_policy_requests": self.requests,
            "actual_inference_calls": self.calls,
            "cache_hits": self.hits,
            "cache_misses": self.misses,
            "cache_hit_rate": self.hits / self.requests if self.requests else 0.0,
            "total_inference_seconds": self.inference_seconds,
            "average_inference_seconds": self.inference_seconds / self.calls if self.calls else 0.0,
            "average_cache_lookup_seconds": self.cache_seconds / self.hits if self.hits else 0.0,
            "inference_errors": self.errors,
        }
