from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence
from typing import Any

from ddm_mcts.environments.connect_four import COLS, ROWS, ConnectFourState

from .base import Policy, normalize_probabilities

LOGGER = logging.getLogger(__name__)


class LayaUnavailableError(RuntimeError):
    """Raised when the optional Laya runtime cannot provide an evaluation."""


class LayaPolicy(Policy[ConnectFourState, int]):
    """Laya choice probabilities exposed as a generic policy prior.

    The official API is loaded lazily so importing ddm-mcts never downloads model weights.
    An injected predictor makes parsing testable without installing Laya.
    """

    def __init__(
        self,
        *,
        device: str | None = None,
        model: str | None = None,
        cache: bool = True,
        predictor: Callable[[Any, dict[str, Any]], dict[str, Any]] | None = None,
    ) -> None:
        self.device = device
        self.model = model
        self.cache_enabled = cache
        self._predictor = predictor
        self._router: Any = None
        self._cache: dict[tuple[ConnectFourState, tuple[int, ...]], dict[int, float]] = {}
        self.total_requests = 0
        self.inference_calls = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.inference_seconds = 0.0
        self.cache_lookup_seconds = 0.0
        self.inference_errors = 0
        self.last_raw_output: dict[str, Any] | None = None
        self.last_error: str | None = None

    def diagnostics(self) -> dict[str, int | float]:
        """Return measured cache and inference counters for experiment reports."""
        return {
            "total_policy_requests": self.total_requests,
            "actual_inference_calls": self.inference_calls,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": self.cache_hits / self.total_requests if self.total_requests else 0.0,
            "total_inference_seconds": self.inference_seconds,
            "average_inference_seconds": (
                self.inference_seconds / self.inference_calls if self.inference_calls else 0.0
            ),
            "total_cache_lookup_seconds": self.cache_lookup_seconds,
            "average_cache_lookup_seconds": (
                self.cache_lookup_seconds / self.cache_hits if self.cache_hits else 0.0
            ),
            "inference_errors": self.inference_errors,
            "cached_states": len(self._cache),
        }

    def _ensure_predictor(self) -> Callable[[Any, dict[str, Any]], dict[str, Any]]:
        if self._predictor is not None:
            return self._predictor
        try:
            from laya import Router
        except ImportError as exc:
            raise LayaUnavailableError(
                'Laya is not installed. Run: pip install -e ".[laya]"'
            ) from exc
        try:
            kwargs = {"preload": False}
            if self.device:
                kwargs["device"] = self.device
            self._router = Router(**kwargs)
            self._predictor = self._router.predict
            return self._predictor
        except Exception as exc:
            raise LayaUnavailableError(f"Laya could not initialize: {exc}") from exc

    @staticmethod
    def _state_text(state: ConnectFourState) -> str:
        symbols = {0: ".", 1: "X", 2: "O"}
        rows = ["".join(symbols[state.at(r, c)] for c in range(COLS)) for r in range(ROWS)]
        return (
            "Connect Four board, top row first; . empty, X player 1, O player 2.\n"
            + "\n".join(rows)
            + f"\nPlayer {state.player} ({symbols[state.player]}) must move."
        )

    def probabilities(self, state: ConnectFourState, legal_actions: Sequence[int]) -> dict[int, float]:
        self.total_requests += 1
        legal = tuple(legal_actions)
        if not legal:
            return {}
        key = (state, legal)
        lookup_started = time.perf_counter()
        if self.cache_enabled and key in self._cache:
            self.last_error = None
            self.cache_hits += 1
            self.cache_lookup_seconds += time.perf_counter() - lookup_started
            return dict(self._cache[key])
        self.cache_misses += 1
        labels = {f"column_{c}": f"Drop the piece in legal column {c}." for c in legal}
        questions = {
            "move": {
                "type": "choice",
                "instructions": "Which legal move is most likely to lead the current player to win?",
                "criteria": labels,
            }
        }
        try:
            predictor = self._ensure_predictor()
            inference_started = time.perf_counter()
            self.inference_calls += 1
            if self.model and self._router is not None:
                result = self._router.predict(self._state_text(state), questions, model=self.model)
            else:
                result = predictor(self._state_text(state), questions)
            self.inference_seconds += time.perf_counter() - inference_started
            inference_started = None
            self.last_raw_output = result
            self.last_error = None
            raw = result["answers"]["move"]["probabilities"]
            parsed = {c: raw.get(f"column_{c}", 0.0) for c in legal}
            probs = normalize_probabilities(parsed, legal)
        except LayaUnavailableError:
            raise
        except Exception as exc:
            if "inference_started" in locals() and inference_started is not None:
                self.inference_seconds += time.perf_counter() - inference_started
            self.inference_errors += 1
            self.last_error = str(exc)
            # An already configured model may occasionally return a malformed result or fail a
            # single inference. Keep long benchmarks alive, but make the uniform fallback explicit.
            LOGGER.warning("Laya evaluation failed; using a uniform prior for this state: %s", exc)
            probs = normalize_probabilities({}, legal)
        if self.cache_enabled:
            self._cache[key] = dict(probs)
        return probs
