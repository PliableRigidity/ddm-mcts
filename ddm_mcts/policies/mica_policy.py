from __future__ import annotations

import json
import time
from collections.abc import Callable, Hashable, Sequence
from typing import Any, Generic, TypeVar
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .base import Policy, normalize_probabilities

StateT = TypeVar("StateT", bound=Hashable)
ActionT = TypeVar("ActionT", bound=Hashable)


class MicaUnavailableError(RuntimeError):
    """Raised when the official Mica SystemOne service cannot be reached."""


class MicaPolicy(Policy[StateT, ActionT], Generic[StateT, ActionT]):
    """Direct choice-probability adapter for Mica's official SystemOne API.

    Mica is intentionally used as a scorer: this adapter never requests or parses
    generated prose. The presentation-order-sensitive cache key is explicit.
    """

    def __init__(
        self,
        state_text: Callable[[StateT], str],
        *,
        objective: str,
        action_text: Callable[[ActionT], str] = str,
        endpoint: str = "http://127.0.0.1:8010/v1/systemone",
        model: str = "mica-v0.1-4b",
        timeout: float = 120.0,
        requester: Callable[[str, dict[str, Any], float], dict[str, Any]] | None = None,
    ) -> None:
        self.state_text, self.objective, self.action_text = state_text, objective, action_text
        self.endpoint, self.model, self.timeout = endpoint, model, timeout
        self._requester = requester or self._http_request
        self._cache: dict[tuple[str, str, StateT, tuple[ActionT, ...]], dict[ActionT, float]] = {}
        self.requests = self.calls = self.hits = self.misses = self.errors = 0
        self.inference_seconds = self.cache_seconds = 0.0
        self.last_raw_output: dict[str, Any] | None = None
        self.last_error: str | None = None

    @staticmethod
    def _http_request(endpoint: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
        request = Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=timeout) as response:  # noqa: S310 - configured local/service endpoint
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            try:
                detail = exc.read().decode("utf-8", errors="replace")
            except OSError:
                detail = ""
            raise MicaUnavailableError(
                f"Mica endpoint {endpoint!r} rejected the request with HTTP {exc.code}: {detail or exc.reason}"
            ) from exc
        except (URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            raise MicaUnavailableError(f"Mica endpoint {endpoint!r} is unavailable: {exc}") from exc

    def probabilities(self, state: StateT, legal_actions: Sequence[ActionT]) -> dict[ActionT, float]:
        self.requests += 1
        legal = tuple(legal_actions)
        if not legal:
            return {}
        if len(legal) == 1:
            return {legal[0]: 1.0}
        key = (self.model, self.endpoint, state, legal)
        started = time.perf_counter()
        if key in self._cache:
            self.hits += 1
            self.cache_seconds += time.perf_counter() - started
            return dict(self._cache[key])
        self.misses += 1
        criteria = {f"option_{index}": self.action_text(action) for index, action in enumerate(legal)}
        payload = {
            "model": self.model,
            "state": self.state_text(state),
            "questions": {"action": {"type": "choice", "instructions": self.objective, "criteria": criteria}},
        }
        inference_started = time.perf_counter()
        self.calls += 1
        try:
            result = self._requester(self.endpoint, payload, self.timeout)
            self.inference_seconds += time.perf_counter() - inference_started
            self.last_raw_output, self.last_error = result, None
            raw = result["answers"]["action"]["probabilities"]
            probabilities = normalize_probabilities(
                {action: raw.get(f"option_{index}", 0.0) for index, action in enumerate(legal)}, legal
            )
        except MicaUnavailableError:
            self.inference_seconds += time.perf_counter() - inference_started
            self.errors += 1
            raise
        except Exception as exc:
            self.inference_seconds += time.perf_counter() - inference_started
            self.errors += 1
            self.last_error = str(exc)
            raise MicaUnavailableError(f"Mica returned an invalid choice response: {exc}") from exc
        self._cache[key] = dict(probabilities)
        return probabilities

    def diagnostics(self) -> dict[str, Any]:
        return {
            "backend": "mica",
            "model": self.model,
            "endpoint": self.endpoint,
            "total_policy_requests": self.requests,
            "actual_inference_calls": self.calls,
            "cache_hits": self.hits,
            "cache_misses": self.misses,
            "cache_hit_rate": self.hits / self.requests if self.requests else 0.0,
            "total_inference_seconds": self.inference_seconds,
            "average_inference_seconds": self.inference_seconds / self.calls if self.calls else 0.0,
            "average_cache_lookup_seconds": self.cache_seconds / self.hits if self.hits else 0.0,
            "inference_errors": self.errors,
            "cached_presentations": len(self._cache),
            "last_error": self.last_error,
        }
