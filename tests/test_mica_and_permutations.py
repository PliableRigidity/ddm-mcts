from __future__ import annotations

import pytest

from ddm_mcts.policies import MicaPolicy, MicaUnavailableError, MixedPolicy, PermutationAveragedPolicy, Policy


class PositionPolicy(Policy[str, str]):
    def __init__(self) -> None:
        self.calls: list[tuple[str, ...]] = []

    def probabilities(self, state: str, legal_actions: tuple[str, ...] | list[str]) -> dict[str, float]:
        legal = tuple(legal_actions)
        self.calls.append(legal)
        return {action: (0.8 if index == 0 else 0.2 / (len(legal) - 1)) for index, action in enumerate(legal)}


def test_mica_maps_presentation_labels_back_to_semantic_actions_and_caches() -> None:
    seen = []

    def requester(endpoint, payload, timeout):
        seen.append((endpoint, payload, timeout))
        return {"answers": {"action": {"type": "choice", "probabilities": {"option_0": 0.25, "option_1": 0.75}}}}

    policy = MicaPolicy(str, objective="Choose.", action_text=lambda value: f"Action {value}", requester=requester)
    assert policy.probabilities("state", ["right", "left"]) == {"right": 0.25, "left": 0.75}
    assert policy.probabilities("state", ["right", "left"]) == {"right": 0.25, "left": 0.75}
    assert seen[0][1]["questions"]["action"]["criteria"] == {"option_0": "Action right", "option_1": "Action left"}
    assert policy.diagnostics()["actual_inference_calls"] == 1
    assert policy.diagnostics()["cache_hits"] == 1


def test_mica_cache_separates_presentation_order() -> None:
    calls = 0

    def requester(_endpoint, _payload, _timeout):
        nonlocal calls
        calls += 1
        return {"answers": {"action": {"probabilities": {"option_0": 0.9, "option_1": 0.1}}}}

    policy = MicaPolicy(str, objective="Choose.", requester=requester)
    assert policy.probabilities("s", ["a", "b"])["a"] == pytest.approx(0.9)
    assert policy.probabilities("s", ["b", "a"])["b"] == pytest.approx(0.9)
    assert calls == 2


def test_mica_reports_malformed_or_unavailable_service() -> None:
    policy = MicaPolicy(str, objective="Choose.", requester=lambda *_: {})
    with pytest.raises(MicaUnavailableError, match="invalid choice response"):
        policy.probabilities("s", [1, 2])


def test_mica_single_legal_action_needs_no_server_call() -> None:
    policy = MicaPolicy(str, objective="Choose.", requester=lambda *_: pytest.fail("server should not be called"))
    assert policy.probabilities("s", ["only"]) == {"only": 1.0}
    assert policy.diagnostics()["actual_inference_calls"] == 0


def test_permutation_average_is_deterministic_unique_and_semantic() -> None:
    base = PositionPolicy()
    policy = PermutationAveragedPolicy(base, samples=3, seed=7)
    first = policy.probabilities("state", ["a", "b", "c"])
    assert len(base.calls) == len(set(base.calls)) == 3
    assert sum(first.values()) == pytest.approx(1.0)
    assert set(first) == {"a", "b", "c"}
    replay = PermutationAveragedPolicy(PositionPolicy(), samples=3, seed=7)
    assert replay.probabilities("state", ["a", "b", "c"]) == first
    assert replay.last_orders == policy.last_orders


def test_permutation_k1_is_exact_base_and_filters_illegal_values() -> None:
    base = PositionPolicy()
    averaged = PermutationAveragedPolicy(base, samples=1)
    assert averaged.probabilities("s", ["a", "b"]) == base.probabilities("s", ["a", "b"])
    assert set(averaged.probabilities("s", ["a", "b"])) == {"a", "b"}


def test_permutation_policy_composes_with_alpha_mixing() -> None:
    averaged = PermutationAveragedPolicy(PositionPolicy(), samples=2, seed=1)
    mixed = MixedPolicy(averaged, alpha=0.5)
    distribution = mixed.probabilities("s", ["a", "b", "c"])
    assert sum(distribution.values()) == pytest.approx(1.0)
    assert all(1 / 6 <= probability <= 2 / 3 for probability in distribution.values())


@pytest.mark.parametrize("samples", [0, 4])
def test_permutation_sample_bounds(samples: int) -> None:
    with pytest.raises(ValueError):
        PermutationAveragedPolicy(PositionPolicy(), samples=samples)
