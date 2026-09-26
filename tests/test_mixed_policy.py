import pytest

from ddm_mcts.policies import MixedPolicy, Policy


class StaticPolicy(Policy):
    def __init__(self):
        self.calls = 0

    def probabilities(self, state, legal_actions):
        self.calls += 1
        return {0: 0.8, 1: 0.2, 99: 100}


def test_alpha_zero_is_uniform_and_bypasses_base_policy():
    base = StaticPolicy()
    assert MixedPolicy(base, 0).probabilities(None, [0, 1]) == {0: 0.5, 1: 0.5}
    assert base.calls == 0


def test_alpha_one_matches_legal_normalized_policy():
    result = MixedPolicy(StaticPolicy(), 1).probabilities(None, [0, 1])
    assert result == pytest.approx({0: 0.8, 1: 0.2})
    assert 99 not in result


def test_half_alpha_mixes_and_normalizes():
    result = MixedPolicy(StaticPolicy(), 0.5).probabilities(None, [0, 1])
    assert result == pytest.approx({0: 0.65, 1: 0.35})
    assert sum(result.values()) == pytest.approx(1)


def test_invalid_alpha_rejected():
    with pytest.raises(ValueError):
        MixedPolicy(StaticPolicy(), 1.1)
