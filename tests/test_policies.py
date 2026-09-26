from ddm_mcts.environments import ConnectFour
from ddm_mcts.policies import LayaPolicy, RandomPolicy, normalize_probabilities


def test_probabilities_normalize_and_filter_illegal_actions():
    assert normalize_probabilities({0: 2, 1: 1, 99: 100}, [0, 1]) == {0: 2 / 3, 1: 1 / 3}
    probs = RandomPolicy(seed=1).probabilities(object(), [1, 2, 3])
    assert abs(sum(probs.values()) - 1) < 1e-12


def test_laya_official_response_shape_and_cache_without_runtime():
    calls = []

    def predictor(state, questions):
        calls.append((state, questions))
        return {
            "answers": {
                "move": {
                    "type": "choice",
                    "choice": "column_3",
                    "probabilities": {"column_0": 0.2, "column_3": 0.8, "column_9": 1.0},
                }
            }
        }

    env = ConnectFour()
    policy = LayaPolicy(predictor=predictor)
    state = env.initial_state()
    assert policy.probabilities(state, [0, 3]) == {0: 0.2, 3: 0.8}
    policy.probabilities(state, [0, 3])
    assert len(calls) == 1
    stats = policy.diagnostics()
    assert stats["total_policy_requests"] == 2
    assert stats["actual_inference_calls"] == 1
    assert stats["cache_hits"] == 1


def test_malformed_laya_response_falls_back_to_explicit_uniform_prior(caplog):
    env = ConnectFour()
    policy = LayaPolicy(predictor=lambda state, questions: {"unexpected": True})
    assert policy.probabilities(env.initial_state(), [1, 4]) == {1: 0.5, 4: 0.5}
    assert "uniform prior" in caplog.text
