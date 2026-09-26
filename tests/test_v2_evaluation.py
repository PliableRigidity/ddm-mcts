from ddm_mcts.environments import DelayedReward, Job, JobScheduling
from ddm_mcts.evaluation.v2_experiment import Scenario, _adaptive_ablation, optimal_action_analysis, option_order_metrics, regret
from ddm_mcts.policies import MixedPolicy, TextLayaPolicy


def test_multiple_optimal_actions_are_retained():
    env = JobScheduling((Job("A", 1, 10, 5), Job("B", 1, 10, 5)))
    actions, value, action_values = optimal_action_analysis(env, env.initial_state())
    assert actions == {"A", "B"}
    assert action_values == {"A": 10, "B": 10}
    assert value == 10


def test_regret_is_nonnegative_and_none_without_known_optimum():
    assert regret(10, 7) == 3
    assert regret(10, 10.00000000001) == 0
    assert regret(None, 7) is None


def test_option_order_mapping_detects_semantic_consistency():
    rows = []
    for order, positions in (("normal", {"A": 0, "B": 1}), ("reversed", {"A": 1, "B": 0})):
        rows += [
            {"domain": "test", "scenario_id": "one", "order": order, "action": "A", "presentation_position": positions["A"], "probability": 0.8, "rank": 1},
            {"domain": "test", "scenario_id": "one", "order": order, "action": "B", "presentation_position": positions["B"], "probability": 0.2, "rank": 2},
        ]
    result = option_order_metrics(rows)
    assert result["top_action_consistency"] == 1
    assert result["mean_absolute_probability_change"] == 0
    assert result["mean_rank_correlation"] == 1


def test_laya_cache_is_reused_across_alpha_values():
    calls = []

    def predictor(state, questions):
        calls.append((state, questions))
        return {"answers": {"action": {"probabilities": {"option_0": 0.75, "option_1": 0.25}}}}

    policy = TextLayaPolicy(str, objective="choose", predictor=predictor)
    for alpha in (0.25, 0.5, 0.75, 1.0):
        MixedPolicy(policy, alpha).probabilities("state", ["A", "B"])
    assert len(calls) == 1
    assert policy.diagnostics()["cache_hits"] == 3


def test_adaptive_ablation_tracks_extra_order_requests_and_trace():
    def predictor(state, questions):
        labels = list(questions["action"]["criteria"])
        probabilities = {label: 0.05 for label in labels}
        probabilities[labels[0]] = 0.9
        return {"answers": {"action": {"probabilities": probabilities}}}

    env = DelayedReward()
    scenario = Scenario("delayed_reward", "Delayed Reward", "test", env, env.initial_state(), "maximize", "reward")
    rows = _adaptive_ablation([scenario], predictor, "quick", 4, 2)
    without = next(row for row in rows if not row["order_stability_check"])
    checked = next(row for row in rows if row["order_stability_check"])
    assert checked["laya_requests"] > without["laya_requests"]
    assert checked["decision_trace"]["order_comparisons"]
    assert checked["simulations"] >= without["simulations"]
