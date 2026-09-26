from ddm_mcts.agents import AdaptiveLayaMCTSAgent, AdaptiveSearchConfig
from ddm_mcts.environments import DelayedReward
from ddm_mcts.policies import Policy


class OrderSensitivePolicy(Policy):
    def __init__(self):
        self.calls = 0

    def probabilities(self, state, legal_actions):
        del state
        self.calls += 1
        weights = {action: 0.01 for action in legal_actions}
        weights[legal_actions[0]] = 0.98
        total = sum(weights.values())
        return {action: value / total for action, value in weights.items()}


def test_adaptive_agent_detects_order_instability_and_reduces_alpha():
    env = DelayedReward()
    policy = OrderSensitivePolicy()
    config = AdaptiveSearchConfig(minimum_simulations=2, maximum_simulations=5, order_samples=2, seed=3)
    decision = AdaptiveLayaMCTSAgent(env, policy, config).decide(env.initial_state())
    assert decision.simulations >= 2
    assert decision.metadata["order_stability"] == "LOW"
    assert decision.metadata["alpha"] <= 0.25
    assert decision.metadata["stability_changed_alpha"] is True
    assert decision.metadata["triggered_additional_search"] is True


def test_adaptive_agent_without_stability_still_runs_mandatory_search():
    env = DelayedReward()
    policy = OrderSensitivePolicy()
    config = AdaptiveSearchConfig(minimum_simulations=3, maximum_simulations=5, order_stability_check=False)
    decision = AdaptiveLayaMCTSAgent(env, policy, config).decide(env.initial_state())
    assert decision.simulations >= 3
    assert decision.metadata["order_comparisons"] == []
    assert policy.calls >= 1
