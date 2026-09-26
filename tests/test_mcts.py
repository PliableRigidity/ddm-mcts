from ddm_mcts.environments import ConnectFour
from ddm_mcts.policies.base import Policy
from ddm_mcts.search import MCTS, MCTSConfig


def play(actions):
    env = ConnectFour()
    state = env.initial_state()
    for action in actions:
        state = env.step(state, action)
    return env, state


def test_mcts_never_selects_illegal_move():
    env, state = play([0, 0, 0, 0, 0, 0])
    result = MCTS(env, config=MCTSConfig(simulations=20, seed=1)).search(state)
    assert result.action in env.legal_actions(state)


def test_mcts_finds_immediate_win():
    env, state = play([0, 1, 0, 1, 0, 2])
    result = MCTS(env, config=MCTSConfig(simulations=100, seed=2)).search(state)
    assert result.action == 0


def test_mcts_blocks_immediate_opponent_win():
    env, state = play([0, 3, 1, 3, 2, 3])
    result = MCTS(env, config=MCTSConfig(simulations=300, seed=3)).search(state)
    assert result.action == 3


class BiasedPolicy(Policy):
    def probabilities(self, state, legal_actions):
        return {action: (10.0 if action == 6 else 1.0) for action in legal_actions}


def test_guided_mcts_accepts_arbitrary_priors():
    env = ConnectFour()
    result = MCTS(env, BiasedPolicy(), MCTSConfig(simulations=10, seed=4)).search(env.initial_state())
    stats = {row["action"]: row for row in result.root_statistics()}
    assert stats[6]["prior"] > stats[0]["prior"]
    assert result.action in range(7)


class CountingPolicy(Policy):
    def __init__(self):
        self.calls = 0

    def probabilities(self, state, legal_actions):
        self.calls += 1
        return {action: 1 for action in legal_actions}


def test_root_policy_is_called_once_and_not_on_descendants():
    env = ConnectFour()
    root_policy = CountingPolicy()
    descendant_policy = CountingPolicy()
    MCTS(env, descendant_policy, MCTSConfig(simulations=20, seed=5), root_policy=root_policy).search(env.initial_state())
    assert root_policy.calls == 1
    assert descendant_policy.calls > 1
