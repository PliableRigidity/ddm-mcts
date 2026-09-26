import pytest

from ddm_mcts.environments.connect_four import ConnectFour, ConnectFourState


def play(env, actions):
    state = env.initial_state()
    for action in actions:
        state = env.step(state, action)
    return state


def test_legal_moves_and_piece_placement():
    env = ConnectFour()
    state = play(env, [0, 0, 0, 0, 0, 0])
    assert 0 not in env.legal_actions(state)
    assert state.at(5, 0) == 1
    assert state.at(4, 0) == 2
    with pytest.raises(ValueError):
        env.step(state, 0)


@pytest.mark.parametrize(
    "actions",
    [
        [0, 0, 1, 1, 2, 2, 3],  # horizontal
        [0, 1, 0, 1, 0, 1, 0],  # vertical
        [0, 1, 1, 2, 4, 2, 2, 3, 4, 3, 5, 3, 3],  # rising diagonal
        [3, 2, 2, 1, 4, 1, 1, 0, 4, 0, 5, 0, 0],  # falling diagonal
    ],
)
def test_wins(actions):
    env = ConnectFour()
    state = play(env, actions)
    assert env.is_terminal(state)
    assert state.winner == 1
    assert env.get_reward(state, 1) == 1
    assert env.get_reward(state, 2) == -1


def test_known_draw_is_terminal():
    # Full board with no four in a row.
    rows = (
        (1, 1, 2, 2, 1, 1, 2),
        (2, 2, 1, 1, 2, 2, 1),
        (1, 1, 2, 2, 1, 1, 2),
        (2, 2, 1, 1, 2, 2, 1),
        (1, 1, 2, 2, 1, 1, 2),
        (2, 2, 1, 1, 2, 2, 1),
    )
    state = ConnectFourState(tuple(cell for row in rows for cell in row), player=1, moves=42)
    env = ConnectFour()
    assert env.is_terminal(state)
    assert env.get_reward(state, 1) == 0
    assert env.legal_actions(state) == ()
