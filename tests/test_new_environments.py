import pytest

from ddm_mcts.environments.delayed_reward import DelayedReward
from ddm_mcts.environments.delivery_routing import DeliveryRouting
from ddm_mcts.environments.grid_navigation import GridNavigation, GridState
from ddm_mcts.environments.inventory_management import InventoryManagement
from ddm_mcts.environments.job_scheduling import Job, JobScheduling


def test_grid_legal_actions_obstacles_goal_and_reward():
    env = GridNavigation(GridState((0, 0), (0, 2), frozenset({(1, 0)}), 2, 3, max_steps=5))
    state = env.initial_state()
    assert env.legal_actions(state) == ("RIGHT",)
    with pytest.raises(ValueError):
        env.step(state, "DOWN")
    state = env.step(env.step(state, "RIGHT"), "RIGHT")
    assert env.is_terminal(state)
    assert env.get_reward(state, 1) == pytest.approx(0.96)


def test_scheduling_advances_time_and_applies_deadline_penalty():
    env = JobScheduling((Job("A", 3, 2, 10), Job("B", 1, 5, 2)), lateness_penalty=2)
    state = env.step(env.initial_state(), "A")
    assert state.time == 3
    assert state.reward == 8
    state = env.step(state, "B")
    assert env.is_terminal(state)
    assert env.get_reward(state, 1) == 10


def test_routing_accumulates_cost_visits_and_returns_to_depot():
    env = DeliveryRouting()
    state = env.step(env.initial_state(), "A")
    assert state.cost == 2 and "A" not in state.unvisited
    state = env.step(env.step(state, "C"), "B")
    assert env.legal_actions(state) == ("Depot",)
    state = env.step(state, "Depot")
    assert env.is_terminal(state)
    assert env.get_reward(state, 1) == -10


def test_inventory_ordering_costs_and_seeded_reproducibility():
    first, second = InventoryManagement(seed=42, horizon=2), InventoryManagement(seed=42, horizon=2)
    a = first.step(first.initial_state(), 4)
    b = second.step(second.initial_state(), 4)
    assert a == b
    assert a.timestep == 1 and a.total_cost >= 2
    a = first.step(a, 0)
    assert first.is_terminal(a)
    assert first.get_reward(a, 1) == -a.total_cost


def test_delayed_reward_known_optimal_path():
    env = DelayedReward()
    delayed = env.step(env.step(env.step(env.initial_state(), "B_DELAYED"), "CONTINUE_B"), "FINISH_B")
    immediate = env.step(env.step(env.initial_state(), "A_IMMEDIATE"), "CONTINUE_A")
    safe = env.step(env.initial_state(), "C_SAFE")
    assert env.get_reward(delayed, 1) == 102
    assert env.get_reward(immediate, 1) == -40
    assert env.get_reward(safe, 1) == 30
