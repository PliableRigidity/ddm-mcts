# ruff: noqa: E402
"""V5 semantic composition, physical predicates and real transition isolation."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from ddm_mcts.robotics.contacts import ContactInfo
from ddm_mcts.robotics.interactions import InteractionExecutor, Push, push_candidates
from ddm_mcts.robotics.manipulation_scene import ManipulationObject
from ddm_mcts.robotics.physical_goals import GoalConjunction, PhysicalGoal, Predicate, TemporalGoal, parse_physical_goals
from ddm_mcts.robotics.physical_predicates import PhysicalState, support_surface
from ddm_mcts.robotics.pose_control import PoseTarget
from ddm_mcts.robotics.reasoning_scene import panda_reasoning_scene

HEADLINE = "Pick up the cube and place it on the cylinder. Once that is done, wait for 2 seconds and topple the tower."


@pytest.mark.parametrize(
    "instruction,predicate",
    [
        ("Put the cube on the cylinder.", Predicate.ON_TOP_OF),
        ("Place the cube on top of the cylinder.", Predicate.ON_TOP_OF),
        ("Pick up the cube and put it on the cylinder.", Predicate.ON_TOP_OF),
        ("pick up the cube", Predicate.HELD),
        ("pick up the cylinder", Predicate.HELD),
        ("Place the cylinder next to the cube.", Predicate.NEXT_TO),
        ("Pick up the cylinder and place it next to the cube.", Predicate.NEXT_TO),
        ("Topple the tower.", Predicate.TOPPLED),
        ("Push the cylinder over.", Predicate.TOPPLED),
        ("Move the cube to the target location.", Predicate.AT_LOCATION),
    ],
)
def test_language_describes_goals(instruction, predicate):
    goal = parse_physical_goals(instruction).goals[0]
    if isinstance(goal, GoalConjunction):
        goal = goal.predicates[0]
    assert goal.predicate == predicate


def test_ordered_conjunction_wait_topple():
    task = parse_physical_goals(HEADLINE)
    assert len(task.goals) == 3
    assert [p.predicate for p in task.goals[0].predicates] == [Predicate.ON_TOP_OF, Predicate.SUPPORTED_BY, Predicate.STABLE]
    assert task.goals[1] == TemporalGoal(2)
    assert task.goals[2].subject == "tower"
    assert parse_physical_goals("wait for 2.5 seconds").goals[0].duration == 2.5


@pytest.mark.parametrize(
    "instruction",
    [
        "",
        "throw cube",
        "put sphere on cylinder",
        "put cube inside cylinder",
        "put cube on cube",
        "wait for 0 seconds",
        "wait for 11 seconds",
        "topple unknown",
    ],
)
def test_unsupported_goals_rejected(instruction):
    with pytest.raises(ValueError):
        parse_physical_goals(instruction)


def observed_state():
    objects = {
        "cube": ManipulationObject("cube", "pickup_cube", "box", (0.04, 0.04, 0.04), PoseTarget((0.5, 0.08, 0.42), (1, 0, 0, 0))),
        "cylinder": ManipulationObject(
            "cylinder", "pickup_cylinder", "cylinder", (0.04, 0.04, 0.05), PoseTarget((0.5, 0.08, 0.375), (1, 0, 0, 0))
        ),
    }
    state = PhysicalState.__new__(PhysicalState)
    state.world = SimpleNamespace(
        object_state=objects.__getitem__,
        backend=SimpleNamespace(data=SimpleNamespace(time=1)),
        controller=SimpleNamespace(pose=lambda: objects["cube"].pose),
    )
    contacts = [
        ContactInfo("cube", "cyl", "pickup_cube", "pickup_cylinder", -0.0001, 1),
        ContactInfo("cyl", "table", "pickup_cylinder", "support_table", -0.0001, 1),
    ]
    state.contacts = SimpleNamespace(contacts=lambda: tuple(contacts))
    state.gripper = SimpleNamespace(get_width=lambda: 0.04)
    state.history = {label: [(t, o.pose, 0.0, 0.0) for t in (0.5, 0.75, 1.0)] for label, o in objects.items()}
    state.groups = {}
    state.held_reference = {}
    return state, objects, contacts


@pytest.mark.parametrize("predicate", [Predicate.ON_TOP_OF, Predicate.SUPPORTED_BY, Predicate.CONTACTING])
def test_support_predicates_require_real_contact_and_geometry(predicate):
    state, objects, contacts = observed_state()
    goal = PhysicalGoal(predicate, "cube", "cylinder")
    assert state.evaluate(goal).satisfied
    contacts.clear()
    assert not state.evaluate(goal).satisfied
    contacts.append(ContactInfo("a", "b", "pickup_cube", "pickup_cylinder", -0.0001, 1))
    objects["cube"] = replace(objects["cube"], pose=PoseTarget((0.6, 0.08, 0.42), (1, 0, 0, 0)))
    if predicate != Predicate.CONTACTING:
        assert not state.evaluate(goal).satisfied


@pytest.mark.parametrize("predicate", [Predicate.UPRIGHT, Predicate.RELEASED, Predicate.STABLE])
def test_unary_true_false(predicate):
    state, objects, contacts = observed_state()
    goal = PhysicalGoal(predicate, "cube")
    assert state.evaluate(goal).satisfied
    if predicate == Predicate.UPRIGHT:
        objects["cube"] = replace(objects["cube"], pose=PoseTarget((0.5, 0.08, 0.42), (0.70710678, 0.70710678, 0, 0)))
    elif predicate == Predicate.RELEASED:
        contacts.append(ContactInfo("a", "b", "pickup_cube", "left_finger", 0, 1))
    else:
        state.history["cube"][-1] = (1, objects["cube"].pose, 0.02, 0.2)
    assert not state.evaluate(goal).satisfied


def test_stability_needs_time_not_requested_truth():
    state, objects, _ = observed_state()
    state.history["cube"] = [(1, objects["cube"].pose, 0, 0)]
    assert not state.evaluate(PhysicalGoal(Predicate.STABLE, "cube")).satisfied


def test_held_requires_bilateral_contact_width_transform_and_no_table():
    state, _, contacts = observed_state()
    goal = PhysicalGoal(Predicate.HELD, "cube")
    assert not state.evaluate(goal).satisfied
    state.held_reference["cube"] = PoseTarget((0, 0, 0), (1, 0, 0, 0))
    contacts[:] = [ContactInfo("a", "b", "pickup_cube", f, 0, 1) for f in ("left_finger", "right_finger")]
    assert state.evaluate(goal).satisfied
    contacts.append(ContactInfo("a", "b", "pickup_cube", "support_table", 0, 1))
    assert not state.evaluate(goal).satisfied


def test_tower_is_verified_logical_group_not_requested_truth():
    state, objects, _ = observed_state()
    goal = PhysicalGoal(Predicate.TOPPLED, "tower")
    assert not state.evaluate(goal).satisfied
    state.groups["tower"] = ("cube", "cylinder")
    assert not state.evaluate(goal).satisfied
    objects["cube"] = replace(objects["cube"], pose=PoseTarget((0.6, 0.08, 0.37), (0.70710678, 0, 0.70710678, 0)))
    assert state.evaluate(goal).satisfied


def test_object_toppled_requires_support_and_release():
    state, objects, contacts = observed_state()
    goal = PhysicalGoal(Predicate.TOPPLED, "cylinder")
    assert not state.evaluate(goal).satisfied
    objects["cylinder"] = replace(objects["cylinder"], pose=PoseTarget((0.5, 0.08, 0.37), (0.70710678, 0, 0.70710678, 0)))
    assert state.evaluate(goal).satisfied
    contacts.clear()
    assert not state.evaluate(goal).satisfied


def test_next_to_and_location_from_actual_poses():
    state, objects, contacts = observed_state()
    objects["cube"] = replace(objects["cube"], pose=PoseTarget((0.58, 0.08, 0.37), (1, 0, 0, 0)))
    assert state.evaluate(PhysicalGoal(Predicate.NEXT_TO, "cube", "cylinder")).satisfied
    contacts.append(ContactInfo("a", "b", "pickup_cube", "support_table", 0, 1))
    location = PhysicalGoal(Predicate.AT_LOCATION, "cube", location=(0.58, 0.08, 0.37))
    assert state.evaluate(location).satisfied
    objects["cube"] = replace(objects["cube"], pose=PoseTarget((0.60, 0.08, 0.37), (1, 0, 0, 0)))
    assert not state.evaluate(location).satisfied
    assert not state.evaluate(PhysicalGoal(Predicate.NEXT_TO, "cube", "cylinder")).satisfied


def test_support_surface_tracks_pose_and_com_region():
    state, objects, _ = observed_state()
    surface = support_surface(state.world, "cylinder")
    assert surface.pose.position[2] == pytest.approx(0.4)
    assert surface.contains_com((0.5, 0.08, 0.42))
    assert not surface.contains_com((0.52, 0.08, 0.42))
    objects["cylinder"] = replace(objects["cylinder"], pose=PoseTarget((0.6, 0.1, 0.375), (1, 0, 0, 0)))
    assert support_surface(state.world, "cylinder").pose.position[0] == 0.6
    objects["cylinder"] = replace(objects["cylinder"], pose=PoseTarget((0.6, 0.1, 0.375), (0.70710678, 0, 0.70710678, 0)))
    with pytest.raises(ValueError):
        support_surface(state.world, "cylinder")


@pytest.fixture
def world():
    import os
    from pathlib import Path

    model = os.environ.get("PANDA_MODEL", "/home/ishaan/robot-arm-playground/mujoco_menagerie/franka_emika_panda/scene.xml")
    if not Path(model).exists():
        pytest.skip("external Panda model unavailable")
    return panda_reasoning_scene(model)


def test_physical_wait_advances_time_and_unstable_object(world):
    executor = InteractionExecutor(world)
    # Test fixture only: start with unsupported cube; execution never sets object state.
    address = world.backend.model.joint("pickup_cube_free").qposadr[0]
    world.backend.data.qpos[address + 2] += 0.005
    mujoco.mj_forward(world.backend.model, world.backend.data)
    initial = world.object_state("cube").pose.position[2]
    elapsed = executor.wait(2)
    assert elapsed == pytest.approx(2, abs=0.002)
    assert world.object_state("cube").pose.position[2] < initial - 0.003
    assert executor.counts["substeps"] == 1000
    assert executor.state.evaluate(PhysicalGoal(Predicate.STABLE, "cube")).satisfied


def test_geometry_derived_push_candidates(world):
    choices = push_candidates(world, "cylinder")
    assert len(choices) == 4
    assert len(set(a.action_id for a in choices)) == 4
    assert choices[1].contact_point[2] > choices[0].contact_point[2]
    assert choices[1].distance > choices[0].distance
    obj = world.object_state("cylinder")
    assert all(
        obj.pose.position[2] - obj.dimensions[2] / 2 < a.contact_point[2] < obj.pose.position[2] + obj.dimensions[2] / 2 for a in choices
    )
    assert all(np.linalg.norm(a.direction) == 1 for a in choices)


@pytest.mark.parametrize(
    "distance,speed,direction", [(0, 0.02, (1, 0, 0)), (0.2, 0.02, (1, 0, 0)), (0.05, 0, (1, 0, 0)), (0.05, 0.02, (2, 0, 0))]
)
def test_push_bounds(distance, speed, direction):
    with pytest.raises(ValueError):
        Push("cylinder", (0.5, 0.08, 0.4), direction, distance, speed)


def test_mu_joco_search_overrides_misleading_push_prior_and_restores(world):
    from ddm_mcts.robotics.interaction_search import search_push
    from ddm_mcts.search.mcts import MCTSConfig

    class InferiorPrior:
        def probabilities(self, state, actions):
            weights = {a: (0.99 if a.distance == 0.015 and a.direction[0] > 0 else 0.01 / 3) for a in actions}
            return weights

    state = PhysicalState(world)
    state.observe()
    before = world.snapshot()
    action, search = search_push(world, state, "cylinder", policy=InferiorPrior(), config=MCTSConfig(60, c_puct=0.005, seed=0))
    assert world.snapshot() == before
    assert action.distance > 0.015
    assert search["speculative_substeps"] > 0
    weak = [r for r in search["transitions"] if r["action"]["distance"] == 0.015]
    strong = [r for r in search["transitions"] if r["action"]["distance"] > 0.015]
    assert all(not r["verification"]["satisfied"] for r in weak)
    assert any(r["verification"]["satisfied"] and not r["failure"] for r in strong)
    executor = InteractionExecutor(world, state=state)
    executor.push(action)
    assert state.evaluate(PhysicalGoal(Predicate.TOPPLED, "cylinder")).satisfied
    assert executor.counts["forbidden_contacts"] == 0


def test_failure_stops_later_goals_and_bounds_push_attempts(world, monkeypatch, tmp_path):
    import ddm_mcts.robotics.physical_reasoning as module
    from ddm_mcts.robotics.physical_reasoning import CompositionalPhysicalAgent

    calls = []

    def ineffective(*args, **kwargs):
        calls.append("search")
        return push_candidates(world, "cylinder")[0], {"speculative_substeps": 0}

    monkeypatch.setattr(module, "search_push", ineffective)
    monkeypatch.setattr(InteractionExecutor, "push", lambda *args: {"contact_seen": False})
    agent = CompositionalPhysicalAgent(world)
    result = agent.run_goal_instruction("Push the cylinder over, then wait for 2 seconds.", tmp_path)
    assert not result.success
    assert calls == ["search", "search"]
    assert result.trace["failure"]["reason"] == "topple_retry_limit_exceeded"
    assert not any(r.get("wait_elapsed") for r in result.trace["goal_results"])


def test_unstable_stack_prevents_wait_and_topple(world, monkeypatch, tmp_path):
    from ddm_mcts.robotics.physical_reasoning import CompositionalPhysicalAgent

    agent = CompositionalPhysicalAgent(world)
    monkeypatch.setattr(agent, "_placement", lambda *args: None)
    result = agent.run_goal_instruction(HEADLINE, tmp_path)
    assert not result.success
    assert len(result.trace["goal_results"]) == 1
    assert result.trace["goal_results"][0]["success"] is False
    assert result.trace["speculative_substeps"] == 0
    assert not any(r.get("wait_elapsed") for r in result.trace["goal_results"])


def test_satisfied_goal_avoids_interaction(world, monkeypatch, tmp_path):
    from ddm_mcts.robotics.physical_reasoning import CompositionalPhysicalAgent

    # Fixture initialization only: an already placed cube, before agent execution.
    address = world.backend.model.joint("pickup_cube_free").qposadr[0]
    world.backend.data.qpos[address : address + 3] = (0.46, 0.10, 0.37)
    mujoco.mj_forward(world.backend.model, world.backend.data)
    for _ in range(10):
        world.backend.advance()
    before = world.snapshot()

    def forbidden(*args):
        raise AssertionError("unnecessary manipulation of an already satisfied physical goal")

    monkeypatch.setattr(InteractionExecutor, "grasp", forbidden)
    result = CompositionalPhysicalAgent(world).run_goal_instruction("Move the cube to the target location.", tmp_path)
    assert result.success
    assert result.trace["live_physics"]["substeps"] == 0
    assert world.snapshot() == before


def test_table_support_uses_actual_geom(world):
    surface = support_surface(world, "table")
    assert surface.pose.position[2] == pytest.approx(0.35)
    assert surface.contains_com((0.5, 0, 0.37))
    assert not surface.contains_com((0.8, 0, 0.37))


def test_affordances_exclude_tipped_grasp_and_support():
    state, objects, _ = observed_state()
    assert state.affordances("cube")["graspable"]
    objects["cube"] = replace(objects["cube"], pose=PoseTarget((0.5, 0.08, 0.37), (0.70710678, 0, 0.70710678, 0)))
    assert not state.affordances("cube")["graspable"]
    assert not state.affordances("cube")["support_surface"]
    assert state.affordances("cube")["pushable"]


def test_wait_resolution_and_configuration_restored(world):
    executor = InteractionExecutor(world)
    original = world.backend.physics_steps
    assert executor.wait(0.013) == pytest.approx(0.014, abs=1e-8)
    assert world.backend.physics_steps == original


def test_primitive_observe_verify_and_gripper_callbacks(world):
    from ddm_mcts.robotics.interactions import Primitive

    executor = InteractionExecutor(world)
    executor.execute(Primitive.CLOSE_GRIPPER)
    closed = executor.gripper.get_width()
    executor.execute(Primitive.OPEN_GRIPPER)
    assert executor.gripper.get_width() > closed + 0.05
    assert executor.counts["substeps"] > 0
    observed = executor.execute(Primitive.OBSERVE)
    assert observed["objects"]["cube"]["position"] == world.object_state("cube").pose.position
    assert executor.execute(Primitive.VERIFY, goal=PhysicalGoal(Predicate.RELEASED, "cube")).satisfied


def test_held_open_requires_supported_release(world):
    from ddm_mcts.robotics.interactions import Primitive

    executor = InteractionExecutor(world)
    executor.phase = "carry"
    before = world.snapshot()
    with pytest.raises(RuntimeError, match="supported_release"):
        executor.execute(Primitive.OPEN_GRIPPER)
    assert world.snapshot() == before


def test_failed_wait_restores_block_size(world, monkeypatch):
    executor = InteractionExecutor(world)
    original = world.backend.physics_steps

    def fail():
        raise RuntimeError("controlled callback failure")

    monkeypatch.setattr(executor, "monitor", fail)
    with pytest.raises(RuntimeError):
        executor.wait(0.013)
    assert world.backend.physics_steps == original


@pytest.mark.parametrize("target,point", [("sphere", (0.5, 0.08, 0.4)), ("cube", (float("nan"), 0.08, 0.4))])
def test_invalid_push_entity_or_point_rejected_before_actuation(target, point):
    with pytest.raises(ValueError):
        Push(target, point, (1, 0, 0), 0.05)


def test_wait_verification_gates_later_interactions(world, monkeypatch, tmp_path):
    from ddm_mcts.robotics.physical_reasoning import CompositionalPhysicalAgent

    monkeypatch.setattr(InteractionExecutor, "wait", lambda *args: 0.0)
    result = CompositionalPhysicalAgent(world).run_goal_instruction("Wait for 2 seconds, then push the cylinder over.", tmp_path)
    assert not result.success
    assert result.trace["failure"]["reason"] == "physical_wait_duration_not_verified"
    assert result.trace["failure"]["goal_index"] == 0
    assert result.trace["speculative_substeps"] == 0
    assert result.trace["goal_results"][0]["success"] is False


def test_pick_then_support_goal_reuses_held_precondition_without_regrasp(world, monkeypatch, tmp_path):
    from ddm_mcts.robotics.physical_reasoning import CompositionalPhysicalAgent

    original = InteractionExecutor.grasp
    grasps = []

    def recorded(executor, label):
        grasps.append(label)
        return original(executor, label)

    monkeypatch.setattr(InteractionExecutor, "grasp", recorded)
    agent = CompositionalPhysicalAgent(world)
    picked = agent.run_goal_instruction("Pick up the cube.", tmp_path)
    assert picked.success
    placed = agent.run_goal_instruction("Put the cube on the cylinder.", tmp_path)
    assert placed.success, placed.trace.get("failure")
    assert grasps == ["cube"]
    assert any(e["event"] == "held_precondition_reused" for e in placed.trace["events"])
    assert picked.trace["state_continuity"][-1]["end"] == placed.trace["state_continuity"][0]["start"]


def test_cli_preserves_explicit_zero_exploration(monkeypatch, tmp_path):
    from ddm_mcts.robotics import physical_reasoning_cli as module
    from ddm_mcts.robotics.cli import build_parser

    captured = []

    class Agent:
        def __init__(self, world, **kwargs):
            captured.append(kwargs["search_config"].c_puct)

        def run_goal_instruction(self, *args):
            return SimpleNamespace(success=True, folder=tmp_path, trace={})

    monkeypatch.setattr(module, "panda_reasoning_scene", lambda *args: object())
    monkeypatch.setattr(module, "CompositionalPhysicalAgent", Agent)
    parser = build_parser()
    args = parser.parse_args(["--model", "unused", "--task", "physical-reason", "--instruction", "Wait for 2 seconds.", "--c-puct", "0"])
    assert module.run_physical_reasoning(args, parser) == 0
    assert captured == [0]
