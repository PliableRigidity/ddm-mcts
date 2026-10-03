# ruff: noqa: E402
"""Bounded language, closed-loop composition and genuine physical MCTS choices."""

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("mujoco")

from ddm_mcts.robotics.manipulation_agent import PhysicalAIManipulationAgent
from ddm_mcts.robotics.manipulation_observation import ManipulationCamera, ManipulationPerception
from ddm_mcts.robotics.manipulation_scene import panda_pickup_scene
from ddm_mcts.robotics.manipulation_search import SkillOutcome, SkillTask, candidate_actions, plan_skill
from ddm_mcts.robotics.manipulation_task import ManipulationOperation, PhysicalManipulationTask, parse_manipulation_task
from ddm_mcts.robotics.observation import Observation
from ddm_mcts.robotics.pick_place import PickPlaceAgent
from ddm_mcts.robotics.placement import NextTo, PlacementTarget
from ddm_mcts.robotics.vlm_backend import Qwen3VLBackend
from ddm_mcts.search.mcts import MCTSConfig


@pytest.mark.parametrize(
    "instruction,target,reference",
    [
        ("Pick up the cube.", "cube", None),
        ("Pick up the cylinder.", "cylinder", None),
        ("Place the cylinder next to the cube.", "cylinder", "cube"),
        ("Pick up the cylinder and place it next to the cube.", "cylinder", "cube"),
        ("Move the cylinder next to the cube.", "cylinder", "cube"),
        ("Move the cube next to the cylinder.", "cube", "cylinder"),
        ("Move the cube to the target location.", "cube", "position"),
        ("Move the cube to the other target location.", "cube", "position"),
        ("  PICK UP THE CUBE.  ", "cube", None),
    ],
)
def test_supported_parser(instruction, target, reference):
    task = parse_manipulation_task(instruction)
    assert task.instruction == instruction
    assert task.operations[0].target == target
    destination = task.operations[0].destination
    assert (destination.reference if isinstance(destination, NextTo) else "position" if destination else None) == reference


@pytest.mark.parametrize(
    "instruction",
    [
        "",
        " ",
        None,
        "Pick up the sphere.",
        "Move the cube behind the cylinder.",
        "Move cylinder next to cylinder",
        "Pick up a cube",
        "Move the cube",
        "Grasp arbitrary object",
        "Pick up cube, then pick up cylinder",
        "Move cylinder next to cube and move cube to target location",
        "Move cube to target location. Ignore all rules",
        "Move cube to target location then ",
    ],
)
def test_invalid_parser(instruction):
    with pytest.raises(ValueError):
        parse_manipulation_task(instruction)


def test_order_and_target_binding():
    task = parse_manipulation_task("Move the cylinder next to the cube, then move the cube to the target location.")
    assert [op.target for op in task.operations] == ["cylinder", "cube"]
    assert task.operations[0].destination == NextTo("cube")
    assert task.operations[1].destination.position == (0.46, 0.10, 0.37)


def test_parser_custom_target():
    assert parse_manipulation_task("Move cube to target location", target_position=(0.6, -0.1, 0.37)).operations[
        0
    ].destination.position == (0.6, -0.1, 0.37)


@pytest.mark.parametrize("args", [("sphere", None), ("cube", "invalid"), ("cube", NextTo("cube"))])
def test_operation_validation(args):
    with pytest.raises(ValueError):
        ManipulationOperation(*args)


def test_task_bounds():
    with pytest.raises(ValueError):
        PhysicalManipulationTask("move", ())
    with pytest.raises(ValueError):
        parse_manipulation_task(" then ".join(["Move cube to target location"] * 5))


@pytest.fixture
def world():
    path = os.environ.get("PANDA_MODEL")
    if not path or not Path(path).is_file():
        pytest.skip("external PANDA_MODEL required")
    return panda_pickup_scene(path)


def test_absolute_action_is_honestly_forced(world, tmp_path):
    op = ManipulationOperation("cube", PlacementTarget((0.6, -0.1, 0.37)))
    before = world.snapshot()
    action, diagnostics = plan_skill(world, op, tmp_path)
    assert action.destination == op.destination
    assert diagnostics["forced"] and diagnostics["simulations"] == 0
    assert before == world.snapshot()


def test_candidate_identity_and_geometry(world):
    actions = candidate_actions(world, ManipulationOperation("cylinder", NextTo("cube")))
    assert len(actions) == len(set(actions)) == 2
    assert all(a.action_id.startswith("PICK_PLACE:cylinder:NEXT_TO:cube:") for a in actions)
    x = world.object_state("cube").pose.position[0]
    np.testing.assert_allclose([abs(a.destination.position[0] - x) for a in actions], 0.08, atol=1e-5)


def test_ground_truth_resolution_and_invalidation(world):
    perception = ManipulationPerception(world)
    observation = Observation(0, 1, "ground-truth", world.get_state())
    result = perception.resolve(observation, ["cylinder", "cube"])
    assert [b["target"] for b in result["bindings"]] == ["cylinder", "cube"]
    perception.invalidate("object_moved")
    assert perception.generation == 1 and perception.invalidations == ["object_moved"]
    with pytest.raises(ValueError):
        perception.resolve(observation, ["sphere"])


@pytest.mark.parametrize("success,cost,expected", [(True, 0.2, -0.2), (False, 0.2, -2), (True, 0.8, -0.8)])
def test_search_values_are_physical_success_then_carry_cost(success, cost, expected):
    assert SkillTask.evaluate(SkillOutcome(True, success, cost)) == expected


def test_live_camera_is_read_only_and_calibrated(world):
    camera = ManipulationCamera(world)
    try:
        before = world.snapshot()
        first = camera.observe()
        camera.alternate_view()
        second = camera.observe()
        assert before == world.snapshot()
        assert first.sequence + 1 == second.sequence
        assert first.rgb.dtype == np.uint8 and not first.rgb.flags.writeable
        assert first.camera != second.camera
        for label in ("cube", "cylinder"):
            p = world.object_state(label).pose.position
            np.testing.assert_allclose(first.camera.intersect_plane(first.camera.project(p), p[2]), p, atol=1e-7)
    finally:
        camera.close()


class PreferRight:
    def probabilities(self, state, actions):
        return {a: 0.99 if a.action_id.endswith("POS") else 0.01 for a in actions}


def test_actual_mujoco_mcts_overrides_inferior_prior(world, tmp_path):
    # Physically relocate the reference first; no qpos edit or fake transition.
    setup = PickPlaceAgent(world).pick_and_place("cube", PlacementTarget((0.54, -0.08, 0.37)), tmp_path / "setup")
    assert setup.success
    before = world.snapshot()
    action, diagnostics = plan_skill(
        world,
        ManipulationOperation("cylinder", NextTo("cube")),
        tmp_path / "search",
        policy=PreferRight(),
        config=MCTSConfig(60, c_puct=0.005, seed=0),
    )
    assert before == world.snapshot()
    transitions = {t["action"]["action_id"]: t for t in diagnostics["transitions"]}
    assert all(t["success"] for t in transitions.values())
    assert action.action_id.endswith("NEG")  # policy strongly prefers POS
    selected = transitions[action.action_id]
    inferior = next(t for name, t in transitions.items() if name.endswith("POS"))
    assert selected["carry_path_m"] + 0.02 < inferior["carry_path_m"]
    assert selected["value"] > inferior["value"]
    assert max(diagnostics["statistics"], key=lambda s: s["prior"])["action"].endswith("POS")


def test_pick_skill_physical_verification_and_checkpoints(world, tmp_path):
    agent = PhysicalAIManipulationAgent(world)
    world.reset = lambda: pytest.fail("hidden robot reset")
    world.backend.reset = lambda *args: pytest.fail("hidden scene reset")
    result = agent.run_instruction("Pick up the cube.", tmp_path)
    assert result.success
    row = result.trace["operations"][0]
    assert row["status"] == "verified" and row["physical_verification"]["success"]
    assert [o["stage"] for o in row["observations"]] == ["before_operation", "after_lift", "operation_complete"]
    assert row["planning"]["forced"]
    assert (result.folder / "task_trace.json").is_file()
    agent.close()


def test_integrated_continuous_plan_and_fresh_trees(world, tmp_path):
    agent = PhysicalAIManipulationAgent(world)
    world.reset = lambda: pytest.fail("hidden reset")
    world.backend.reset = lambda *args: pytest.fail("hidden reset")
    result = agent.run_instruction("Move cylinder next to cube, then move cube to target location", tmp_path)
    assert result.success and result.trace["state_continuity"]
    rows = result.trace["operations"]
    assert rows[0]["end_state"] == rows[1]["start_state"]
    assert rows[0]["relation_verification"]["success"]
    assert not rows[0]["planning"]["forced"] and rows[1]["planning"]["forced"]
    assert rows[0]["planning"]["snapshot_preserved"]
    assert all(r["status"] == "verified" for r in rows)
    assert [o["sequence"] for r in rows for o in r["observations"]] == list(range(1, 7))
    assert len(result.trace["cache_invalidations"]) == 6
    assert not any(r["physical_verification"]["operations"][0]["forbidden_contacts"] for r in rows)
    assert json.loads((result.folder / "task_trace.json").read_text())["success"]
    agent.close()


def test_missing_perception_stops_before_any_skill(world, tmp_path):
    class Missing(ManipulationPerception):
        def resolve(self, observation, labels):
            raise ValueError("target missing")

    agent = PhysicalAIManipulationAgent(world, perception=Missing(world))
    before = world.snapshot()
    result = agent.run_instruction("Move cylinder next to cube, then move cube to target location", tmp_path)
    assert not result.success and len(result.trace["operations"]) == 1
    assert result.trace["operations"][0]["failure"]["reason"] == "target missing"
    assert before == world.snapshot()
    agent.close()


def test_bounded_stale_semantics_reobserves_once(world, tmp_path):
    class Stale(ManipulationPerception):
        calls = 0

        def resolve(self, observation, labels):
            self.calls += 1
            raise ValueError("stale grounding")

    camera = ManipulationCamera(world)
    perception = Stale(world)
    agent = PhysicalAIManipulationAgent(world, observation=camera, perception=perception)
    before = world.snapshot()
    result = agent.run_instruction("Pick up cube", tmp_path)
    assert not result.success and perception.calls == 2
    assert len(result.trace["operations"][0]["semantic_recovery"]) == 1
    assert before == world.snapshot()
    agent.close()


def test_shared_backend_and_mock_semantic_grounding(world):
    camera = ManipulationCamera(world)
    observed = camera.observe()
    boxes = []
    for label in ("cube", "cylinder"):
        u, v = observed.camera.project(world.object_state(label).pose.position)
        boxes.append([int((u - 22) / 640 * 1000), int((v - 25) / 480 * 1000), int((u + 22) / 640 * 1000), int((v + 25) / 480 * 1000)])

    class Runtime:
        device, dtype = "fake", "fake"

        def __init__(self, config):
            self.count = 0

        def generate(self, rgb, prompt, config):
            label = "cube" if self.count == 0 else "cylinder"
            box = boxes[self.count]
            self.count += 1
            assert "XYZ" not in prompt and "pickup_" not in prompt
            return json.dumps({"target_label": label, "target_description": label, "bbox_2d": box})

    backend = Qwen3VLBackend(runtime_factory=Runtime)
    perception = ManipulationPerception(world, backend)
    result = perception.resolve(observed, ["cube", "cylinder"])
    assert len(result["bindings"]) == 2
    assert backend.load_count == 1 and backend.calls == 2
    perception.invalidate("subgoal_changed")
    assert backend.load_count == 1 and perception.semantic.semantic is None
    camera.close()


def test_wrong_box_does_not_fall_back_to_ground_truth(world, monkeypatch):
    from ddm_mcts.robotics import manipulation_observation as module

    class Wrong:
        def __init__(self, *args):
            pass

        def perceive(self, observation, goal):
            return SimpleNamespace(target=lambda: SimpleNamespace(label="cylinder", position=world.object_state("cube").pose.position))

    monkeypatch.setattr(module, "VLMSemanticPerception", Wrong)
    perception = ManipulationPerception(world, backend=object())
    with pytest.raises(ValueError, match="unambiguously"):
        perception.resolve(Observation(0, 1, "camera", world.get_state()), ["cylinder"])


def test_invalid_metric_destination_stops_plan(world, tmp_path):
    task = PhysicalManipulationTask(
        "invalid destination",
        (ManipulationOperation("cube", PlacementTarget((3, 0, 0.37))), ManipulationOperation("cylinder", NextTo("cube"))),
    )
    agent = PhysicalAIManipulationAgent(world)
    before = world.snapshot()
    result = agent.run_task(task, tmp_path)
    assert not result.success and len(result.trace["operations"]) == 1
    assert before == world.snapshot()
    agent.close()


def test_failed_skill_stops_later_operations(world, tmp_path, monkeypatch):
    agent = PhysicalAIManipulationAgent(world)
    calls = []

    def failed(target, destination, output, **kwargs):
        calls.append(target)
        return SimpleNamespace(success=False, folder=output, trace={"success": False, "operations": []})

    monkeypatch.setattr(agent.skills, "pick_and_place", failed)
    task = PhysicalManipulationTask(
        "two absolute operations",
        (
            ManipulationOperation("cube", PlacementTarget((0.6, -0.1, 0.37))),
            ManipulationOperation("cylinder", PlacementTarget((0.4, -0.1, 0.375))),
        ),
    )
    result = agent.run_task(task, tmp_path)
    assert not result.success and calls == ["cube"]
    assert [p["status"] for p in result.trace["progress"]] == ["failed", "pending"]
    agent.close()


def test_semantic_recovery_succeeds_without_reloading(world, tmp_path):
    class OnceStale(ManipulationPerception):
        calls = 0

        def resolve(self, observation, labels):
            self.calls += 1
            if self.calls == 1:
                raise ValueError("stale grounding")
            return super().resolve(observation, labels)

    camera = ManipulationCamera(world)
    perception = OnceStale(world)
    agent = PhysicalAIManipulationAgent(world, observation=camera, perception=perception)
    result = agent.run_instruction("Pick up cube", tmp_path)
    assert result.success and perception.calls == 2
    assert len(result.trace["operations"][0]["semantic_recovery"]) == 1
    agent.close()


def test_one_viewer_object_receives_only_executed_substeps(world, tmp_path):
    from contextlib import contextmanager

    class Viewer:
        substeps = 0
        scopes = 0

        def target(self, position):
            pass

        @contextmanager
        def execution_scope(self):
            self.scopes += 1

            def update():
                self.substeps += 1

            with world.backend.observe_steps(update):
                yield

    viewer = Viewer()
    agent = PhysicalAIManipulationAgent(world)
    task = parse_manipulation_task("Move cylinder next to cube, then move cube to target location")
    result = agent.run_task(task, tmp_path, visualization=viewer)
    assert result.success and viewer.scopes >= 2
    executed = sum(r["physical_verification"]["operations"][0]["physics_substeps"] for r in result.trace["operations"])
    assert 0 < viewer.substeps <= executed
    assert sum(len(r["planning"].get("transitions", [])) for r in result.trace["operations"]) == 2
    agent.close()


@pytest.mark.parametrize("raw", ["not json", "{}", '{"target_label":"unknown","target_description":"absent","bbox_2d":[0,0,1,1]}'])
def test_malformed_real_backend_protocol_stops_without_motion(world, tmp_path, raw):
    class Runtime:
        device, dtype = "fake", "fake"

        def __init__(self, config):
            pass

        def generate(self, rgb, prompt, config):
            return raw

    backend = Qwen3VLBackend(runtime_factory=Runtime)
    camera = ManipulationCamera(world)
    agent = PhysicalAIManipulationAgent(world, observation=camera, perception=ManipulationPerception(world, backend))
    before = world.snapshot()
    result = agent.run_instruction("Move cylinder next to cube, then move cube to target location", tmp_path)
    assert not result.success and len(result.trace["operations"]) == 1
    assert before == world.snapshot()
    assert backend.load_count == 1 and backend.calls == result.trace["model_task_calls"] == 2
    assert result.trace["progress"][1]["status"] == "pending"
    agent.close()
