# ruff: noqa: E402
import json
import os
from dataclasses import replace
from math import dist
from pathlib import Path

import pytest

pytest.importorskip("mujoco")
np = pytest.importorskip("numpy")

from ddm_mcts.policies.random_policy import UniformPolicy
from ddm_mcts.policies.text_laya_policy import TextLayaPolicy
from ddm_mcts.robotics.camera import MujocoCameraObservationProvider
from ddm_mcts.robotics.cli import main, state_text
from ddm_mcts.robotics.color_perception import ColorPlanePerception
from ddm_mcts.robotics.core import RoboticsPlanner
from ddm_mcts.robotics.observation import GroundTruthObservationProvider, SemanticGoal
from ddm_mcts.robotics.perception import GroundTruthPerception, ModelPerceptionAdapter
from ddm_mcts.robotics.physical_agent import PhysicalAgent
from ddm_mcts.robotics.representation import SemanticReachTask
from ddm_mcts.robotics.visual import VisualInspection
from ddm_mcts.robotics.visual_policy import VisualDecisionPolicy
from ddm_mcts.robotics.visual_scene import TARGET_PLANE_Z, VisualTarget, panda_visual_reach
from ddm_mcts.search.mcts import MCTSConfig


@pytest.fixture
def model():
    model = os.environ.get("PANDA_MODEL")
    if not model or not Path(model).is_file():
        pytest.skip("PANDA_MODEL required for physical-agent integrations")
    return model


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


@pytest.mark.parametrize("color", ["red", "blue"])
def test_real_camera_closed_loop_semantic_reach(model, tmp_path, color, monkeypatch):
    world = panda_visual_reach(model)
    camera = MujocoCameraObservationProvider(world)
    perception = ColorPlanePerception(TARGET_PLANE_Z)
    planner = RoboticsPlanner(world, config=MCTSConfig(60, seed=0))
    events = []
    searching = False
    observe, perceive, plan, step = camera.observe, perception.perceive, planner.plan, world.step

    def traced_observe():
        events.append("observe")
        return observe()

    def traced_perceive(*args):
        events.append("perceive")
        return perceive(*args)

    def traced_plan(*args, **kwargs):
        nonlocal searching
        events.append("plan")
        searching = True
        try:
            return plan(*args, **kwargs)
        finally:
            searching = False

    def traced_step(action):
        if not searching:
            events.append("execute")
        return step(action)

    monkeypatch.setattr(camera, "observe", traced_observe)
    monkeypatch.setattr(perception, "perceive", traced_perceive)
    monkeypatch.setattr(planner, "plan", traced_plan)
    monkeypatch.setattr(world, "step", traced_step)
    with PhysicalAgent(
        world,
        camera,
        perception,
        SemanticReachTask(SemanticGoal(color)),
        planner,
        diagnostics=world.target_entities,
        save_images=True,
    ) as agent:
        folder = agent.run(tmp_path)
        summary = json.loads((folder / "summary.json").read_text())
        assert summary["success"]
        observations = read_jsonl(folder / "observations.jsonl")
        steps = read_jsonl(folder / "steps.jsonl")
        assert camera.sequence == len(steps) + 1 == len(observations)
        assert events == ["observe", "perceive", "plan", "execute"] * len(steps) + ["observe", "perceive"]
        assert [row["observation"]["sequence"] for row in observations] == list(range(1, len(observations) + 1))
        assert [row["perceived_state"]["robot"]["steps"] for row in observations] == list(range(len(observations)))
        assert all(row["perceived_state"]["semantic_goal"]["label"] == color for row in observations)
        assert all({entity["label"] for entity in row["ground_truth_entities"]} == {"red", "blue"} for row in observations)
        assert all(max(row["perception_errors_m"].values()) < 0.008 for row in observations)
        truth = next(e for e in world.target_entities() if e.label == color)
        assert dist(world.get_state().position, truth.position) < 0.015
        assert len(list((folder / "images").glob("*.ppm"))) == len(observations)
        assert next((folder / "images").glob("*.ppm")).read_bytes().startswith(b"P6\n320 240\n255\n")
        assert all("execution_seconds" in row and "decision_context" in row for row in steps)
        assert all(row["simulations"] == 60 and len(row["root_statistics"]) == 6 for row in steps)
        assert json.loads((folder / "config.json").read_text())["agent"]["diagnostics_enabled"]
    assert camera.renderer is None


def test_ground_truth_path_and_cli(model, tmp_path):
    world = panda_visual_reach(model)
    provider = GroundTruthObservationProvider(world, world.target_entities)
    with PhysicalAgent(
        world,
        provider,
        GroundTruthPerception(),
        SemanticReachTask(SemanticGoal("red")),
        RoboticsPlanner(world, config=MCTSConfig(60, seed=0)),
        diagnostics=world.target_entities,
    ) as agent:
        folder = agent.run(tmp_path)
        assert world.task.is_success(world.get_state())
        assert max(json.loads((folder / "summary.json").read_text())["final_perception"]["perception_errors_m"].values()) == 0
    assert main(["--model", model, "--target", ".5745", "0", ".6245", "--observation", "ground-truth", "--output", str(tmp_path)]) == 0
    with pytest.raises(SystemExit):
        main(["--model", model, "--task", "visual-reach", "--observation", "camera", "--perception", "ground-truth"])


def test_swappable_policies_model_boundaries_and_invisible_search(model):
    world = panda_visual_reach(model)
    camera = MujocoCameraObservationProvider(world)
    perception = ColorPlanePerception(TARGET_PLANE_Z)
    # A model adapter can be injected without downloads; its test predictor derives
    # the representation from RGB through the deterministic reference implementation.
    vlm = ModelPerceptionAdapter(lambda observation, goal: replace(perception.perceive(observation, goal), source="mock-vlm"))
    predictor_requests = []

    def predictor(text, question):
        predictor_requests.append(text)
        return {"answers": {"action": {"probabilities": {key: 1 for key in question["action"]["criteria"]}}}}

    ddm = TextLayaPolicy(state_text, objective="Reach the perceived target", predictor=predictor)
    planner = RoboticsPlanner(world, ddm, MCTSConfig(10, seed=0))

    class MockVisualModel:
        def score(self, rgb, goal, actions):
            assert rgb.shape == (240, 320, 3) and goal.label == "red"
            return {actions[0]: 1}

    visual_policy = VisualDecisionPolicy(MockVisualModel())
    with PhysicalAgent(world, camera, vlm, SemanticReachTask(SemanticGoal("red")), planner, visual_policy=visual_policy) as agent:
        state = agent.prepare()
        assert state.source == "mock-vlm"
        assert camera.sequence == 1
        saved = world.snapshot()
        result = agent.plan()
        assert world.snapshot() == saved
        assert camera.sequence == 1  # No speculative observations.
        assert visual_policy.calls == 1  # Root only, no fake future images.
        with pytest.raises(ValueError, match="root_policy"):
            visual_policy.probabilities(next(iter(result.root.children.values())).state, world.actions)
        assert ddm.calls > 0 and "target XYZ" in predictor_requests[0]
        assert result.root.children[world.actions[0]].prior == 1
        assert planner.adapter.state_projection is None
        planner.search.policy = UniformPolicy()
        agent.prepare()
        agent.plan()
        assert camera.sequence == 2
    with pytest.raises(RuntimeError):
        VisualDecisionPolicy(MockVisualModel()).probabilities(None, world.actions)


def test_visual_perception_uses_pixels_when_scene_positions_change(model):
    targets = (
        VisualTarget("red", (0.65, -0.09, TARGET_PLANE_Z), (1, 0, 0, 1)),
        VisualTarget("blue", (0.45, 0.09, TARGET_PLANE_Z), (0, 0, 1, 1)),
    )
    world = panda_visual_reach(model, targets=targets)
    with MujocoCameraObservationProvider(world, width=640, height=480) as camera:
        observation = camera.observe()
        state = ColorPlanePerception(TARGET_PLANE_Z).perceive(observation, SemanticGoal("red"))
    assert state.goal == pytest.approx(targets[0].position, abs=0.002)
    assert state.target().position[1] < 0  # Not the default red marker coordinates.
    assert world.backend.model.geom("target_red").contype == 0
    assert world.backend.model.geom("target_red").conaffinity == 0

    # The perception component has only pixels/calibration and robot telemetry.
    # Poison every optional privileged answer in its observation: inference must
    # remain identical, including when the chosen semantic goal changes.
    from ddm_mcts.robotics.observation import Entity

    poisoned = replace(
        observation,
        robot=replace(observation.robot, goal=(99, 99, 99)),
        entities=(Entity("red", (99, 99, 99)),),
    )
    assert ColorPlanePerception(TARGET_PLANE_Z).perceive(poisoned, SemanticGoal("red")).goal == state.goal
    # This blue placement is occluded by the initial arm. No privileged fallback
    # may manufacture a detection even though its true coordinates are known.
    with pytest.raises(ValueError, match="expected one"):
        ColorPlanePerception(TARGET_PLANE_Z).perceive(poisoned, SemanticGoal("blue"))


def test_multiple_placements_and_same_components_semantic_goal_switch(model, tmp_path):
    from examples.robotics.validate_visual import validate

    records = validate(model, tmp_path)
    assert len(records) == 6
    assert all(row["success"] and row["true_final_reach_error_m"] < 0.015 for row in records)
    assert all(row["max_perception_error_m"] < 0.008 for row in records)
    assert all(row["observations"] == row["actions"] + 1 for row in records)
    assert len({row["run"] for row in records}) == 6


def test_visualization_compatible_with_agent_without_gui(model, tmp_path):
    from test_visual import FakeViewer

    world = panda_visual_reach(model)
    planner = RoboticsPlanner(world, config=MCTSConfig(10, seed=0))
    with PhysicalAgent(
        world,
        GroundTruthObservationProvider(world, world.target_entities),
        GroundTruthPerception(),
        SemanticReachTask(SemanticGoal("red"), max_steps=1),
        planner,
    ) as agent:
        state = agent.prepare()
        with VisualInspection(world, planner, speed=10000, launch=FakeViewer) as visual:
            visual.perceived(state)
            syncs = visual.viewer.syncs
            agent.plan()
            assert visual.viewer.syncs == syncs
            folder = agent.run(tmp_path, visualization=visual)
            assert json.loads((folder / "summary.json").read_text())["decisions"] == 1
            assert world.backend._step_callback is None


def test_missing_perception_logs_failure_without_action(model, tmp_path):
    world = panda_visual_reach(model)
    provider = GroundTruthObservationProvider(world)
    with PhysicalAgent(world, provider, GroundTruthPerception(), SemanticReachTask(SemanticGoal("red")), RoboticsPlanner(world)) as agent:
        with pytest.raises(ValueError, match="expected one"):
            agent.run(tmp_path)
    summary = json.loads(next(tmp_path.glob("*/summary.json")).read_text())
    assert summary["decisions"] == 0 and summary["error"]
    assert world.steps == 0


def test_visual_annotations_do_not_change_physics(model):
    from ddm_mcts.robotics.reach import ReachTask, panda_reach

    plain = panda_reach(model, ReachTask((0.8, 0.2, 0.5)))
    visual = panda_visual_reach(model)
    assert plain.backend.snapshot().values == visual.backend.snapshot().values
    for action in plain.actions[:3]:
        assert plain.step(action) == visual.step(action)
        assert plain.backend.snapshot().values == visual.backend.snapshot().values


def test_ground_truth_projection_matches_phase2_planning(model):
    from ddm_mcts.robotics.reach import ReachTask, panda_reach
    from ddm_mcts.robotics.representation import CoordinateReachTask

    world = panda_reach(model, ReachTask((0.5945, 0.02, 0.6245)))
    config = MCTSConfig(30, seed=0)
    legacy = RoboticsPlanner(world, config=config).plan()
    with PhysicalAgent(
        world, GroundTruthObservationProvider(world), GroundTruthPerception(), CoordinateReachTask(), RoboticsPlanner(world, config=config)
    ) as agent:
        observed = agent.plan()
    assert legacy.action == observed.action
    assert legacy.root_statistics() == observed.root_statistics()


def test_reuse_agent_and_completed_run_logs_are_immutable(model, tmp_path):
    world = panda_visual_reach(model)
    with PhysicalAgent(
        world,
        GroundTruthObservationProvider(world, world.target_entities),
        GroundTruthPerception(),
        SemanticReachTask(SemanticGoal("red"), max_steps=1),
        RoboticsPlanner(world, config=MCTSConfig(10, seed=0)),
    ) as agent:
        folder = agent.run(tmp_path)
        original = {p.name: p.read_bytes() for p in folder.iterdir() if p.is_file()}
        agent.reset(SemanticReachTask(SemanticGoal("blue"), max_steps=1))
        state = agent.prepare()
        assert state.semantic_goal.label == "blue"
        assert original == {p.name: p.read_bytes() for p in folder.iterdir() if p.is_file()}
        second = agent.run(tmp_path)
        assert second != folder
        assert json.loads((second / "config.json").read_text())["agent"]["semantic_task"]["goal"]["label"] == "blue"
        assert original == {p.name: p.read_bytes() for p in folder.iterdir() if p.is_file()}
