# ruff: noqa: E402
import json
import os
from dataclasses import replace
from pathlib import Path

import pytest

pytest.importorskip("mujoco")
np = pytest.importorskip("numpy")

from ddm_mcts.robotics.approach import ApproachReachTask
from ddm_mcts.robotics.core import RoboticsPlanner
from ddm_mcts.robotics.observation import GroundTruthObservationProvider, SemanticGoal
from ddm_mcts.robotics.ordered_task import ApproachVerifier, parse_physical_task
from ddm_mcts.robotics.perception import GroundTruthPerception
from ddm_mcts.robotics.physical_agent import PhysicalAgent
from ddm_mcts.robotics.semantic_scene import panda_semantic_reach
from ddm_mcts.search.mcts import MCTSConfig


@pytest.fixture
def agent():
    model = os.environ.get("PANDA_MODEL")
    if not model or not Path(model).is_file():
        pytest.skip("PANDA_MODEL required")
    world = panda_semantic_reach(model)
    with PhysicalAgent(
        world,
        GroundTruthObservationProvider(world, world.target_entities),
        GroundTruthPerception(),
        ApproachReachTask(SemanticGoal("cylinder")),
        RoboticsPlanner(world, config=MCTSConfig(60, seed=0)),
        diagnostics=world.target_entities,
    ) as value:
        yield value


def test_full_ordered_task_no_resets_fresh_roots_and_all_object_substep_clearance(agent, tmp_path, monkeypatch):
    def forbidden(*args):
        raise AssertionError("a multi-step task must never reset physical state")

    monkeypatch.setattr(agent.environment, "reset", forbidden)
    monkeypatch.setattr(agent.environment.backend, "reset", forbidden)
    roots = []
    plan = agent.planner.plan

    def traced_plan(*args, **kwargs):
        result = plan(*args, **kwargs)
        roots.append(result.root)
        return result

    monkeypatch.setattr(agent.planner, "plan", traced_plan)
    result = agent.run_task(parse_physical_task("Approach the cylinder, then the sphere, then the cube."), tmp_path)
    assert result.success, result.trace
    assert len({id(r) for r in roots}) == len(roots)
    rows = result.trace["subgoals"]
    assert [r["goal"]["target"] for r in rows] == ["cylinder", "sphere", "cube"]
    assert all(r["status"] == "succeeded" for r in rows)
    for index, row in enumerate(rows):
        assert row["start_physical_state"] == row["prepared_physical_state"]
        if index:
            assert rows[index - 1]["end_physical_state"] == row["start_physical_state"]
        assert row["verification"]["tcp_error_m"] <= row["goal"]["epsilon"]
        assert row["verification"]["minimum_transit_clearance_m"] > 0
        assert len(row["verification"]["final_clearances_m"]) == 3
        assert row["monitored_physics_substeps"] == len(row["actions"]) * 150
    assert agent.observer.sequence == sum(len(r["actions"]) + 1 for r in rows)
    assert agent.environment.steps == sum(len(r["actions"]) for r in rows)
    assert json.loads((result.folder / "task_trace.json").read_text())["success"]


def test_failure_stops_and_remaining_subgoal_is_pending(agent, tmp_path):
    result = agent.run_task(parse_physical_task("Visit cylinder, sphere, cube", max_actions=1), tmp_path)
    assert not result.success and result.trace["failed_subgoal"] == "goal-1"
    assert [r["status"] for r in result.trace["subgoals"]] == ["failed", "pending", "pending"]
    assert agent.environment.steps == 1
    assert result.trace["subgoals"][0]["verification"]["reason"] == "TCP outside final waypoint tolerance"


def test_verifier_uses_final_not_lift_waypoint_and_semantic_identity(agent):
    state = agent.prepare()
    agent.state = replace(state, planning_goal=state.position)
    verifier = ApproachVerifier()
    goal = parse_physical_task("Visit cylinder").subgoals[0]
    result = verifier.verify(agent, goal)
    assert not result.success
    assert result.tcp_error_m > goal.epsilon
    wrong_goal = replace(goal, target="cube")
    assert verifier.verify(agent, wrong_goal).reason == "semantic target mismatch"


def test_second_order_and_duplicate_targets(agent, tmp_path):
    result = agent.run_task(parse_physical_task("Go to cube and then cylinder and then cylinder"), tmp_path)
    assert result.success
    rows = result.trace["subgoals"]
    assert [r["goal"]["target"] for r in rows] == ["cube", "cylinder", "cylinder"]
    assert rows[2]["actions"] == []  # Already at this physically verified waypoint.
    assert rows[2]["final_observation_sequence"] > rows[1]["final_observation_sequence"]


def test_one_viewer_session_and_nested_execution_callbacks(agent, tmp_path):
    from test_visual import FakeViewer

    from ddm_mcts.robotics.visual import VisualInspection

    launches = []

    def launch(model, data):
        launches.append(data)
        return FakeViewer(model, data)

    with VisualInspection(agent.environment, agent.planner, speed=10000, launch=launch) as viewer:
        result = agent.run_task(parse_physical_task("Visit cube and cylinder"), tmp_path, visualization=viewer)
        assert result.success
        assert len(launches) == 1
        assert viewer.viewer.is_running()
        assert all(r["monitored_physics_substeps"] == len(r["actions"]) * 150 for r in result.trace["subgoals"])
    assert agent.environment.backend._step_callback is None


def test_verifier_rejects_negative_clearance(agent, monkeypatch):
    agent.prepare()
    monkeypatch.setattr(
        agent.environment,
        "get_state",
        lambda: replace(agent.state.robot, position=agent.task.diagnostics(agent.state)["pre_contact_target"]),
    )
    verifier = ApproachVerifier()
    goal = parse_physical_task("Visit cylinder").subgoals[0]
    assert verifier.verify(agent, goal).success
    monkeypatch.setattr(verifier, "clearances", lambda environment: {"sphere": -0.001})
    assert verifier.verify(agent, goal).reason == "invalid geometry clearance"


def test_task_failure_on_perception_error_preserves_trace(agent, tmp_path, monkeypatch):
    def fail(*args):
        raise ValueError("missing target")

    monkeypatch.setattr(agent.perception, "perceive", fail)
    result = agent.run_task(parse_physical_task("Visit cylinder then sphere"), tmp_path)
    assert not result.success
    assert "missing target" in result.trace["error"]
    assert result.trace["subgoals"][1]["status"] == "pending"
    assert agent.environment.steps == 0


def test_cli_multi_task_argument_validation():
    from ddm_mcts.robotics.cli import build_parser, main

    args = build_parser().parse_args(["--task", "multi-semantic-reach", "--instruction", "Visit sphere, cube"])
    assert args.task == "multi-semantic-reach" and args.instruction == "Visit sphere, cube"
    assert build_parser().parse_args([]).task == "reach"
    for arguments in (
        ["--task", "multi-semantic-reach"],
        ["--instruction", "Visit cube"],
        ["--task", "multi-semantic-reach", "--instruction", "Grasp cube"],
    ):
        with pytest.raises(SystemExit):
            main(arguments)


def test_completed_results_preserved_when_later_goal_fails(agent, tmp_path):
    task = parse_physical_task("Visit cylinder, sphere")
    task = replace(task, subgoals=(task.subgoals[0], replace(task.subgoals[1], max_actions=1)))
    result = agent.run_task(task, tmp_path)
    assert not result.success
    assert [r["status"] for r in result.trace["subgoals"]] == ["succeeded", "failed"]
    assert result.trace["subgoals"][0]["verification"]["success"]
    assert result.trace["subgoals"][1]["start_physical_state"] == result.trace["subgoals"][0]["end_physical_state"]


def test_episode_success_flag_cannot_advance_unverified_goal(agent, tmp_path, monkeypatch):
    fake_episode = tmp_path / "fake_episode"
    fake_episode.mkdir()
    (fake_episode / "summary.json").write_text(json.dumps({"success": True, "total_mcts_search_seconds": 0, "total_execution_seconds": 0}))
    (fake_episode / "steps.jsonl").write_text("")
    monkeypatch.setattr(agent, "run", lambda *args, **kwargs: fake_episode)
    result = agent.run_task(parse_physical_task("Visit cylinder, sphere"), tmp_path)
    assert not result.success
    assert result.trace["subgoals"][1]["status"] == "pending"


def test_semantic_and_visual_goal_caches_refresh_without_reloading_model():
    from test_vlm import FakeRuntime, make_observation

    from ddm_mcts.robotics.core import SearchState
    from ddm_mcts.robotics.qwen_visual_policy import QwenVisualPolicy
    from ddm_mcts.robotics.reach import cartesian_actions
    from ddm_mcts.robotics.vlm_backend import Qwen3VLBackend
    from ddm_mcts.robotics.vlm_perception import VLMSemanticPerception

    class Runtime(FakeRuntime):
        def generate(self, rgb, prompt, config):
            self.requests.append((rgb.copy(), prompt))
            if "Candidates:" in prompt:
                return json.dumps({"scores": {a.name: 1 for a in cartesian_actions()}})
            label = prompt.split("Goal: ")[1].split(".")[0]
            return json.dumps({"target_label": label, "target_description": "from fake image", "bbox_2d": [370, 500, 443, 610]})

    backend = Qwen3VLBackend(runtime_factory=Runtime)
    perception = VLMSemanticPerception(backend, 0, (0.15, 0.8, 0.7))
    visual = QwenVisualPolicy(backend)
    observation = make_observation()
    for i, label in enumerate(("cylinder", "sphere", "cube")):
        live = replace(observation, sequence=i + 1)
        goal = SemanticGoal(label)
        perceived = perception.perceive(live, goal)
        assert perceived.target().label == label
        perception.perceive(live, goal)  # Semantic cache within this goal.
        visual.bind(live, goal)
        visual.probabilities(SearchState(0, 0), cartesian_actions())
        assert visual.last["goal"] == label
        assert visual.last["observation_sequence"] == i + 1
        assert len(visual.model.records) == 1
    assert backend.load_count == 1 and backend.calls == 6
    assert perception.requests == 6 and perception.inference_calls == 3 and perception.cache_hits == 3
    assert visual.calls == 3 and len(visual.inference_times) == 3
    perception.reset()
    perception.perceive(observation, SemanticGoal("cube"))
    assert backend.load_count == 1 and perception.inference_calls == 4
    assert perception.requests == 7  # Reset clears goals, not session accounting.


def test_mislabeled_grounding_cannot_pass_true_target_verification(agent, monkeypatch):
    state = agent.prepare()
    # Claim a cylinder at a different location. The evaluator may reach the
    # perceived waypoint, but physical verification must reject that mismatch.
    incorrect = replace(state.target(), position=(0.7, -0.2, 0.6245))
    agent.state = replace(state, entities=(incorrect,))
    alleged_waypoint = agent.task.diagnostics(agent.state)["pre_contact_target"]
    monkeypatch.setattr(agent.environment, "get_state", lambda: replace(state.robot, position=alleged_waypoint))
    result = ApproachVerifier().verify(agent, parse_physical_task("Visit cylinder").subgoals[0])
    assert result.tcp_error_m == 0
    assert not result.success
    assert result.true_waypoint_error_m > 0.1
    assert result.reason == "TCP outside requested object's true approach tolerance"


def test_mock_qwen_full_multi_step_camera_visual_mcts_and_session_accounting(agent, tmp_path):
    from test_vlm import FakeRuntime

    from ddm_mcts.robotics.camera import MujocoCameraObservationProvider
    from ddm_mcts.robotics.qwen_visual_policy import QwenVisualPolicy
    from ddm_mcts.robotics.reach import cartesian_actions
    from ddm_mcts.robotics.semantic_scene import OBJECT_RGB, SEMANTIC_PLANE_Z
    from ddm_mcts.robotics.vlm_backend import Qwen3VLBackend
    from ddm_mcts.robotics.vlm_perception import VLMSemanticPerception

    class Runtime(FakeRuntime):
        """Mock grounding fixtures, not a claim of real model recognition."""

        def generate(self, rgb, prompt, config):
            self.requests.append((rgb.copy(), prompt))
            if "Candidates:" in prompt:
                return json.dumps({"scores": {a.name: 100 if a.name == "MOVE_Z_POS" else 1 for a in cartesian_actions()}})
            label = prompt.split("Goal: ")[1].split(".")[0]
            # Image-space grounding fixtures for the fixed inspection scene.
            boxes = {"cube": [370, 500, 445, 610], "cylinder": [570, 400, 625, 505], "sphere": [490, 605, 550, 690]}
            return json.dumps({"target_label": label, "target_description": "mock shape grounding", "bbox_2d": boxes[label]})

    backend = Qwen3VLBackend(runtime_factory=Runtime)
    visual = QwenVisualPolicy(backend)
    world = agent.environment
    with PhysicalAgent(
        world,
        MujocoCameraObservationProvider(world, width=640, height=480),
        VLMSemanticPerception(backend, SEMANTIC_PLANE_Z, OBJECT_RGB),
        agent.task,
        RoboticsPlanner(world, config=MCTSConfig(60, c_puct=0.05, seed=0)),
        visual_policy=visual,
    ) as combined:
        result = combined.run_task(parse_physical_task("Visit cylinder, sphere, cube"), tmp_path)
        assert result.success, result.trace["error"]
        rows = result.trace["subgoals"]
        actions = sum(len(r["actions"]) for r in rows)
        assert backend.load_count == 1 and backend.calls == actions + 3
        assert visual.calls == actions and combined.observer.sequence == actions + 3
        assert result.trace["accounting"]["physical_perception_inference_calls"] == 3
        assert result.trace["accounting"]["physical_visual_inference_calls"] == actions
        assert sum(r["mcts_overrides"] for r in rows) > 0
        assert all(r["preparation_preserved_snapshot"] for r in rows)
        for index, row in enumerate(rows):
            if index:
                assert row["first_observation_sequence"] > rows[index - 1]["final_observation_sequence"]
            assert row["accounting"]["physical_perception_inference_calls"] == 1
            assert row["accounting"]["physical_visual_inference_calls"] == len(row["actions"])
            steps = [json.loads(line) for line in (Path(row["run"]) / "steps.jsonl").read_text().splitlines()]
            assert all(
                all(
                    stat["prior"] == pytest.approx(step["visual_decision"]["actual_root_priors"][stat["action"]])
                    for stat in step["root_statistics"]
                )
                for step in steps
            )
        assert all("semantic_object" not in prompt and "0.6045" not in prompt for _, prompt in backend.runtime.requests)


def test_cli_multi_viewer_holds_only_after_whole_task(agent, tmp_path, monkeypatch):
    import mujoco.viewer
    from test_visual import FakeViewer
    from test_vlm import FakeRuntime

    from ddm_mcts.robotics import vlm_backend
    from ddm_mcts.robotics.cli import main
    from ddm_mcts.robotics.visual import VisualInspection

    original_backend = vlm_backend.Qwen3VLBackend
    models, windows, holds = [], [], []

    def backend(config):
        value = original_backend(config, runtime_factory=FakeRuntime)
        models.append(value)
        return value

    def launch(model, data):
        viewer = FakeViewer(model, data)
        windows.append(viewer)
        return viewer

    def hold(self):
        trace = json.loads(next(tmp_path.glob("*/task_trace.json")).read_text())
        assert trace["success"] and all(r["status"] == "succeeded" for r in trace["subgoals"])
        assert trace["subgoals"][0]["end_physical_state"] == trace["subgoals"][1]["start_physical_state"]
        assert self.viewer.is_running()
        holds.append(True)
        self.viewer.close()

    monkeypatch.setattr(vlm_backend, "Qwen3VLBackend", backend)
    monkeypatch.setattr(mujoco.viewer, "launch_passive", launch)
    monkeypatch.setattr(VisualInspection, "wait_until_closed", hold)
    assert (
        main(
            [
                "--model",
                os.environ["PANDA_MODEL"],
                "--task",
                "multi-semantic-reach",
                "--instruction",
                "Visit cube and cube",
                "--viewer",
                "--viewer-speed",
                "10000",
                "--output",
                str(tmp_path),
            ]
        )
        == 0
    )
    assert len(windows) == len(holds) == len(models) == 1
    assert not windows[0].is_running()
    assert models[0].load_count == 1 and models[0].calls == 2
