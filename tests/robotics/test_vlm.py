# ruff: noqa: E402
import builtins
import json
import os
from dataclasses import replace
from math import dist
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

from ddm_mcts.robotics.observation import Entity, SemanticGoal
from ddm_mcts.robotics.vlm_backend import Qwen3VLBackend, VLMConfig


class FakeRuntime:
    device, dtype = "fake-cuda", "fake-bfloat16"

    def __init__(self, config):
        self.requests = []

    def generate(self, rgb, prompt, config):
        self.requests.append((rgb.copy(), prompt))
        return json.dumps(
            {"target_label": "cube", "target_description": "cube from pixels", "bbox_2d": [370, 500, 443, 610], "confidence": 0.9}
        )


def test_backend_lazy_once_rgb_goal_and_metrics():
    backend = Qwen3VLBackend(runtime_factory=FakeRuntime)
    assert backend.runtime is None
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    backend.infer(image, "Reach the cube.")
    backend.infer(image, "Reach the sphere.")
    assert backend.load_count == 1 and backend.calls == 2
    assert len(backend.runtime.requests) == 2
    assert "Reach the sphere." in backend.runtime.requests[-1][1]
    assert "Never output world coordinates" in backend.runtime.requests[-1][1]
    assert backend.diagnostics()["total_inference_seconds"] >= 0
    with pytest.raises(ValueError):
        backend.infer(image.astype(float), "cube")
    with pytest.raises(ValueError):
        backend.infer(image, "")


def test_missing_optional_dependencies_and_runtime_failures(monkeypatch):
    real_import = builtins.__import__

    def importing(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("test missing torch")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", importing)
    with pytest.raises(RuntimeError, match="optional extra"):
        Qwen3VLBackend().infer(np.zeros((10, 10, 3), dtype=np.uint8), "cube")

    class FailingRuntime(FakeRuntime):
        def generate(self, *args):
            raise RuntimeError("CUDA out of memory")

    backend = Qwen3VLBackend(runtime_factory=FailingRuntime)
    with pytest.raises(RuntimeError, match="out of memory"):
        backend.infer(np.zeros((10, 10, 3), dtype=np.uint8), "cube")
    assert backend.calls == 1 and len(backend.inference_seconds) == 1


def make_observation():
    from ddm_mcts.robotics.camera import CameraCalibration
    from ddm_mcts.robotics.observation import Observation
    from ddm_mcts.robotics.reach import ReachState

    image = np.zeros((480, 640, 3), dtype=np.uint8)
    image[:] = (60, 90, 140)  # Blue floor must not pollute the teal foreground centroid.
    image[245:280, 245:280] = (30, 160, 140)
    camera = CameraCalibration(640, 480, 400, (0, 0, 2), (1, 0, 0, 0, 1, 0, 0, 0, 1), "test")
    robot = ReachState((0, 0, 0), (99, 99, 99), 0, 0, (), ())
    return Observation(0, 1, "camera", robot, (Entity("cube", (99, 99, 99)),), image, camera)


def test_semantic_cache_refresh_geometry_and_no_privileged_answers(tmp_path):
    pytest.importorskip("mujoco")
    from ddm_mcts.robotics.vlm_perception import VLMSemanticPerception

    backend = Qwen3VLBackend(runtime_factory=FakeRuntime)
    perception = VLMSemanticPerception(backend, 0, (0.15, 0.8, 0.7), refresh=2, debug=True)
    observation = make_observation()
    goal = SemanticGoal("choose the cube")
    state = perception.perceive(observation, goal)
    assert state.goal == pytest.approx(observation.camera.intersect_plane((262, 262), 0))
    assert state.target().label == "cube" and state.semantic_goal == goal and state.goal != (99, 99, 99)
    perception.perceive(replace(observation, sequence=2), goal)
    assert backend.calls == 1
    perception.perceive(replace(observation, sequence=3), goal)
    assert backend.calls == 2
    perception.perceive(observation, SemanticGoal("different goal"))
    assert backend.calls == 3 and backend.load_count == 1
    perception.save_debug(tmp_path, observation)
    assert (tmp_path / "vlm_debug/observation_0001_grounding.png").exists()
    blank = replace(observation, rgb=np.zeros_like(observation.rgb))
    perception.refresh = 0
    assert not perception.perceive(blank, SemanticGoal("different goal")).target().observed
    perception.reset()
    with pytest.raises(ValueError, match="no valid foreground"):
        perception.perceive(blank, goal)


def test_cli_and_defaults():
    from ddm_mcts.robotics.cli import build_parser

    args = build_parser().parse_args(["--task", "semantic-reach", "--perception", "vlm", "--goal", "Reach the cube.", "--vlm-refresh", "3"])
    assert args.goal == "Reach the cube." and args.vlm_refresh == 3
    assert args.vlm_model == VLMConfig().model_id
    assert build_parser().parse_args([]).task == "reach"


def test_mock_vlm_closed_loop_real_mcts_and_logging(tmp_path):
    pytest.importorskip("mujoco")
    model = os.environ.get("PANDA_MODEL")
    if not model or not Path(model).is_file():
        pytest.skip("PANDA_MODEL required for VLM physical integration")
    from ddm_mcts.policies.text_laya_policy import TextLayaPolicy
    from ddm_mcts.robotics.camera import MujocoCameraObservationProvider
    from ddm_mcts.robotics.cli import state_text
    from ddm_mcts.robotics.core import RoboticsPlanner
    from ddm_mcts.robotics.physical_agent import PhysicalAgent
    from ddm_mcts.robotics.representation import SemanticReachTask
    from ddm_mcts.robotics.semantic_scene import OBJECT_RGB, SEMANTIC_PLANE_Z, panda_semantic_reach
    from ddm_mcts.robotics.vlm_perception import VLMSemanticPerception
    from ddm_mcts.search.mcts import MCTSConfig

    # Preserve coverage of the original centroid/hand-frame API explicitly;
    # safe semantic-reach defaults are exercised by test_approach.py.
    world = panda_semantic_reach(model, use_tcp=False)
    assert len({world.backend.model.geom_matid[world.backend.model.geom(f"semantic_object_{index}").id] for index in range(3)}) == 1
    backend = Qwen3VLBackend(runtime_factory=FakeRuntime)
    ddm_calls = []

    def predictor(text, question):
        ddm_calls.append(text)
        return {"answers": {"action": {"probabilities": {key: 1 for key in question["action"]["criteria"]}}}}

    policy = TextLayaPolicy(state_text, objective="Reach the perceived object", predictor=predictor)
    camera = MujocoCameraObservationProvider(world, width=640, height=480)
    with PhysicalAgent(
        world,
        camera,
        VLMSemanticPerception(backend, SEMANTIC_PLANE_Z, OBJECT_RGB),
        SemanticReachTask(SemanticGoal("cube")),
        RoboticsPlanner(world, policy, MCTSConfig(60, seed=0)),
        diagnostics=world.target_entities,
    ) as agent:
        before = world.snapshot()
        result = agent.plan()
        assert result.simulations == 60 and world.snapshot() == before and camera.sequence == 1
        assert backend.calls == 1 and ddm_calls
        folder = agent.run(tmp_path)
        summary = json.loads((folder / "summary.json").read_text())
        assert summary["success"] and backend.load_count == 1
        truth = next(e for e in world.target_entities() if e.label == "cube")
        assert dist(world.get_state().position, truth.position) < 0.025
        assert summary["perception_diagnostics"]["backend"]["inference_count"] == 1
        assert not (folder / "vlm_debug").exists()
        assert camera.sequence == summary["decisions"] + 2  # manual plan + ordinary closed-loop observations


def test_official_runtime_image_conversion_and_generated_token_trimming():
    pytest.importorskip("PIL")
    from contextlib import nullcontext
    from types import SimpleNamespace

    from ddm_mcts.robotics.vlm_backend import _TransformersRuntime

    class Inputs(dict):
        input_ids = np.array([[1, 2, 3]])

        def to(self, device):
            assert device == "fake"
            return self

    class Processor:
        def apply_chat_template(self, messages, **kwargs):
            image = messages[0]["content"][0]["image"]
            assert image.mode == "RGB" and image.size == (20, 10)
            assert messages[0]["content"][1]["text"] == "language goal"
            assert kwargs["return_dict"] and kwargs["tokenize"]
            return Inputs()

        def batch_decode(self, trimmed, **kwargs):
            np.testing.assert_array_equal(trimmed, [[9, 10]])
            return ["structured response"]

    runtime = _TransformersRuntime.__new__(_TransformersRuntime)
    runtime.processor = Processor()
    runtime.torch = SimpleNamespace(inference_mode=nullcontext)
    runtime.model = SimpleNamespace(device="fake", generate=lambda **kw: np.array([[1, 2, 3, 9, 10]]))
    runtime.sync = lambda: None
    assert runtime.generate(np.zeros((10, 20, 3), dtype=np.uint8), "language goal", VLMConfig()) == "structured response"
