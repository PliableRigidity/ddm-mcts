# ruff: noqa: E402
import json
import os
from dataclasses import replace
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("mujoco")

from test_core import LineWorld
from test_vlm import FakeRuntime, make_observation

from ddm_mcts.robotics.cli import build_parser
from ddm_mcts.robotics.core import RoboticsPlanner, SearchState
from ddm_mcts.robotics.observation import SemanticGoal
from ddm_mcts.robotics.qwen_visual_policy import QwenVisualPolicy
from ddm_mcts.robotics.reach import cartesian_actions
from ddm_mcts.robotics.vlm_backend import Qwen3VLBackend


class DecisionRuntime(FakeRuntime):
    def generate(self, rgb, prompt, config):
        self.requests.append((rgb.copy(), prompt))
        if "Candidates:" not in prompt:
            return json.dumps({"target_label": "cube", "target_description": "cube", "bbox_2d": [370, 500, 443, 610]})
        candidates = json.loads(prompt.split("Candidates: ")[1].split("\nRequired JSON")[0])
        return json.dumps({"scores": {a["id"]: (100 if a["id"] == "MOVE_X_NEG" else 1) for a in candidates}})


def test_alpha_permutations_cache_camera_mapping_and_no_target_leak():
    backend = Qwen3VLBackend(runtime_factory=DecisionRuntime)
    policy = QwenVisualPolicy(backend, alpha=0.5, permutations=3)
    observation = make_observation()
    policy.bind(observation, SemanticGoal("Reach the cube."))
    actions = cartesian_actions()
    state = SearchState(0, 0)
    scores = policy.probabilities(state, tuple(reversed(actions)))
    assert sum(scores.values()) == pytest.approx(1)
    assert scores[actions[1]] == pytest.approx(0.5 / 6 + 0.5 * 100 / 105)
    assert backend.calls == 3 and backend.load_count == 1
    records = policy.last["presentations"]
    assert len({tuple(r["presentation_order"]) for r in records}) == 3
    assert records[0]["presentation_order"] == sorted(a.name for a in actions)
    assert policy.probabilities(state, actions) == scores and policy.hits == 1
    assert backend.calls == 3
    prompt = backend.runtime.requests[0][1]
    assert "99" not in prompt and "semantic_object" not in prompt and "target XYZ" not in prompt
    assert "world X -0.020 m" in prompt and "camera_motion_pixels" in prompt
    # Camera above origin: world +X projects right, world +Y projects up.
    assert policy.model.describe(actions[0])["camera_motion_pixels"]["right"] > 0
    assert policy.model.describe(actions[2])["camera_motion_pixels"]["down"] < 0
    assert np.array_equal(observation.rgb, make_observation().rgb)  # Overlay is on a copy.
    policy.bind(replace(observation, sequence=2), SemanticGoal("Reach the cube."))
    policy.probabilities(state, actions)
    assert backend.calls == 6 and backend.load_count == 1
    with pytest.raises(ValueError, match="root_policy"):
        policy.probabilities(SearchState(0, 0, 1), actions)
    ignored = QwenVisualPolicy(backend, alpha=0)
    ignored.bind(observation, SemanticGoal("cube"))
    assert all(p == pytest.approx(1 / 6) for p in ignored.probabilities(state, actions).values())
    assert backend.calls == 6


def test_actual_mcts_overrides_bad_visual_top1_and_deep_nodes_are_uniform():
    actions = cartesian_actions(axes=(0,))

    class World(LineWorld):
        def legal_actions(self):
            return () if self.task.is_terminal(self.value) else actions

        def step(self, action):
            self.value += 1 if action == actions[0] else -1
            return self.value

    from ddm_mcts.search.mcts import MCTSConfig

    world = World()
    backend = Qwen3VLBackend(runtime_factory=DecisionRuntime)
    visual = QwenVisualPolicy(backend, alpha=1)
    visual.bind(make_observation(), SemanticGoal("Reach the cube."))
    planner = RoboticsPlanner(world, config=MCTSConfig(600, seed=0))
    planner.search.root_policy = visual
    result = planner.plan()
    assert visual.last["model_top1"] == "MOVE_X_NEG"
    assert result.action == actions[0]
    assert result.root.children[actions[1]].prior == pytest.approx(100 / 101)
    assert backend.calls == 1 and visual.calls == 1
    assert world.value == 0
    expanded = [child for child in result.root.children.values() if child.children]
    assert expanded
    assert all(grandchild.prior == pytest.approx(0.5) for child in expanded for grandchild in child.children.values())


def test_shared_backend_closed_loop_root_only_and_logs(tmp_path):
    model = os.environ.get("PANDA_MODEL")
    if not model or not Path(model).is_file():
        pytest.skip("PANDA_MODEL required")
    from ddm_mcts.robotics.approach import ApproachReachTask
    from ddm_mcts.robotics.camera import MujocoCameraObservationProvider
    from ddm_mcts.robotics.physical_agent import PhysicalAgent
    from ddm_mcts.robotics.semantic_scene import OBJECT_RGB, SEMANTIC_PLANE_Z, panda_semantic_reach
    from ddm_mcts.robotics.vlm_perception import VLMSemanticPerception
    from ddm_mcts.search.mcts import MCTSConfig

    world = panda_semantic_reach(model)
    backend = Qwen3VLBackend(runtime_factory=DecisionRuntime)
    visual = QwenVisualPolicy(backend)
    with PhysicalAgent(
        world,
        MujocoCameraObservationProvider(world, width=640, height=480),
        VLMSemanticPerception(backend, SEMANTIC_PLANE_Z, OBJECT_RGB),
        ApproachReachTask(SemanticGoal("Reach the cube."), max_steps=2),
        RoboticsPlanner(world, config=MCTSConfig(20, seed=0)),
        visual_policy=visual,
    ) as agent:
        folder = agent.run(tmp_path)
        assert agent.observer.sequence == 3
        assert backend.calls == 3 and backend.load_count == 1  # One semantic + two visual decisions, never simulations.
        assert visual.calls == 2
        summary = json.loads((folder / "summary.json").read_text())
        assert summary["policy_diagnostics"]["direct_visual"]["physical_visual_inference_calls"] == 2
        rows = [json.loads(s) for s in (folder / "steps.jsonl").read_text().splitlines()]
        assert len(rows) == 2
        assert rows[0]["visual_decision"]["observation_sequence"] != rows[1]["visual_decision"]["observation_sequence"]
        for row in rows:
            priors = row["visual_decision"]["actual_root_priors"]
            assert all(r["prior"] == pytest.approx(priors[r["action"]]) for r in row["root_statistics"])
            assert row["mcts_search_seconds"] >= 0


def test_cli_visual_policy_is_distinct_from_perception():
    parser = build_parser()
    args = parser.parse_args(["--decision-policy", "visual", "--visual-alpha", ".25", "--visual-permutations", "2"])
    assert args.decision_policy == "visual" and args.visual_alpha == 0.25 and args.visual_permutations == 2
    assert args.perception is None
    assert parser.parse_args([]).decision_policy == "structured"
