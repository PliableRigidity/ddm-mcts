# ruff: noqa: E402
import json
import os
from math import dist
from pathlib import Path

import pytest

mujoco = pytest.importorskip("mujoco")
np = pytest.importorskip("numpy")

from ddm_mcts.robotics.approach import ApproachReachTask
from ddm_mcts.robotics.core import RoboticsPlanner
from ddm_mcts.robotics.observation import GroundTruthObservationProvider, SemanticGoal
from ddm_mcts.robotics.perception import GroundTruthPerception
from ddm_mcts.robotics.physical_agent import PhysicalAgent
from ddm_mcts.robotics.semantic_scene import panda_semantic_reach
from ddm_mcts.search.mcts import MCTSConfig


@pytest.mark.parametrize("label", ["cylinder", "cube", "sphere"])
def test_approach_tcp_outside_geometry_and_physical_clearance(label, tmp_path):
    model = os.environ.get("PANDA_MODEL")
    if not model or not Path(model).is_file():
        pytest.skip("PANDA_MODEL required")
    world = panda_semantic_reach(model, use_tcp=True)
    task = ApproachReachTask(SemanticGoal(label))
    observer = GroundTruthObservationProvider(world, world.target_entities)
    with PhysicalAgent(
        world,
        observer,
        GroundTruthPerception(),
        task,
        RoboticsPlanner(world, config=MCTSConfig(60, seed=0)),
        diagnostics=world.target_entities,
    ) as agent:
        state = agent.prepare()
        assert dist(task.diagnostics(state)["pre_contact_target"], state.target().position) == pytest.approx(0.09)
        assert state.goal[2] > state.target().position[2] + 0.028
        assert state.goal != state.target().position
        assert world.controller.site is not None
        site = world.backend.model.site("panda_gripper_tcp").id
        assert tuple(world.backend.model.site_pos[site]) == pytest.approx((0, 0, 0.1034))
        np.testing.assert_allclose(world.controller.position(), world.backend.data.site_xpos[site])
        substep_distances = []

        class Monitor:
            def before_plan(self, *args):
                pass

            def after_plan(self, *args):
                pass

            def after_step(self, *args):
                pass

            def finish(self, *args):
                pass

            def execute(self, environment, action):
                def capture():
                    substep_distances.append(environment.geometry_diagnostics(label)["minimum_gripper_target_distance_m"])

                with environment.backend.observe_steps(capture):
                    return environment.step(action)

        folder = agent.run(tmp_path, visualization=Monitor())
        summary = json.loads((folder / "summary.json").read_text())
        assert summary["success"]
        diag = summary["execution_diagnostics"]
        assert diag["final_tcp_error_m"] <= task.epsilon
        assert diag["minimum_gripper_target_distance_m"] > 0.01
        steps = [json.loads(line) for line in (folder / "steps.jsonl").read_text().splitlines()]
        assert all(row["execution_diagnostics"]["minimum_gripper_target_distance_m"] > 0 for row in steps)
        assert all(row["requested_end_effector_target"] for row in steps)
        assert len(substep_distances) == summary["decisions"] * world.backend.physics_steps
        assert min(substep_distances) > 0


def test_approach_validation():
    with pytest.raises(ValueError):
        ApproachReachTask(SemanticGoal("cube"), standoff=0)
    with pytest.raises(ValueError):
        ApproachReachTask(SemanticGoal("cube"), approach_vector=(0, 0, 0))
