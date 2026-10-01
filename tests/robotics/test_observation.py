from dataclasses import asdict

import pytest

from ddm_mcts.robotics.observation import Entity, GroundTruthObservationProvider, Observation, SemanticGoal
from ddm_mcts.robotics.perception import GroundTruthPerception, ModelPerceptionAdapter
from ddm_mcts.robotics.reach import ReachState
from ddm_mcts.robotics.representation import SemanticReachTask, WorldState


class World:
    def get_state(self):
        return ReachState((0, 0, 0), (1, 0, 0), 0, 0.1, (), ())


def test_ground_truth_observation_and_representation():
    provider = GroundTruthObservationProvider(World(), lambda: (Entity("red", (1, 0, 0)), Entity("blue", (0, 1, 0))))
    observation = provider.observe()
    assert isinstance(observation, Observation)
    assert observation.rgb is None
    assert observation.timestamp == 0.1
    assert provider.observe().sequence == 2
    state = GroundTruthPerception().perceive(observation, SemanticGoal("red"))
    assert state.goal == (1, 0, 0)
    assert state.position == (0, 0, 0)
    assert asdict(state)["semantic_goal"]["label"] == "red"
    assert SemanticReachTask(SemanticGoal("red")).resolve(state).evaluate(state) == -1
    assert state.predict_observation(World().get_state()).entities == state.entities
    assert GroundTruthPerception().perceive(observation).goal == (1, 0, 0)


def test_model_perception_boundary_and_missing_goal():
    observation = GroundTruthObservationProvider(World()).observe()
    expected = WorldState(observation.robot, (Entity("red", (1, 0, 0)),), SemanticGoal("red"), "mock-vlm")
    model = ModelPerceptionAdapter(lambda obs, goal: expected)
    assert model.perceive(observation, SemanticGoal("red")) == expected
    with pytest.raises(ValueError, match="mismatched"):
        model.perceive(observation, SemanticGoal("blue"))
    with pytest.raises(TypeError):
        ModelPerceptionAdapter(lambda obs, goal: {}).perceive(observation)
    with pytest.raises(ValueError, match="expected one"):
        GroundTruthPerception().perceive(observation, SemanticGoal("missing"))
    with pytest.raises(ValueError):
        SemanticGoal("")
    with pytest.raises(ValueError):
        GroundTruthPerception().perceive(Observation(0, 1, "camera", observation.robot))
