"""Language/status contracts need neither MuJoCo nor a model."""

import pytest

from ddm_mcts.robotics.ordered_task import ApproachGoal, PhysicalTask, parse_physical_task


@pytest.mark.parametrize(
    "instruction,targets",
    [
        ("Approach the cylinder, then the sphere, then the cube.", ["cylinder", "sphere", "cube"]),
        ("Go to the cube and then the cylinder.", ["cube", "cylinder"]),
        ("Visit the sphere, cube, and cylinder.", ["sphere", "cube", "cylinder"]),
        ("Approach the cylindrical object then approach the spherical object.", ["cylinder", "sphere"]),
        ("Visit cube and cube", ["cube", "cube"]),
    ],
)
def test_supported_grammar_preserves_order_and_occurrence_ids(instruction, targets):
    task = parse_physical_task(instruction)
    assert task.instruction == instruction
    assert [g.target for g in task.subgoals] == targets
    assert [g.id for g in task.subgoals] == [f"goal-{i + 1}" for i in range(len(targets))]


@pytest.mark.parametrize(
    "instruction", ["", "Approach", "Approach cube then", "Visit pyramid", "Do not visit cube", "Grasp cube", "Visit cube or sphere"]
)
def test_unknown_empty_negated_or_unsupported_language_rejected(instruction):
    with pytest.raises(ValueError):
        parse_physical_task(instruction)


def test_representation_criteria_and_cumulative_budget():
    goal = ApproachGoal("a", "sphere", max_actions=7)
    assert goal.task(23).max_steps == 30
    with pytest.raises(ValueError):
        ApproachGoal("a", "sphere", standoff=0)
    with pytest.raises(ValueError):
        PhysicalTask("Visit sphere", ())
    with pytest.raises(ValueError):
        PhysicalTask("Visit sphere", (goal, goal))
