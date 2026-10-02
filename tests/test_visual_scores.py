import json

import pytest

from ddm_mcts.robotics.qwen_visual_policy import parse_scores
from ddm_mcts.robotics.reach import cartesian_actions

ACTIONS = cartesian_actions()


@pytest.mark.parametrize("wrapper", ["{}", "```json\n{}\n```", "Result:\n{}\nDone."])
def test_score_parser_normalizes_stable_ids(wrapper):
    raw = json.dumps({"scores": {a.name: i + 1 for i, a in enumerate(ACTIONS)}})
    scores, _, _ = parse_scores(wrapper.format(raw), tuple(reversed(ACTIONS)))
    assert sum(scores.values()) == pytest.approx(1)
    assert scores[ACTIONS[-1]] == pytest.approx(6 / 21)


@pytest.mark.parametrize("value", [-1, float("nan"), float("inf"), "1", True, None])
def test_invalid_score_values(value):
    scores = {a.name: 1 for a in ACTIONS}
    scores[ACTIONS[0].name] = value
    with pytest.raises(ValueError):
        parse_scores(json.dumps({"scores": scores}), ACTIONS)


@pytest.mark.parametrize("kind", ["missing", "unknown", "zero", "duplicate", "malformed"])
def test_invalid_score_schema(kind):
    scores = {a.name: 1 for a in ACTIONS}
    if kind == "missing":
        scores.pop(ACTIONS[0].name)
    if kind == "unknown":
        scores["teleport"] = 1
    if kind == "zero":
        scores = {key: 0 for key in scores}
    raw = json.dumps({"scores": scores})
    if kind == "duplicate":
        raw = raw.replace('"scores": {', '"scores": {"MOVE_X_POS": 9,')
    if kind == "malformed":
        raw = "no scores"
    with pytest.raises(ValueError):
        parse_scores(raw, ACTIONS)
