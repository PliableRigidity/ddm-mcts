"""Dependency-light schema checks: no model, GPU, rendering or downloads."""

import json

import pytest

from ddm_mcts.robotics.vlm_backend import VLMConfig, parse_semantics


def response(**updates):
    data = {"target_label": "cylinder", "target_description": "upright round object", "bbox_2d": [100, 200, 300, 400], "confidence": 0.8}
    return json.dumps({**data, **updates})


def test_json_and_fenced_grounding_normalization():
    raw = response()
    result = parse_semantics(raw)
    assert result.bbox == (0.1, 0.2, 0.3, 0.4)
    assert result.target_label == "cylinder" and result.raw_response == raw
    assert parse_semantics("```json\n" + raw + "\n```").bbox == result.bbox
    assert parse_semantics(response(confidence=None)).confidence is None


@pytest.mark.parametrize(
    "raw",
    [
        "not JSON",
        "[]",
        "{}",
        "```json\n{}",
        response(target_label="unknown"),
        response(target_label=""),
        response(target_description=None),
        response(bbox_2d=[1, 2]),
        response(bbox_2d=[True, 0, 30, 40]),
        response(bbox_2d=[100, 100, 90, 200]),
        response(bbox_2d=[0, 0, 1001, 999]),
        response(bbox_2d=[0, 0, float("nan"), 100]),
        response(confidence=2),
        response(confidence=True),
    ],
)
def test_malformed_semantic_outputs_fail(raw):
    with pytest.raises(ValueError):
        parse_semantics(raw)


def test_config_validation():
    with pytest.raises(ValueError):
        VLMConfig(max_new_tokens=1)
