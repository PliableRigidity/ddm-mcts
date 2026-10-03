# ruff: noqa: E402
"""Geometry, verification, bounded failure handling and actual contact execution."""

import json
import os
from dataclasses import replace
from pathlib import Path

import pytest

mujoco = pytest.importorskip("mujoco")
np = pytest.importorskip("numpy")

from ddm_mcts.robotics.cli import build_parser, main
from ddm_mcts.robotics.contacts import ContactInfo
from ddm_mcts.robotics.manipulation import ManipulationAgent, PickupResult
from ddm_mcts.robotics.manipulation_scene import ManipulationObject, panda_pickup_scene
from ddm_mcts.robotics.pick_place import ManipulationPlan, PickPlace, PickPlaceAgent, PickPlaceConfig, rearrangement_plan
from ddm_mcts.robotics.placement import (
    NextTo,
    PlacementTarget,
    PlacementWorkspace,
    compose,
    inverse,
    measured_grasp,
    next_to_target,
    placement_poses,
    placement_tcp,
    verify_next_to,
)
from ddm_mcts.robotics.placement_verification import PlacementContactRules, verify_placement, verify_retention
from ddm_mcts.robotics.pose_control import PoseTarget


def object_at(label="cube", xyz=(0.50, -0.08, 0.37), quaternion=(1, 0, 0, 0)):
    return ManipulationObject(
        label,
        "pickup_" + label,
        "box" if label == "cube" else "cylinder",
        (0.04, 0.04, 0.04 if label == "cube" else 0.05),
        PoseTarget(xyz, quaternion),
    )


def contact(first, second, distance=-0.0001, force=1):
    return ContactInfo(first + "_geom", second + "_geom", first, second, distance, force)


@pytest.fixture
def world():
    path = os.environ.get("PANDA_MODEL")
    if not path or not Path(path).is_file():
        pytest.skip("external PANDA_MODEL required")
    return panda_pickup_scene(path)


@pytest.mark.parametrize("quaternion", [(1, 0, 0, 0), (0.707, 0, 0, 0.707), (0, 1, 0, 0)])
def test_rigid_transform_inverse_composition(quaternion):
    pose = PoseTarget((0.3, -0.2, 0.7), quaternion)
    identity = compose(pose, inverse(pose))
    np.testing.assert_allclose(identity.position, 0, atol=1e-12)
    np.testing.assert_allclose(identity.rotation, np.eye(3), atol=1e-12)
    reverse = compose(inverse(pose), pose)
    np.testing.assert_allclose(reverse.position, 0, atol=1e-12)


@pytest.mark.parametrize("offset", [(0, 0, 0), (0.003, -0.002, 0.006)])
def test_measured_offset_grasp_drives_tcp_placement(offset):
    tcp = PoseTarget((0.5, 0.08, 0.48), (0, 1, 0, 0))
    relative = PoseTarget(offset, (1, 0, 0, 0))
    actual = compose(tcp, relative)
    measured = measured_grasp(tcp, actual)
    np.testing.assert_allclose(measured.position, offset)
    desired = PoseTarget((0.6, -0.1, 0.375), (1, 0, 0, 0))
    required = placement_tcp(desired, measured)
    reconstructed = compose(required, measured)
    np.testing.assert_allclose(reconstructed.position, desired.position)
    np.testing.assert_allclose(reconstructed.rotation, desired.rotation)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"position": (0, 0, float("nan"))},
        {"position": (0, 0)},
        {"position_tolerance": 0},
        {"orientation_tolerance": -1},
        {"support_surface": "air"},
    ],
)
def test_invalid_placement_target(kwargs):
    with pytest.raises(ValueError):
        PlacementTarget(**{"position": (0.6, -0.1, 0.37), **kwargs})


@pytest.mark.parametrize("position", [(0.8, 0, 0.37), (0.35, 0, 0.37), (0.5, 0.25, 0.37), (0.6, 0, 0.5)])
def test_workspace_invalid(position):
    with pytest.raises(ValueError, match="invalid_destination"):
        PlacementWorkspace().validate(PlacementTarget(position), object_at())


def test_absolute_workspace_and_obstacle_rejection():
    workspace = PlacementWorkspace()
    target = PlacementTarget((0.60, -0.10, 0.37))
    assert workspace.validate(target, object_at()) == target
    with pytest.raises(ValueError, match="overlaps"):
        workspace.validate(target, object_at(), [object_at("cylinder", target.position)])


@pytest.mark.parametrize("gap", [0.025, 0.04, 0.07])
def test_next_to_geometry_depends_on_extents_and_gap(gap):
    obj, reference = object_at("cylinder", (0.6, -0.08, 0.375)), object_at()
    target, direction = next_to_target(obj, reference, NextTo("cube", gap), PlacementWorkspace(), [reference])
    assert direction == (1, 0, 0)
    assert target.position[0] == pytest.approx(reference.pose.position[0] + 0.02 + 0.02 + gap)
    enlarged = replace(reference, dimensions=(0.06, 0.04, 0.04))
    bigger, _ = next_to_target(obj, enlarged, NextTo("cube", gap), PlacementWorkspace(), [enlarged])
    assert bigger.position[0] == pytest.approx(target.position[0] + 0.01)


def test_next_to_uses_workspace_and_no_self_reference():
    obj = object_at("cylinder", (0.5, 0.08, 0.375))
    reference = object_at(xyz=(0.66, 0, 0.37))
    _, direction = next_to_target(obj, reference, NextTo("cube"), PlacementWorkspace(), [reference])
    assert direction != (1, 0, 0)
    with pytest.raises(ValueError, match="itself"):
        next_to_target(obj, obj, NextTo("cylinder"), PlacementWorkspace())


@pytest.mark.parametrize("args", [("sphere", 0.04), ("cube", -0.1), ("cube", float("nan"))])
def test_invalid_next_to(args):
    with pytest.raises(ValueError):
        NextTo(*args)


def test_transport_preplace_and_retreat_are_distinct():
    obj = object_at(xyz=(0.5, -0.08, 0.47))
    tcp = PoseTarget((0.5, -0.08, 0.47), (0, 1, 0, 0))
    rel = measured_grasp(tcp, obj.pose)
    poses = placement_poses(
        tcp, obj, PlacementTarget((0.6, -0.1, 0.37)), rel, PlacementWorkspace(), [object_at("cylinder", (0.5, 0.08, 0.375))]
    )
    assert len(poses.transport) == 2
    assert poses.transport_height >= 0.375 + 0.025 + 0.02 + 0.06
    assert poses.preplace.position[2] - poses.place.position[2] == pytest.approx(0.06)
    assert poses.retreat.position[2] - poses.place.position[2] == pytest.approx(0.10)
    assert compose(poses.place, rel).position == pytest.approx((0.6, -0.1, 0.37))


@pytest.mark.parametrize("distance,success", [(0.08, True), (0.06, False), (0.15, False), (0.03, False)])
def test_relation_evaluated_from_final_actual_positions(distance, success):
    ref = object_at()
    actual = object_at("cylinder", (0.5 + distance, -0.08, 0.375))
    check = verify_next_to(actual, ref)
    assert check.success == success
    assert check.surface_gap == pytest.approx(distance - 0.04)
    assert check.overlap == pytest.approx(max(0, 0.04 - distance))


def test_relation_rejects_tipped_cylinder():
    actual = object_at("cylinder", (0.58, -0.08, 0.375), (0.707, 0.707, 0, 0))
    assert not verify_next_to(actual, object_at()).success


@pytest.mark.parametrize(
    "phase,other,allowed",
    [
        ("transport", "left_finger", True),
        ("transport", "support_table", False),
        ("descend", "support_table", True),
        ("release", "left_finger", True),
        ("retreat", "left_finger", False),
        ("settle", "support_table", True),
        ("descend", "pickup_cylinder", False),
        ("release", "hand", False),
    ],
)
def test_phase_contact_semantics(phase, other, allowed):
    c = contact("pickup_cube", other)
    assert bool(PlacementContactRules("pickup_cube", phase).violations([c])) == (not allowed)


def test_hand_table_forbidden_even_during_descent():
    assert PlacementContactRules("pickup_cube", "descend").violations([contact("hand", "support_table")])
    assert PlacementContactRules("pickup_cube", "descend").violations([contact("pickup_cube", "support_table", -0.004)])


@pytest.mark.parametrize("case", ["valid", "missing_contact", "drift", "rotation", "closed_empty"])
def test_retention_multiple_physical_signals(case):
    obj = object_at(xyz=(0.5, -0.08, 0.47))
    tcp = PoseTarget(obj.pose.position, (0, 1, 0, 0))
    rel = measured_grasp(tcp, obj.pose)
    cs = [contact(obj.body_name, finger) for finger in ("left_finger", "right_finger")]
    if case == "missing_contact":
        cs.pop()
    if case == "drift":
        obj = replace(obj, pose=PoseTarget((0.52, -0.08, 0.47), (1, 0, 0, 0)))
    if case == "rotation":
        obj = replace(obj, pose=PoseTarget(obj.pose.position, (0.707, 0.707, 0, 0)))
    check = verify_retention(tcp, obj, rel, 0.001 if case == "closed_empty" else 0.04, cs)
    assert check.success == (case == "valid")


@pytest.mark.parametrize(
    "case", ["valid", "position", "orientation", "unsupported", "still_held", "linear", "angular", "pose_change", "forbidden", "tipped"]
)
def test_placement_verifier_failure_signals(case):
    obj = object_at()
    target = PlacementTarget(obj.pose.position)
    cs = [contact(obj.body_name, "support_table")]
    if case == "position":
        obj = replace(obj, pose=PoseTarget((0.52, -0.08, 0.37), (1, 0, 0, 0)))
    if case in ("orientation", "tipped"):
        obj = object_at("cylinder" if case == "tipped" else "cube", obj.pose.position, (0.707, 0.707, 0, 0))
        cs = [contact(obj.body_name, "support_table")]
    if case == "unsupported":
        cs = []
    if case == "still_held":
        cs.append(contact(obj.body_name, "left_finger"))
    check = verify_placement(
        obj,
        target,
        cs,
        0.03 if case == "linear" else 0,
        0.2 if case == "angular" else 0,
        pose_change=0.004 if case == "pose_change" else 0,
        forbidden=case == "forbidden",
    )
    assert check.success == (case == "valid")


@pytest.mark.parametrize("kwargs", [{"max_retries": 2}, {"max_retries": -1}, {"settle_seconds": 0.1}, {"preplace_clearance": 0.01}])
def test_bounded_configuration(kwargs):
    with pytest.raises(ValueError):
        PickPlaceConfig(**kwargs)


def test_plan_is_focused_and_ordered():
    plan = rearrangement_plan()
    assert [o.target for o in plan.operations] == ["cylinder", "cube"]
    with pytest.raises(ValueError):
        ManipulationPlan(())
    with pytest.raises(ValueError):
        PickPlace("sphere", NextTo("cube"))


def test_invalid_destination_stops_without_physics_or_reset(world, tmp_path):
    initial = world.backend.data.qpos.copy()
    plan = ManipulationPlan((PickPlace("cube", PlacementTarget((0.9, 0, 0.37))), PickPlace("cylinder", NextTo("cube"))))
    result = PickPlaceAgent(world).run_manipulation_plan(plan, tmp_path)
    assert not result.success
    assert len(result.trace["operations"]) == 1
    assert "invalid_destination" in result.trace["operations"][0]["failure"]["reason"]
    assert result.trace["operations"][0]["physics_substeps"] == 0
    np.testing.assert_array_equal(initial, world.backend.data.qpos)
    assert json.loads((result.folder / "summary.json").read_text())["success"] is False


@pytest.mark.parametrize("succeed_on_retry", [True, False])
def test_retry_is_bounded_and_executes_retreat(world, tmp_path, monkeypatch, succeed_on_retry):
    pickup = ManipulationAgent(world)
    calls = []

    def fake(task, folder, visualization=None):
        calls.append(world.backend.data.qpos.copy())
        success = succeed_on_retry and len(calls) == 2
        return PickupResult(
            success, Path(folder), {"failure_stage": "verify_grasp", "reason": "bilateral contact absent", "contact_violations": []}
        )

    monkeypatch.setattr(pickup, "run", fake)
    agent = PickPlaceAgent(world, pickup=pickup)
    row = {"recovery": [], "forbidden_contacts": []}
    operation = PickPlace("cube", PlacementTarget((0.6, -0.1, 0.37)))
    if succeed_on_retry:
        assert agent._pickup(operation, row, tmp_path, None, lambda *a, **k: None).success
    else:
        with pytest.raises(RuntimeError):
            agent._pickup(operation, row, tmp_path, None, lambda *a, **k: None)
    assert len(calls) == 2 and len(row["recovery"]) == 1
    assert not np.array_equal(calls[0], calls[1])  # physical retreat, not reset


def test_unrecoverable_failure_no_retry(world, tmp_path, monkeypatch):
    pickup = ManipulationAgent(world)
    calls = []

    def fail(*a, **k):
        calls.append(1)
        return PickupResult(False, tmp_path, {"failure_stage": "hold", "reason": "dropped", "contact_violations": []})

    monkeypatch.setattr(pickup, "run", fail)
    result = PickPlaceAgent(world, pickup=pickup).run_manipulation_plan(rearrangement_plan(), tmp_path)
    assert not result.success and len(calls) == 1 and len(result.trace["operations"]) == 1


def test_cli_new_modes_and_old_pickup():
    parser = build_parser()
    assert parser.parse_args(["--task", "pickup", "--goal", "cube"]).task == "pickup"
    assert parser.parse_args(["--task", "pick-place", "--goal", "cube", "--place-position", ".6", "-.1", ".37"]).place_position == [
        0.6,
        -0.1,
        0.37,
    ]
    assert parser.parse_args(["--task", "rearrange-demo"]).task == "rearrange-demo"
    with pytest.raises(SystemExit):
        main(["--task", "pick-place", "--model", "invalid", "--goal", "cube"])


@pytest.mark.parametrize("kind", ["cube", "cylinder", "sequential"])
def test_real_contact_placement_and_state_continuity(world, tmp_path, kind):
    agent = PickPlaceAgent(world)
    initial = world.backend.data.qpos.copy()
    if kind == "sequential":
        result = agent.run_manipulation_plan(rearrangement_plan(), tmp_path)
    else:
        result = agent.pick_and_place(kind, NextTo("cube") if kind == "cylinder" else PlacementTarget((0.60, -0.10, 0.37)), tmp_path)
    assert result.success, result.trace["operations"][-1].get("failure")
    assert result.trace["reset_count"] == 0 and result.trace["state_continuity"]
    assert all(c["success"] for c in result.trace["final_scene_verification"].values())
    events = [json.loads(line) for line in (result.folder / "events.jsonl").read_text().splitlines()]
    motions = [e for e in events if e["event"] == "trajectory"]
    assert motions and all("contacts" in e and "gripper" in e and "tcp_object" in e for e in motions)
    assert not np.array_equal(initial, world.backend.data.qpos)
    for row in result.trace["operations"]:
        assert row["placement_verification"]["supported"] and row["placement_verification"]["released"]
        assert row["transport"]["grasp_retained"]
        assert row["physics_substeps"] > 1000 and not row["forbidden_contacts"]
        assert row["release_gripper"]["actual_width"] > 0.07
    if kind == "sequential":
        first, second = result.trace["operations"]
        assert first["end_state"] == second["start_state"]
        assert second["pickup_attempts"][0]["trace"]["initial_qpos"] == first["end_state"]["qpos"]
        assert second["start_state"]["objects"]["cylinder"]["pose"] != first["start_state"]["objects"]["cylinder"]["pose"]


def test_cylinder_yaw_symmetry_preserves_tilt_monitoring():
    obj = object_at("cylinder", (0.5, 0.08, 0.47))
    tcp = PoseTarget(obj.pose.position, (0, 1, 0, 0))
    rel = measured_grasp(tcp, obj.pose)
    obj = replace(obj, pose=PoseTarget(obj.pose.position, (0.707, 0, 0, 0.707)))
    contacts = [contact(obj.body_name, finger) for finger in ("left_finger", "right_finger")]
    check = verify_retention(tcp, obj, rel, 0.04, contacts)
    assert check.success and check.rotation_drift < 1e-6
    assert check.full_rotation_drift == pytest.approx(np.pi / 2)


@pytest.mark.parametrize("mode", ["missing", "unloaded"])
def test_transport_contact_loss_stops_plan_with_bounded_grace(world, tmp_path, monkeypatch, mode):
    agent = PickPlaceAgent(world)
    original = agent.contacts.contacts
    first = []

    def contacts():
        cs = original()
        if agent.phase == "transport":
            first.append(float(world.backend.data.time))
            if mode == "missing":
                return tuple(c for c in cs if "pickup_cylinder" not in (c.body1, c.body2))
            return tuple(replace(c, normal_force=0) if "pickup_cylinder" in (c.body1, c.body2) else c for c in cs)
        return cs

    monkeypatch.setattr(agent.contacts, "contacts", contacts)
    result = agent.run_manipulation_plan(rearrangement_plan(), tmp_path)
    assert not result.success and len(result.trace["operations"]) == 1
    row = result.trace["operations"][0]
    assert row["failure"]["reason"] == "grasp_lost_during_transport"
    assert row["failure"]["phase"] == "transport" and not row["recovery"]
    assert row["end_state"]["time"] - min(first) < 0.024


def test_continuous_viewer_one_session_no_reset(world, tmp_path):
    from contextlib import contextmanager

    from ddm_mcts.robotics.manipulation_visual import PickupInspection

    class Viewer:
        def __init__(self):
            self.cam = type("Camera", (), {"lookat": np.zeros(3)})()
            self.user_scn = mujoco.MjvScene(world.backend.model, maxgeom=10)
            self.closed = False
            self.sync_count = 0

        @contextmanager
        def lock(self):
            yield

        def is_running(self):
            return not self.closed

        def sync(self):
            self.sync_count += 1

        def close(self):
            self.closed = True

    viewer, launches = Viewer(), []

    def launch(model, data):
        assert model is not world.backend.model and data is not world.backend.data
        launches.append(1)
        return viewer

    with PickupInspection(world, speed=1000, launch=launch) as visual:
        result = PickPlaceAgent(world).run_manipulation_plan(rearrangement_plan(), tmp_path, visualization=visual)
        assert result.success, result.trace["operations"][-1].get("failure")
        assert not viewer.closed and viewer.sync_count > 4
    assert launches == [1] and viewer.closed


def test_relation_rejects_hovering_reference():
    assert not verify_next_to(object_at("cylinder", (0.58, -0.08, 0.375)), object_at(xyz=(0.5, -0.08, 0.47))).success


def test_final_scene_checks_both_objects_supported(world):
    checks = PickPlaceAgent(world)._scene_checks()
    assert all(c["success"] for c in checks.values())
