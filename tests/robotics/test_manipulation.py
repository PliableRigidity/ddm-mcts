# ruff: noqa: E402
"""Real-physics tests need only MuJoCo and an external Panda model, no GUI/model."""

import json
import os
from dataclasses import replace
from pathlib import Path

import pytest

mujoco = pytest.importorskip("mujoco")
np = pytest.importorskip("numpy")

from ddm_mcts.robotics.cli import build_parser, main
from ddm_mcts.robotics.contacts import ContactInfo, ContactInspector, ContactRules
from ddm_mcts.robotics.grasp import ParallelJawGraspGenerator, verify_grasp, verify_lift
from ddm_mcts.robotics.gripper import PandaGripper
from ddm_mcts.robotics.manipulation import ManipulationAgent, PickupStage, PickupTask
from ddm_mcts.robotics.manipulation_scene import panda_pickup_scene
from ddm_mcts.robotics.pose_control import PoseIKConfig, PoseTarget, rotation_error, rotation_matrix


@pytest.fixture
def world():
    path = os.environ.get("PANDA_MODEL")
    if not path or not Path(path).is_file():
        pytest.skip("PANDA_MODEL external Menagerie model required")
    return panda_pickup_scene(path)


@pytest.mark.parametrize("angle", [0, 1e-7, 0.1, np.pi / 2, np.pi])
def test_so3_error(angle):
    q = (np.cos(angle / 2), 0, 0, np.sin(angle / 2))
    np.testing.assert_allclose(rotation_error(rotation_matrix(q), np.eye(3)), (0, 0, angle), atol=1e-8)


def test_quaternion_sign_and_normalization():
    first = PoseTarget((0, 0, 0), (1, 2, 3, 4))
    second = PoseTarget((0, 0, 0), (-1, -2, -3, -4))
    np.testing.assert_allclose(first.rotation, second.rotation)
    assert np.linalg.norm(first.quaternion) == pytest.approx(1)


@pytest.mark.parametrize(
    "position,quaternion",
    [((1, 2), (1, 0, 0, 0)), ((0, 0, np.nan), (1, 0, 0, 0)), ((0, 0, 0), (0, 0, 0, 0)), ((0, 0, 0), (1, np.inf, 0, 0))],
)
def test_invalid_pose(position, quaternion):
    with pytest.raises(ValueError):
        PoseTarget(position, quaternion)


@pytest.mark.parametrize("kwargs", [{"damping": 0}, {"orientation_weight": -1}, {"position_weight": np.nan}, {"iterations": 0}])
def test_invalid_pose_config(kwargs):
    with pytest.raises(ValueError):
        PoseIKConfig(**kwargs)


@pytest.mark.parametrize("angle", [0, np.pi / 4, -np.pi / 4])
def test_pose_ik_rotation_and_position_converge(world, angle, monkeypatch):
    controller = world.controller
    start = controller.pose()
    target_rotation = rotation_matrix((np.cos(angle / 2), 0, 0, np.sin(angle / 2))) @ start.rotation
    q = np.empty(4)
    mujoco.mju_mat2Quat(q, target_rotation.ravel())
    target = PoseTarget((0.54, 0.025, 0.56), tuple(q))
    calls = []
    original = mujoco.mj_jacSite

    def jac(model, data, jp, jr, site):
        assert jr is not None
        calls.append(1)
        return original(model, data, jp, jr, site)

    monkeypatch.setattr(mujoco, "mj_jacSite", jac)
    errors = controller.move_to(target)
    assert calls
    assert errors["position_error_m"] < 0.002
    assert errors["orientation_error_rad"] < 0.025
    model = world.backend.model
    assert np.isfinite(world.backend.data.qpos).all()
    assert np.all(world.backend.data.ctrl >= model.actuator_ctrlrange[:, 0])
    assert np.all(world.backend.data.ctrl <= model.actuator_ctrlrange[:, 1])
    for joint, address in zip(controller.joints, controller.qpos, strict=True):
        assert model.jnt_range[joint, 0] - 0.002 <= world.backend.data.qpos[address] <= model.jnt_range[joint, 1] + 0.002


def test_pose_commands_never_teleport_live_state(world, monkeypatch):
    before = world.backend.data.qpos.copy()
    advance = world.backend.advance

    def checked():
        np.testing.assert_array_equal(world.backend.data.qpos, before)
        advance()

    monkeypatch.setattr(world.backend, "advance", checked)
    world.controller.execute_pose(PoseTarget((0.55, 0.01, 0.54), (0, 1, 0, 0)))


def test_position_only_still_available(world):
    before = world.controller.position()
    world.controller.execute((0, 0, 0.01))
    for _ in range(10):
        world.backend.advance()
    assert world.controller.position()[2] > before[2] + 0.005


def test_gripper_actuation_bounds_state_and_callback(world):
    gripper = PandaGripper(world.backend)
    before = world.backend.data.qpos.copy()
    callbacks = []
    with world.backend.observe_steps(lambda: callbacks.append(gripper.state())):
        closed = gripper.close()
        assert closed.actual_width < 0.003
        opened = gripper.open()
        assert opened.actual_width == pytest.approx(0.08, abs=0.001)
        midway = gripper.set_width(0.04)
    assert midway.actual_width == pytest.approx(0.04, abs=0.001)
    assert len(callbacks) == 36 * world.backend.physics_steps
    assert callbacks[0].actual_width > closed.actual_width  # physical finite movement
    assert not np.array_equal(before, world.backend.data.qpos)
    assert midway.commanded_width == pytest.approx(0.04)
    assert midway.left_position + midway.right_position == midway.actual_width
    assert not midway.moving
    for value in (-1, 0.081, np.nan):
        with pytest.raises(ValueError):
            gripper.set_width(value)


def test_contact_support_and_no_finger_contacts(world):
    inspector = ContactInspector(world.backend)
    assert inspector.touching("pickup_cube", "support_table")
    assert not inspector.touching("left_finger", "pickup_cube")
    assert not ContactRules("pickup_cube").violations(inspector.contacts())
    assert any(c.normal_force > 0 for c in inspector.contacts())


@pytest.mark.parametrize(
    "other,allowed,distance,violation",
    [
        ("left_finger", False, -0.001, True),
        ("left_finger", True, -0.001, False),
        ("right_finger", True, -0.004, True),
        ("hand", True, -0.001, True),
        ("support_table", False, -0.0001, False),
        ("link7", True, -0.001, True),
    ],
)
def test_phase_contact_rules(other, allowed, distance, violation):
    c = ContactInfo("target", "other", "pickup_cube", other, distance, 1)
    assert bool(ContactRules("pickup_cube", allowed).violations((c,))) == violation
    assert not ContactRules("pickup_cube").violations(())


@pytest.mark.parametrize("label", ["cube", "cylinder"])
def test_grasp_geometry(world, label):
    obj = world.object_state(label)
    grasp = ParallelJawGraspGenerator().generate(obj)
    assert grasp.tcp_pose.position == obj.pose.position
    assert grasp.pregrasp_pose.position[2] - grasp.tcp_pose.position[2] == pytest.approx(0.10)
    assert grasp.lift_pose.position[2] - grasp.tcp_pose.position[2] == pytest.approx(0.10)
    np.testing.assert_allclose(grasp.tcp_pose.rotation[:, 2], (0, 0, -1), atol=1e-10)
    assert obj.dimensions[1] < grasp.finger_width <= 0.08
    assert world.backend.model.body_gravcomp[world.backend.model.body(obj.body_name).id] == 0
    assert world.backend.model.jnt_type[world.backend.model.joint(f"pickup_{label}_free").id] == mujoco.mjtJoint.mjJNT_FREE
    assert world.backend.model.body(obj.body_name).mass > 0
    assert world.backend.model.neq == 1  # only existing finger coupling, no object weld


@pytest.mark.parametrize("label", ["cube", "cylinder"])
def test_real_pickup_and_hold_no_resets(world, label, tmp_path, monkeypatch):
    def forbidden(*args):
        raise AssertionError("reset during pickup")

    monkeypatch.setattr(world, "reset", forbidden)
    monkeypatch.setattr(world.backend, "reset", forbidden)
    initial_time = world.backend.data.time
    result = ManipulationAgent(world).run(PickupTask(label), tmp_path)
    assert result.success, result.trace.get("reason")
    trace = result.trace
    assert trace["stages"][-1]["stage"] == "success"
    assert [s["stage"] for s in trace["stages"]] == [str(s) for s in PickupStage if s != PickupStage.FAILED]
    assert trace["pregrasp_errors"]["position_error_m"] < 0.002
    assert trace["grasp_pose_errors"]["orientation_error_rad"] < 0.025
    assert trace["grasp_verification"]["left_contact"] and trace["grasp_verification"]["right_contact"]
    assert trace["lift_verification"]["height_increase"] > 0.075
    assert trace["lift_verification"]["relative_drift"] < 0.015
    assert not trace["lift_verification"]["table_contact"]
    assert trace["hold_checks"] and all(c["success"] for c in trace["hold_checks"])
    assert trace["physics_steps_monitored"] > 1000
    assert not trace["contact_violations"]
    assert world.backend.data.time > initial_time
    assert json.loads((result.folder / "summary.json").read_text())["success"]
    events = [json.loads(line) for line in (result.folder / "events.jsonl").read_text().splitlines()]
    assert any(e["event"] == "trajectory" for e in events)
    assert any(e["event"] == "contacts" for e in events)


def test_invalid_grasp_fails_and_does_not_lift(world, tmp_path):
    class BadGenerator:
        def generate(self, obj, **kwargs):
            g = ParallelJawGraspGenerator().generate(obj, **kwargs)
            shifted = PoseTarget((g.tcp_pose.position[0] + 0.08, *g.tcp_pose.position[1:]), g.tcp_pose.quaternion)
            return replace(g, tcp_pose=shifted)

    result = ManipulationAgent(world, generator=BadGenerator()).run(PickupTask("cube"), tmp_path)
    assert not result.success
    assert result.trace["failure_stage"] == "verify_grasp"
    assert not result.trace["grasp_verification"]["success"]
    assert "lift" not in [s["stage"] for s in result.trace["stages"]]


def test_dropped_or_unsupported_object_fails_verifiers(world):
    obj = world.object_state("cube")
    g = PandaGripper(world.backend)
    c = ContactInspector(world.backend)
    assert not verify_grasp(world.controller, g, c, obj).success
    assert not verify_lift(obj, obj, (0, 0, 0), world.controller, g, c).success


def test_pregrasp_pose_failure_stops(world, tmp_path, monkeypatch):
    monkeypatch.setattr(world.controller, "move_to", lambda *args, **kwargs: None)
    result = ManipulationAgent(world).run(PickupTask("cube"), tmp_path)
    assert not result.success and result.trace["failure_stage"] == "verify_pregrasp"


def test_cli_pickup_and_validation(world, tmp_path):
    args = build_parser().parse_args(["--task", "pickup", "--goal", "cube"])
    assert args.task == "pickup" and args.lift_distance == 0.1 and not args.viewer
    with pytest.raises(SystemExit):
        main(["--task", "pickup", "--goal", "sphere"])
    assert main(["--model", os.environ["PANDA_MODEL"], "--task", "pickup", "--goal", "cube", "--output", str(tmp_path)]) == 0


def test_snapshot_restore_includes_objects_and_fingers(world):
    snapshot = world.snapshot()
    PandaGripper(world.backend).close()
    assert world.snapshot() != snapshot
    world.restore(snapshot)
    assert world.snapshot() == snapshot


def test_pose_limit_and_unreachable_target_fails_verification(world):
    target = PoseTarget((5, 5, 5), (0, 1, 0, 0))
    errors = world.controller.execute_pose(target)
    assert np.isfinite(tuple(errors.values())).all()
    for joint, address in zip(world.controller.joints, world.controller.qpos, strict=True):
        assert world.backend.model.jnt_range[joint, 0] <= world.controller.scratch.qpos[address] <= world.backend.model.jnt_range[joint, 1]
    with pytest.raises(RuntimeError):
        ManipulationAgent(world)._pose_verified(target)


def test_table_hand_and_wrong_object_contacts_forbidden():
    rules = ContactRules("pickup_cube", True)
    assert rules.violations((ContactInfo("hand", "table", "hand", "support_table", -0.001, 1),))
    assert rules.violations((ContactInfo("finger", "target", "left_finger", "pickup_cylinder", -0.001, 1),))
    assert rules.violations((ContactInfo("hand", "floor", "hand", "world", -0.001, 1),))


def test_no_force_contact_is_not_grasp(world, monkeypatch):
    inspector = ContactInspector(world.backend)
    monkeypatch.setattr(inspector, "contacts", lambda: (ContactInfo("a", "b", "left_finger", "pickup_cube", 0, 0),))
    assert not inspector.touching("left_finger", "pickup_cube")


def test_insufficient_open_or_clearance_stops_before_approach(world, tmp_path, monkeypatch):
    agent = ManipulationAgent(world)
    monkeypatch.setattr(agent, "_clearance", lambda target: 0.001)
    result = agent.run(PickupTask("cube"), tmp_path)
    assert not result.success and result.trace["failure_stage"] == "verify_pregrasp"
    assert "approach_grasp" not in [s["stage"] for s in result.trace["stages"]]


def test_drop_during_hold_stops_task(world, tmp_path, monkeypatch):
    import ddm_mcts.robotics.manipulation as module
    from ddm_mcts.robotics.grasp import LiftVerification

    monkeypatch.setattr(module, "verify_lift", lambda *args, **kwargs: LiftVerification(False, 0, 0.1, True, False, "dropped"))
    result = ManipulationAgent(world).run(PickupTask("cube"), tmp_path)
    assert not result.success and result.trace["failure_stage"] == "hold"
    assert "dropped" in result.trace["reason"]


def test_pickup_viewer_one_session_and_callbacks(world, tmp_path):
    from contextlib import contextmanager

    from ddm_mcts.robotics.manipulation_visual import PickupInspection

    class FakeViewer:
        def __init__(self):
            self.cam = type("Camera", (), {"lookat": np.zeros(3), "distance": 0, "azimuth": 0, "elevation": 0})()
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

    viewer = FakeViewer()
    launches = []

    def launch(model, data):
        assert model is not world.backend.model and data is not world.backend.data
        launches.append(1)
        return viewer

    with PickupInspection(world, speed=1000, launch=launch) as visual:
        result = ManipulationAgent(world).run(PickupTask("cube"), tmp_path, visualization=visual)
        assert result.success and not viewer.closed
        assert viewer.sync_count > 2
    assert launches == [1] and viewer.closed


@pytest.mark.parametrize(
    "kwargs",
    [
        {"target": "sphere"},
        {"target": "cube", "hold_seconds": 0},
        {"target": "cube", "lift_distance": 0.5},
        {"target": "cube", "pregrasp_offset": np.nan},
    ],
)
def test_pickup_task_bounds(kwargs):
    with pytest.raises(ValueError):
        PickupTask(**kwargs)
