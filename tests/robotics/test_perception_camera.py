# ruff: noqa: E402
import pytest

mujoco = pytest.importorskip("mujoco")
np = pytest.importorskip("numpy")

from ddm_mcts.robotics.camera import CameraCalibration, MujocoCameraObservationProvider
from ddm_mcts.robotics.color_perception import ColorPlanePerception
from ddm_mcts.robotics.mujoco_backend import MujocoBackend
from ddm_mcts.robotics.observation import Observation, SemanticGoal
from ddm_mcts.robotics.reach import ReachState

XML = """<mujoco><asset><material name="redmat" rgba="1 0 0 1" emission="1" specular="0"/><material name="bluemat" rgba="0 0 1 1" emission="1" specular="0"/></asset><worldbody>
<camera name="test" pos="0 0 2" fovy="45"/>
<geom name="red" type="cylinder" pos=".2 .15 0" size=".06 .001" material="redmat" contype="0" conaffinity="0"/>
<geom name="blue" type="cylinder" pos="-.2 -.15 0" size=".06 .001" material="bluemat" contype="0" conaffinity="0"/>
<body><joint type="slide"/><geom size=".01" pos="0 0 -1"/></body>
</worldbody></mujoco>"""


class CameraWorld:
    def __init__(self):
        self.backend = MujocoBackend(xml=XML, physics_steps=10)

    def get_state(self):
        data = self.backend.data
        return ReachState((0, 0, 0), (99, 99, 99), 0, data.time, tuple(data.qpos), tuple(data.qvel))


@pytest.mark.parametrize("width,height", [(160, 120), (320, 240)])
def test_real_camera_rendering_dimensions_restore_and_perception(width, height):
    world = CameraWorld()
    snapshot = world.backend.snapshot()
    with MujocoCameraObservationProvider(world, "test", width, height) as observer:
        try:
            observation = observer.observe()
        except (mujoco.FatalError, RuntimeError) as exc:
            pytest.skip(f"offscreen OpenGL unavailable: {exc}; configure MUJOCO_GL=egl or osmesa")
        assert observation.rgb.shape == (height, width, 3)
        assert observation.rgb.dtype == np.uint8
        assert not observation.rgb.flags.writeable
        assert observation.entities == ()
        assert observation.robot.goal == (0, 0, 0)
        assert world.backend.snapshot() == snapshot
        world.backend.advance()
        observer.observe()
        world.backend.restore(snapshot)
        np.testing.assert_array_equal(observer.observe().rgb, observation.rgb)
        assert world.backend.snapshot() == snapshot
        state = ColorPlanePerception(0.001).perceive(observation, SemanticGoal("red"))
        assert np.linalg.norm(np.asarray(state.goal) - (0.2, 0.15, 0.001)) < 0.007
        blue = ColorPlanePerception(0.001).perceive(observation, SemanticGoal("blue"))
        assert np.linalg.norm(np.asarray(blue.goal) - (-0.2, -0.15, 0.001)) < 0.007
        with world.backend.observe_steps(lambda: None), pytest.raises(RuntimeError, match="outside"):
            observer.observe()
    assert observer.renderer is None
    with pytest.raises(ValueError, match="unknown camera"):
        MujocoCameraObservationProvider(world, "missing")
    with pytest.raises(ValueError):
        MujocoCameraObservationProvider(world, "test", 1, 1)


def test_camera_geometry_roundtrip_and_coordinate_conventions():
    calibration = CameraCalibration(101, 101, 100, (0, 0, 2), (1, 0, 0, 0, 1, 0, 0, 0, 1), "synthetic")
    assert calibration.project((0, 0, 0)) == (50, 50)
    assert calibration.project((0.2, 0.2, 0)) == (60, 40)
    assert calibration.intersect_plane((60, 40), 0) == pytest.approx((0.2, 0.2, 0))
    with pytest.raises(ValueError, match="behind"):
        calibration.project((0, 0, 3))
    with pytest.raises(ValueError, match="behind"):
        calibration.intersect_plane((50, 50), 3)
    rotated = CameraCalibration(101, 101, 100, (0, 0, 2), (0, -1, 0, 1, 0, 0, 0, 0, 1), "rotated")
    point = (0.3, -0.1, 0)
    assert rotated.intersect_plane(rotated.project(point), 0) == pytest.approx(point)


def synthetic_observation(rgb):
    camera = CameraCalibration(rgb.shape[1], rgb.shape[0], 100, (0, 0, 2), (1, 0, 0, 0, 1, 0, 0, 0, 1), "synthetic")
    robot = ReachState((0, 0, 0), (123, 456, 789), 0, 0, (), ())
    return Observation(0, 1, "camera", robot, rgb=rgb, camera=camera)


def test_color_masks_noise_missing_tracking_and_goal_resolution():
    image = np.zeros((101, 101, 3), dtype=np.uint8)
    image[36:45, 56:65] = (240, 5, 5)
    image[56:65, 36:45] = (5, 5, 240)
    image[1, 1] = (255, 0, 0)  # Ignore isolated noise.
    perception = ColorPlanePerception(0, max_missed_frames=1)
    state = perception.perceive(synthetic_observation(image), SemanticGoal("red"))
    assert state.goal == pytest.approx((0.2, 0.2, 0))
    assert state.target().pixels == 81
    assert perception.perceive(synthetic_observation(image), SemanticGoal("blue")).goal == pytest.approx((-0.2, -0.2, 0))
    blank = synthetic_observation(np.zeros_like(image))
    stale = perception.perceive(blank, SemanticGoal("red")).target()
    assert not stale.observed and stale.missed_frames == 1
    with pytest.raises(ValueError, match="expected one"):
        perception.perceive(blank, SemanticGoal("red"))
    perception.reset()
    with pytest.raises(ValueError):
        perception.perceive(blank, SemanticGoal("red"))
    with pytest.raises(ValueError):
        perception.perceive(Observation(0, 1, "empty", state.robot), SemanticGoal("red"))


def test_ambiguous_color_rejected():
    image = np.zeros((101, 101, 3), dtype=np.uint8)
    image[10:20, 10:20] = (255, 0, 0)
    image[70:80, 70:80] = (255, 0, 0)
    with pytest.raises(ValueError, match="ambiguous"):
        ColorPlanePerception(0).perceive(synthetic_observation(image), SemanticGoal("red"))


def test_older_binding_state_copy_fallback(monkeypatch):
    from ddm_mcts.robotics.mujoco_backend import copy_simulator_data

    backend = MujocoBackend(xml=XML)
    backend.data.qpos[:] = 0.2
    backend.data.qvel[:] = 0.1
    backend.data.qfrc_applied[:] = 0.3
    saved = backend.snapshot()
    destination = mujoco.MjData(backend.model)
    monkeypatch.delattr(mujoco, "mj_copyData", raising=False)
    copy_simulator_data(backend.model, backend.data, destination)
    values = np.empty(len(saved.values))
    mujoco.mj_getState(backend.model, destination, values, backend.signature)
    np.testing.assert_array_equal(values, saved.values)
    assert backend.snapshot() == saved
