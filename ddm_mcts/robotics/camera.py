"""MuJoCo RGB observation and calibrated perspective geometry (optional extra)."""

from __future__ import annotations

import copy
from dataclasses import dataclass, replace
from math import radians, tan

import mujoco
import numpy as np

from .mujoco_backend import copy_simulator_data
from .observation import Observation


@dataclass(frozen=True)
class CameraCalibration:
    width: int
    height: int
    focal: float
    position: tuple[float, float, float]
    rotation: tuple[float, ...]  # camera-to-world row-major rotation
    name: str

    @property
    def center(self):
        return ((self.width - 1) / 2, (self.height - 1) / 2)

    def project(self, point):
        local = np.asarray(self.rotation).reshape(3, 3).T @ (np.asarray(point) - self.position)
        depth = -local[2]
        if depth <= 0:
            raise ValueError("point is behind camera")
        cx, cy = self.center
        return (cx + self.focal * local[0] / depth, cy - self.focal * local[1] / depth)

    def intersect_plane(self, pixel, plane_z: float):
        """Pixels (u right, v down); MuJoCo camera +X right, +Y up, -Z forward."""
        cx, cy = self.center
        ray = np.asarray(self.rotation).reshape(3, 3) @ np.array([(pixel[0] - cx) / self.focal, -(pixel[1] - cy) / self.focal, -1])
        if abs(ray[2]) < 1e-10:
            raise ValueError("camera ray parallel to target plane")
        distance = (plane_z - self.position[2]) / ray[2]
        if distance <= 0:
            raise ValueError("target plane lies behind camera")
        return tuple(float(x) for x in np.asarray(self.position) + distance * ray)


class MujocoCameraObservationProvider:
    """RGB uint8 HxWx3, top-left origin. Render from copies, never live MjData.

    Robot telemetry is explicitly simulator proprioception; visual target positions
    are NOT provided in the observation. Rendering is sequential on its owner thread.
    """

    def __init__(self, world, camera="inspection", width=320, height=240):
        if width < 16 or height < 16:
            raise ValueError("image dimensions must be at least 16")
        self.world, self.camera = world, camera
        self.width, self.height = width, height
        self.model = copy.copy(world.backend.model)
        self.camera_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
        if self.camera_id < 0:
            raise ValueError(f"unknown camera: {camera}")
        perspective = (
            int(self.model.cam_projection[self.camera_id]) == 0
            if hasattr(self.model, "cam_projection")
            else not self.model.cam_orthographic[self.camera_id]
        )
        if not perspective or np.any(self.model.cam_sensorsize[self.camera_id]):
            raise ValueError("provider currently supports perspective fovy cameras only")
        self.model.vis.global_.offwidth = max(width, self.model.vis.global_.offwidth)
        self.model.vis.global_.offheight = max(height, self.model.vis.global_.offheight)
        self.data = mujoco.MjData(self.model)
        self.renderer = None
        self.sequence = 0

    def observe(self) -> Observation:
        if self.world.backend._step_callback is not None:
            raise RuntimeError("camera observations must be acquired outside physical execution")
        copy_simulator_data(self.model, self.world.backend.data, self.data)
        mujoco.mj_forward(self.model, self.data)
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model, height=self.height, width=self.width)
        self.renderer.update_scene(self.data, camera=self.camera_id)
        rgb = self.renderer.render().copy()
        rgb.setflags(write=False)
        focal = self.height / (2 * tan(radians(float(self.model.cam_fovy[self.camera_id])) / 2))
        calibration = CameraCalibration(
            self.width,
            self.height,
            focal,
            tuple(float(x) for x in self.data.cam_xpos[self.camera_id]),
            tuple(float(x) for x in self.data.cam_xmat[self.camera_id]),
            self.camera,
        )
        robot = replace(self.world.get_state(), goal=(0.0, 0.0, 0.0))
        self.sequence += 1
        return Observation(
            robot.time,
            self.sequence,
            "camera",
            robot,
            rgb=rgb,
            camera=calibration,
            metadata=(("format", "RGB uint8 HWC; top-left origin"), ("robot_state", "simulator proprioception")),
        )

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
