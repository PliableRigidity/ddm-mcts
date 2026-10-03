"""Read-only RGB checkpoints and explicit known-scene semantic/metric binding."""

import copy
from dataclasses import asdict, replace
from math import radians, tan

import mujoco
import numpy as np

from .camera import CameraCalibration
from .mujoco_backend import copy_simulator_data
from .observation import Entity, Observation, SemanticGoal
from .perception import GroundTruthPerception
from .vlm_perception import VLMSemanticPerception


class ManipulationCamera:
    """A calibrated free camera adds no bodies, joints or physics to the frozen scene."""

    def __init__(self, world, width=640, height=480):
        self.world = world
        self.model = copy.copy(world.backend.model)
        self.model.vis.global_.offwidth = max(width, self.model.vis.global_.offwidth)
        self.model.vis.global_.offheight = max(height, self.model.vis.global_.offheight)
        self.width, self.height = width, height
        self.data = mujoco.MjData(self.model)
        self.renderer = None
        self.sequence = 0
        self.camera = mujoco.MjvCamera()
        self.camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.camera.lookat[:] = (0.50, 0, 0.37)
        self.camera.distance = 0.80
        self.camera.azimuth = 180
        self.camera.elevation = -45

    def observe(self):
        copy_simulator_data(self.model, self.world.backend.data, self.data)
        mujoco.mj_forward(self.model, self.data)
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model, height=self.height, width=self.width)
        self.renderer.update_scene(self.data, camera=self.camera)
        rgb = self.renderer.render().copy()
        rgb.setflags(write=False)
        cameras = self.renderer.scene.camera
        position = np.mean([c.pos for c in cameras], axis=0)
        forward = np.mean([c.forward for c in cameras], axis=0)
        up = np.mean([c.up for c in cameras], axis=0)
        right = np.cross(forward, up)
        rotation = np.column_stack((right, up, -forward))
        focal = self.height / (2 * tan(radians(float(self.model.vis.global_.fovy)) / 2))
        calibration = CameraCalibration(self.width, self.height, focal, tuple(position), tuple(rotation.ravel()), "manipulation")
        self.sequence += 1
        robot = replace(self.world.get_state(), goal=(0, 0, 0))
        return Observation(robot.time, self.sequence, "camera", robot, rgb=rgb, camera=calibration)

    def alternate_view(self):
        # A bounded second camera viewpoint changes only rendering, never physics.
        self.camera.azimuth = 180
        self.camera.elevation = -25 if self.camera.elevation == -45 else -45

    def close(self):
        if self.renderer:
            self.renderer.close()
            self.renderer = None


class ManipulationPerception:
    """VLM grounding gates access to known-scene metric objects; never prompts XYZ.

    Physics object poses are an explicit calibrated simulator geometry source,
    not claimed to be a vision-only metric estimator. Held checkpoints use robot
    telemetry/contacts; semantic grounding occurs before and after operations.
    """

    def __init__(self, world, backend=None):
        self.world, self.backend = world, backend
        self.semantic = None
        self.generation = 0
        self.requests = 0
        self.invalidations = []

    def invalidate(self, reason):
        self.generation += 1
        self.invalidations.append(reason)
        if self.semantic:
            self.semantic.reset()

    def resolve(self, observation, labels):
        entities = tuple(Entity(label, self.world.object_state(label).pose.position) for label in ("cube", "cylinder"))
        bindings = []
        for label in dict.fromkeys(labels):
            self.requests += 1
            obj = self.world.object_state(label)
            if self.backend:
                # Known support/object-height calibration, not a model-generated metric.
                self.semantic = VLMSemanticPerception(self.backend, obj.pose.position[2], (0.15, 0.8, 0.7))
                result = self.semantic.perceive(observation, SemanticGoal(label))
                detected = result.target()
                accepted = {"cube": {"cube", "box", "cuboid"}, "cylinder": {"cylinder", "cylindrical object"}}
                if detected.label.lower().strip() not in accepted[label]:
                    raise ValueError("VLM semantic label disagrees with requested object")
                candidates = sorted((float(np.linalg.norm(np.asarray(e.position[:2]) - detected.position[:2])), e.label) for e in entities)
                if candidates[0][1] != label or candidates[0][0] > 0.035:
                    raise ValueError("semantic grounding does not bind unambiguously to requested known-scene object")
                bindings.append(
                    {
                        "target": label,
                        "grounding": asdict(detected),
                        "metric_binding_error_m": candidates[0][0],
                        "semantic": self.semantic.diagnostics(),
                    }
                )
            else:
                baseline = replace(observation, source="ground-truth", entities=entities)
                GroundTruthPerception().perceive(baseline, SemanticGoal(label)).target()
                bindings.append({"target": label, "source": "explicit known-scene ground truth"})
        state = GroundTruthPerception().perceive(replace(observation, source="ground-truth", entities=entities), SemanticGoal(labels[0]))
        return {"world": asdict(state), "bindings": bindings, "generation": self.generation}
