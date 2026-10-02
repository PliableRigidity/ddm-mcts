"""Same-color shape scene: semantic identities exist only in diagnostics."""

from dataclasses import dataclass

import mujoco
import numpy as np

from .observation import Entity
from .reach import ReachTask, panda_reach

SEMANTIC_PLANE_Z = 0.6245
OBJECT_RGB = (0.15, 0.8, 0.7)


@dataclass(frozen=True)
class ShapeObject:
    label: str
    shape: str
    position: tuple[float, float, float]


DEFAULT_OBJECTS = (
    ShapeObject("cylinder", "cylinder", (0.6045, 0.08, SEMANTIC_PLANE_Z)),
    ShapeObject("cube", "box", (0.5045, -0.08, SEMANTIC_PLANE_Z)),
    ShapeObject("sphere", "sphere", (0.6245, -0.08, SEMANTIC_PLANE_Z)),
)


def panda_semantic_reach(model_path, *, objects=DEFAULT_OBJECTS, increment=0.02, axes=(0, 1, 2), physics_steps=150, use_tcp=True):
    objects = tuple(objects)
    if not objects or len({obj.label for obj in objects}) != len(objects):
        raise ValueError("unique nonempty semantic scene objects required")

    def configure(spec):
        spec.body("hand").add_site(name="panda_gripper_tcp", pos=(0, 0, 0.1034), size=(0.003, 0, 0), rgba=(0, 0, 0, 0))
        # Oblique camera reveals shape; calibration comes from actual cam_xmat.
        position = np.array([0.85, -0.45, 1.8])
        forward = np.array([0.55, 0, SEMANTIC_PLANE_Z]) - position
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, [0, 0, 1])
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        spec.worldbody.add_camera(name="inspection", pos=position, xyaxes=tuple(np.concatenate([right, up])), fovy=30)
        material = spec.add_material(name="shape_material", rgba=(*OBJECT_RGB, 1), emission=0.25, specular=0)
        shapes = {
            "box": (mujoco.mjtGeom.mjGEOM_BOX, (0.023, 0.023, 0.023)),
            "sphere": (mujoco.mjtGeom.mjGEOM_SPHERE, (0.025, 0, 0)),
            "cylinder": (mujoco.mjtGeom.mjGEOM_CYLINDER, (0.023, 0.028, 0)),
        }
        for index, obj in enumerate(objects):
            kind, size = shapes[obj.shape]
            spec.worldbody.add_geom(
                name=f"semantic_object_{index}",
                type=kind,
                pos=obj.position,
                size=size,
                material=material.name,
                contype=0,
                conaffinity=0,
                mass=0,
            )

    world = panda_reach(
        model_path,
        ReachTask((0.8, 0.2, 0.5)),
        increment=increment,
        axes=axes,
        physics_steps=physics_steps,
        configure_spec=configure,
        tcp_site="panda_gripper_tcp" if use_tcp else None,
    )

    def truth():
        return tuple(
            Entity(
                obj.label, tuple(float(x) for x in world.backend.data.geom_xpos[world.backend.model.geom(f"semantic_object_{index}").id])
            )
            for index, obj in enumerate(objects)
        )

    world.target_entities = truth  # Explicit optional diagnostics/baseline ONLY.

    def geometry_diagnostics(label):
        model, data = world.backend.model, world.backend.data
        target = next((index for index, obj in enumerate(objects) if obj.label == label), None)
        if target is None:
            return {}
        geom = model.geom(f"semantic_object_{target}").id
        bodies = {model.body(name).id for name in ("hand", "left_finger", "right_finger")}
        gripper = [g for g in range(model.ngeom) if model.geom_bodyid[g] in bodies and model.geom_group[g] == 3]
        distances = [float(mujoco.mj_geomDistance(model, data, g, geom, 1.0, None)) for g in gripper]
        return {
            "target_geometry": f"semantic_object_{target}",
            "true_geom_size": model.geom_size[geom].tolist(),
            "minimum_gripper_target_distance_m": min(distances) if distances else None,
        }

    world.geometry_diagnostics = geometry_diagnostics
    return world
