"""Scene configuration only: non-colliding visual targets and calibrated camera."""

from dataclasses import dataclass

import mujoco

from .observation import Entity
from .reach import ReachTask, panda_reach


@dataclass(frozen=True)
class VisualTarget:
    label: str
    position: tuple[float, float, float]
    rgba: tuple[float, float, float, float]
    radius: float = 0.018


DEFAULT_TARGETS = (
    VisualTarget("red", (0.6045, 0.08, 0.6245), (1, 0, 0, 1)),
    VisualTarget("blue", (0.5045, -0.08, 0.6245), (0, 0, 1, 1)),
)
TARGET_PLANE_Z = 0.6245


def panda_visual_reach(model_path, *, targets=DEFAULT_TARGETS, increment=0.02, axes=(0, 1, 2), physics_steps=150):
    """Same robot/controller/planner; change only scene/task configuration.

    target coordinates belong to scene construction and optional diagnostics, never
    to visual perception. Targets are thin static disks with no contact/dynamics.
    """
    targets = tuple(targets)
    if not targets or len({t.label for t in targets}) != len(targets):
        raise ValueError("nonempty unique targets required")

    def configure(spec):
        spec.worldbody.add_camera(name="inspection", pos=(0.55, 0, 1.8), quat=(1, 0, 0, 0), fovy=35)
        for target in targets:
            material = spec.add_material(name=f"visual_{target.label}", rgba=target.rgba, emission=1, specular=0)
            spec.worldbody.add_geom(
                name=f"target_{target.label}",
                type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                pos=(target.position[0], target.position[1], target.position[2] - 0.001),
                size=(target.radius, 0.001, 0),
                material=material.name,
                contype=0,
                conaffinity=0,
                mass=0,
                group=1,
            )

    world = panda_reach(
        model_path, ReachTask((0.8, 0.2, 0.5)), increment=increment, axes=axes, physics_steps=physics_steps, configure_spec=configure
    )

    def target_entities():
        entities = []
        for target in targets:
            geom = world.backend.model.geom(f"target_{target.label}").id
            position = world.backend.data.geom_xpos[geom].copy()
            position[2] += world.backend.model.geom_size[geom, 1]
            entities.append(Entity(target.label, tuple(float(x) for x in position)))
        return tuple(entities)

    # Explicit ground-truth access for the observation baseline/diagnostics only.
    world.target_entities = target_entities
    return world
