"""Separate dynamic pickup workspace; does not alter V3's static visual scene."""

from dataclasses import dataclass

import mujoco
import numpy as np

from .pose_control import PoseController, PoseTarget
from .reach import ReachTask, panda_reach


@dataclass(frozen=True)
class ManipulationObject:
    label: str
    body_name: str
    shape: str
    dimensions: tuple[float, float, float]  # full XYZ bounds
    pose: PoseTarget


TABLE_HEIGHT = 0.35


def panda_pickup_scene(model_path, *, physics_steps=25):
    specs = (
        ("cube", mujoco.mjtGeom.mjGEOM_BOX, (0.02, 0.02, 0.02), (0.50, -0.08, 0.372)),
        ("cylinder", mujoco.mjtGeom.mjGEOM_CYLINDER, (0.02, 0.025, 0), (0.50, 0.08, 0.377)),
    )

    def configure(spec):
        # Use 5 ms critically damped contact compliance for the small pads;
        # the default 20 ms contact response allowed the cylinder to slip out.
        # Increase the soft finger servo's stiffness 100 -> 400 N/m and
        # damping 10 -> 20 N s/m to prevent long-hold slip. Preserve width
        # mapping, original force limits and friction; external XML is untouched.
        actuator = next(a for a in spec.actuators if a.name == "actuator8")
        actuator.gainprm[0] *= 4
        actuator.biasprm[1] *= 4
        actuator.biasprm[2] *= 2
        for name in ("left_finger", "right_finger"):
            for geom in spec.body(name).geoms:
                if geom.type == mujoco.mjtGeom.mjGEOM_BOX:
                    geom.solref = (0.005, 1)
        spec.body("hand").add_site(name="panda_gripper_tcp", pos=(0, 0, 0.1034), size=(0.003, 0, 0), rgba=(0, 0, 0, 0))
        table = spec.worldbody.add_body(name="support_table", pos=(0.52, 0, TABLE_HEIGHT - 0.025))
        table.add_geom(
            name="pickup_table",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=(0.18, 0.22, 0.025),
            rgba=(0.45, 0.38, 0.3, 1),
            friction=(1, 0.005, 0.0001),
        )
        for label, kind, size, position in specs:
            body = spec.worldbody.add_body(name=f"pickup_{label}", pos=position)
            body.add_freejoint(name=f"pickup_{label}_free")
            body.add_geom(
                name=f"pickup_{label}_geom",
                type=kind,
                size=size,
                mass=0.08,
                rgba=(0.15, 0.8, 0.7, 1),
                friction=(1, 0.005, 0.0001),
                condim=4,
                solref=(0.005, 1),
            )
        # Extend initialization keyframes with free-body poses. Without this,
        # mj_resetDataKeyframe would initialize the new free joints at zero.
        for key in spec.keys:
            key.qpos = np.concatenate((key.qpos, *(np.array((*position, 1, 0, 0, 0)) for _, _, _, position in specs)))
        # Objects are added after arm gravity compensation; default gravcomp=0.

    world = panda_reach(
        model_path, ReachTask((0.5, 0, 0.6)), physics_steps=physics_steps, configure_spec=configure, tcp_site="panda_gripper_tcp"
    )
    world.controller = PoseController(
        world.backend, "hand", tuple(f"joint{i}" for i in range(1, 8)), tuple(f"actuator{i}" for i in range(1, 8)), site="panda_gripper_tcp"
    )

    def object_state(label):
        if label not in ("cube", "cylinder"):
            raise ValueError("pickup supports cube or cylinder")
        model, data = world.backend.model, world.backend.data
        body = model.body(f"pickup_{label}").id
        geom = model.geom(f"pickup_{label}_geom").id
        size = model.geom_size[geom]
        dimensions = (2 * size[0], 2 * size[1], 2 * size[2]) if label == "cube" else (2 * size[0], 2 * size[0], 2 * size[1])
        q = np.empty(4)
        mujoco.mju_mat2Quat(q, data.xmat[body])
        return ManipulationObject(
            label,
            f"pickup_{label}",
            "box" if label == "cube" else "cylinder",
            tuple(float(x) for x in dimensions),
            PoseTarget(tuple(data.xpos[body]), tuple(q)),
        )

    world.object_state = object_state
    for _ in range(20):
        world.backend.advance()  # physical settling before the episode; no resets inside pickup
    return world
