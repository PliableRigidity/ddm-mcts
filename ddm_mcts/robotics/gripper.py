"""Physical Menagerie Panda tendon actuation; widths are summed joint travel."""

from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class GripperState:
    commanded_width: float
    actual_width: float
    left_position: float
    right_position: float
    moving: bool


class PandaGripper:
    def __init__(self, backend):
        self.backend = backend
        model = backend.model
        self.actuator = model.actuator("actuator8").id
        self.joints = [model.joint(name).id for name in ("finger_joint1", "finger_joint2")]
        self.addresses = model.jnt_qposadr[self.joints]
        self.dofs = model.jnt_dofadr[self.joints]
        self.maximum_width = float(sum(model.jnt_range[j, 1] for j in self.joints))
        self.minimum_width = float(sum(model.jnt_range[j, 0] for j in self.joints))

    def state(self):
        data, model = self.backend.data, self.backend.model
        positions = data.qpos[self.addresses]
        low, high = model.actuator_ctrlrange[self.actuator]
        commanded = self.minimum_width + (data.ctrl[self.actuator] - low) / (high - low) * (self.maximum_width - self.minimum_width)
        return GripperState(
            float(commanded), float(sum(positions)), *map(float, positions), any(abs(v) > 0.001 for v in data.qvel[self.dofs])
        )

    def get_width(self):
        return self.state().actual_width

    def set_width(self, width, *, cycles=12):
        if not isfinite(width) or not self.minimum_width <= width <= self.maximum_width or cycles < 1:
            raise ValueError("gripper width outside physical joint bounds or invalid duration")
        low, high = self.backend.model.actuator_ctrlrange[self.actuator]
        self.backend.data.ctrl[self.actuator] = low + (high - low) * (width - self.minimum_width) / (
            self.maximum_width - self.minimum_width
        )
        for _ in range(cycles):
            self.backend.advance()
        return self.state()

    def open(self):
        return self.set_width(self.maximum_width)

    def close(self):
        return self.set_width(self.minimum_width)
