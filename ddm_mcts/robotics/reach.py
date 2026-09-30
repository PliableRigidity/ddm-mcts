"""Composable Cartesian actions, reaching task, and first Panda integration."""

from dataclasses import dataclass
from math import dist, isfinite

from .core import RoboticsEnvironment, Task


@dataclass(frozen=True)
class CartesianAction:
    name: str
    displacement: tuple[float, float, float]

    def __str__(self):
        return self.name


def cartesian_actions(increment: float = 0.02, axes: tuple[int, ...] = (0, 1, 2)):
    if (
        not isfinite(increment)
        or not 0 < increment <= 0.05
        or not axes
        or len(set(axes)) != len(axes)
        or any(a not in (0, 1, 2) for a in axes)
    ):
        raise ValueError("invalid increment or axes")
    return tuple(
        CartesianAction(f"MOVE_{'XYZ'[axis]}_{name}", tuple(sign * increment if i == axis else 0.0 for i in range(3)))
        for axis in axes
        for name, sign in [("POS", 1), ("NEG", -1)]
    )


@dataclass(frozen=True)
class ReachState:
    position: tuple[float, float, float]
    goal: tuple[float, float, float]
    steps: int
    time: float
    qpos: tuple[float, ...]
    qvel: tuple[float, ...]


@dataclass(frozen=True)
class ReachTask(Task[ReachState]):
    goal: tuple[float, float, float]
    epsilon: float = 0.012
    max_steps: int = 30

    def __post_init__(self):
        if (
            len(self.goal) != 3
            or not all(isfinite(x) for x in self.goal)
            or not isfinite(self.epsilon)
            or self.epsilon <= 0
            or self.max_steps < 1
        ):
            raise ValueError("invalid reach task")

    def evaluate(self, state):
        return -dist(state.position, self.goal)

    def is_success(self, state):
        return -self.evaluate(state) <= self.epsilon

    def is_terminal(self, state):
        return self.is_success(state) or state.steps >= self.max_steps


@dataclass(frozen=True)
class RobotSnapshot:
    physics: object
    steps: int


class ControlledRobot(RoboticsEnvironment):
    """Task and action set are independent of the robot/controller."""

    def __init__(self, backend, controller, task, actions, reset_keyframe=None):
        if not actions or len(set(actions)) != len(actions):
            raise ValueError("nonempty unique action set required")
        self.backend, self.controller, self.task = backend, controller, task
        self.actions, self.reset_keyframe = tuple(actions), reset_keyframe
        self.steps = 0
        self.reset()

    def get_state(self):
        data = self.backend.data
        return ReachState(
            self.controller.position(),
            self.task.goal,
            self.steps,
            float(data.time),
            tuple(float(x) for x in data.qpos),
            tuple(float(x) for x in data.qvel),
        )

    def legal_actions(self):
        return () if self.task.is_terminal(self.get_state()) else self.actions

    def step(self, action):
        if action not in self.legal_actions():
            raise ValueError(f"illegal action: {action}")
        self.controller.execute(action.displacement)
        self.steps += 1
        return self.get_state()

    def snapshot(self):
        return RobotSnapshot(self.backend.snapshot(), self.steps)

    def restore(self, snapshot):
        self.backend.restore(snapshot.physics)
        self.steps = snapshot.steps

    def reset(self):
        self.backend.reset(self.reset_keyframe)
        self.steps = 0
        return self.get_state()


def panda_reach(model_path, task: ReachTask, *, increment=0.02, axes=(0, 1, 2), physics_steps=150):
    from .controller import CartesianController
    from .mujoco_backend import MujocoBackend

    backend = MujocoBackend(model_path, physics_steps=physics_steps, gravity_compensation=True)
    # Menagerie position servos need gravity compensation for Cartesian holding.
    controller = CartesianController(backend, "hand", tuple(f"joint{i}" for i in range(1, 8)), tuple(f"actuator{i}" for i in range(1, 8)))
    return ControlledRobot(backend, controller, task, cartesian_actions(increment, axes), "home")
