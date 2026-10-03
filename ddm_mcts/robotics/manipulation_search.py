"""Actual MuJoCo skill futures for choosing geometry-valid placement alternatives."""

import contextlib
import io
from dataclasses import asdict, dataclass

import numpy as np

from ddm_mcts.search.mcts import MCTSConfig

from .core import RoboticsPlanner
from .pick_place import PickPlaceAgent
from .placement import NextTo, PlacementTarget, PlacementWorkspace, horizontal_extent, verify_next_to


@dataclass(frozen=True)
class PhysicalAction:
    action_id: str
    target: str
    destination: PlacementTarget | None

    def __str__(self):
        return self.action_id


def candidate_actions(world, operation, workspace=None):
    if not isinstance(operation.destination, NextTo):
        return (
            PhysicalAction(
                f"{'PICK' if operation.destination is None else 'PICK_PLACE'}:{operation.target}", operation.target, operation.destination
            ),
        )
    workspace = workspace or PlacementWorkspace()
    obj = world.object_state(operation.target)
    ref = world.object_state(operation.destination.reference)
    if obj.label == ref.label:
        raise ValueError("self-reference is not a destination")
    choices = []
    # Keep the frozen Phase 2 reference jaw-sweep constraint: narrow gaps use X.
    for sign in (-1, 1):
        direction = (sign, 0, 0)
        offset = horizontal_extent(obj, direction) + horizontal_extent(ref, direction) + operation.destination.gap
        p = np.asarray(ref.pose.position).copy()
        p[0] += sign * offset
        p[2] = workspace.surface_z + obj.dimensions[2] / 2
        target = PlacementTarget(tuple(p))
        try:
            workspace.validate(target, obj, [ref])
        except ValueError:
            continue
        choices.append(PhysicalAction(f"PICK_PLACE:{obj.label}:NEXT_TO:{ref.label}:X_{'NEG' if sign < 0 else 'POS'}", obj.label, target))
    if not choices:
        raise ValueError("invalid_destination: no feasible next_to candidate")
    return tuple(choices)


@dataclass(frozen=True)
class SkillOutcome:
    terminal: bool = False
    success: bool = False
    cost: float = 0.0


class SkillTask:
    @staticmethod
    def evaluate(state):
        # Verified physical success dominates; among successes prefer less carry travel.
        return (-min(state.cost, 1.0)) if state.success else (-2.0 if state.terminal else 0.0)

    @staticmethod
    def is_terminal(state):
        return state.terminal


class SkillWorldModel:
    """Short high-level futures. Frozen controllers/contact verifiers do transitions.

    Rendering, semantic inference and live-viewer callbacks are absent in search.
    Speculative skill traces are retained separately; physical integration state
    and progress are restored by the existing SimulatorAdapter, including failures.
    """

    def __init__(self, world, actions, output, relation=None):
        self.world, self.actions, self.output, self.relation = world, tuple(actions), output, relation
        self.task = SkillTask()
        self.outcome = SkillOutcome()
        self.transitions = []

    def get_state(self):
        return self.outcome

    def legal_actions(self):
        return () if self.outcome.terminal else self.actions

    def snapshot(self):
        return self.world.snapshot(), self.outcome

    def restore(self, snapshot):
        self.world.restore(snapshot[0])
        self.outcome = snapshot[1]

    def step(self, action):
        if action not in self.legal_actions():
            raise ValueError("invalid high-level action")
        with contextlib.redirect_stdout(io.StringIO()):
            result = PickPlaceAgent(self.world).pick_and_place(action.target, action.destination, self.output)
        row = result.trace["operations"][0]
        success = result.success
        if self.relation and success:
            success = verify_next_to(
                self.world.object_state(action.target), self.world.object_state(self.relation.reference), self.relation.gap
            ).success
        cost = row["transport"]["path_length_m"]
        self.outcome = SkillOutcome(True, success, cost)
        self.transitions.append(
            {
                "action": asdict(action),
                "success": success,
                "carry_path_m": cost,
                "value": self.task.evaluate(self.outcome),
                "skill_log": str(result.folder),
            }
        )
        return self.outcome


def plan_skill(world, operation, output, *, policy=None, config=None):
    actions = candidate_actions(world, operation)
    if len(actions) == 1:
        return actions[0], {"forced": True, "simulations": 0, "candidates": [asdict(actions[0])]}
    future = SkillWorldModel(world, actions, output, operation.destination)
    before = world.snapshot()
    planner = RoboticsPlanner(future, policy, config or MCTSConfig(60, c_puct=0.05, seed=0), horizon=1)
    result = planner.plan()
    if world.snapshot() != before:
        raise RuntimeError("search changed live physical state")
    statistics = [{**s, "action": str(s["action"])} for s in result.root_statistics()]
    return result.action, {
        "prior_source": type(policy).__name__ if policy is not None else "uniform",
        "c_puct": planner.search.config.c_puct,
        "forced": False,
        "simulations": result.simulations,
        "horizon": 1,
        "candidates": [asdict(a) for a in actions],
        "statistics": statistics,
        "transitions": future.transitions,
        "selected": str(result.action),
        "snapshot_preserved": True,
    }
