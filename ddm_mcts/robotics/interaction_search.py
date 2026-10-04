"""MCTS over real, bounded Panda pushes; speculative physics is restored."""

from copy import deepcopy
from dataclasses import asdict, dataclass

from ddm_mcts.search.mcts import MCTSConfig

from .core import RoboticsPlanner
from .interactions import InteractionExecutor, push_candidates
from .physical_goals import PhysicalGoal, Predicate
from .physical_predicates import PhysicalState


@dataclass(frozen=True)
class InteractionOutcome:
    terminal: bool = False
    success: bool = False
    progress: float = 0
    cost: float = 0
    failure: str = ""


class InteractionTask:
    def evaluate(self, state):
        return -state.cost if state.success else (-2 + min(state.progress, 1)) if state.terminal else 0

    def is_terminal(self, state):
        return state.terminal


class InteractionWorldModel:
    def __init__(self, world, state, actions, goal):
        self.world, self.state, self.actions, self.goal = world, state, actions, goal
        self.task = InteractionTask()
        self.outcome = InteractionOutcome()
        self.transitions = []
        self.search_substeps = 0

    def get_state(self):
        return self.outcome

    def legal_actions(self):
        return () if self.outcome.terminal else self.actions

    def snapshot(self):
        return self.world.snapshot(), deepcopy((self.state.history, self.state.groups, self.state.held_reference)), self.outcome

    def restore(self, snapshot):
        self.world.restore(snapshot[0])
        self.state.history, self.state.groups, self.state.held_reference = deepcopy(snapshot[1])
        self.outcome = snapshot[2]

    def step(self, action):
        executor = InteractionExecutor(self.world, state=self.state)
        failure = ""
        try:
            executed = executor.push(action)
            verified = self.state.evaluate(self.goal)
        except RuntimeError as exc:
            failure = str(exc)
            executed = {}
            verified = self.state.evaluate(self.goal)
        self.search_substeps += executor.counts["substeps"]
        tilt = verified.metrics.get("tilt", verified.metrics.get("maximum_tilt", 0))
        self.outcome = InteractionOutcome(True, verified.satisfied and not failure, tilt, action.distance, failure)
        self.transitions.append(
            {
                "action": asdict(action),
                "execution": executed,
                "verification": asdict(verified),
                "failure": failure,
                "physics": executor.counts,
                "value": self.task.evaluate(self.outcome),
            }
        )
        return self.outcome


def search_push(world, state, subject, *, policy=None, config=None):
    target = state.groups["tower"][1] if subject == "tower" and "tower" in state.groups else subject
    if target == "tower":
        raise ValueError("no verified tower group")
    actions = push_candidates(world, target)
    if len(actions) < 2:
        raise ValueError("insufficient physically feasible push alternatives")
    future_state = PhysicalState(world)
    future_state.history, future_state.groups, future_state.held_reference = deepcopy((state.history, state.groups, state.held_reference))
    future = InteractionWorldModel(world, future_state, actions, PhysicalGoal(Predicate.TOPPLED, subject))
    before = world.snapshot()
    planner = RoboticsPlanner(future, policy, config or MCTSConfig(60, c_puct=0.05, seed=0), horizon=1)
    result = planner.plan()
    chosen = next(row for row in future.transitions if row["action"] == asdict(result.action))
    if chosen["failure"]:
        raise RuntimeError("no_safe_search_selection: " + chosen["failure"])
    if world.snapshot() != before:
        raise RuntimeError("speculative search contaminated live physics")
    return result.action, {
        "simulations": result.simulations,
        "prior_source": type(policy).__name__ if policy else "uniform",
        "candidates": [asdict(a) for a in actions],
        "selected": str(result.action),
        "statistics": [{**s, "action": str(s["action"])} for s in result.root_statistics()],
        "transitions": future.transitions,
        "speculative_substeps": future.search_substeps,
        "snapshot_restored": True,
    }
