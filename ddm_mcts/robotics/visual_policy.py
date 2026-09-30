"""Future direct VLDM boundary, distinct from image-to-state VLM perception."""

from collections.abc import Mapping, Sequence
from typing import Protocol

from ddm_mcts.policies.base import Policy, normalize_probabilities

from .observation import Observation, SemanticGoal


class VisualDecisionModel(Protocol):
    def score(self, rgb, goal: SemanticGoal, actions: Sequence) -> Mapping: ...


class VisualDecisionPolicy(Policy):
    """Inject a direct image+goal+choices scorer as MCTS's ROOT policy.

    Future nodes have predicted structured state, not predicted images. Accordingly
    this adapter is root-only; the ordinary structured/uniform policy handles
    expanded future nodes. A new live image is bound before every decision.
    """

    def __init__(self, model: VisualDecisionModel):
        self.model = model
        self.observation = None
        self.goal = None
        self.calls = 0

    def bind(self, observation: Observation, goal: SemanticGoal):
        if observation.rgb is None:
            raise ValueError("direct visual policy requires an RGB observation")
        self.observation, self.goal = observation, goal

    def probabilities(self, state, legal_actions):
        if self.observation is None:
            raise RuntimeError("bind a live observation before using the visual policy")
        if getattr(state, "depth", 0) != 0:
            raise ValueError("direct visual priors must be attached as root_policy; no predicted images exist")
        self.calls += 1
        return normalize_probabilities(self.model.score(self.observation.rgb, self.goal, legal_actions), legal_actions)

    def diagnostics(self):
        return {"direct_visual_inference_calls": self.calls}
