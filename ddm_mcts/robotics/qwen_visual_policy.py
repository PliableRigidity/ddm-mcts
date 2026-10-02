"""Prompted image/action priors; not a trained robotics VLDM or controller."""

import json
import math
from time import perf_counter

from ddm_mcts.policies.mixed_policy import MixedPolicy
from ddm_mcts.policies.permutation_averaged_policy import PermutationAveragedPolicy

from .visual_policy import VisualDecisionPolicy


def parse_scores(raw, actions):
    """Reject duplicate/missing/unknown IDs before semantic normalization."""

    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate visual score field: {key}")
            result[key] = value
        return result

    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end < start:
        raise ValueError("visual decision requires JSON scores")
    try:
        obj = json.loads(raw[start : end + 1], object_pairs_hook=unique)
        scores = obj["scores"]
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError("visual decision requires a JSON scores object") from exc
    names = [a.name for a in actions]
    if len(set(names)) != len(names) or not isinstance(scores, dict) or set(scores) != set(names):
        raise ValueError("visual scores must contain every legal stable ID exactly once and no unknown IDs")
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 for v in scores.values()):
        raise ValueError("visual scores must be finite nonnegative numbers")
    scale = max(scores.values(), default=0)
    if scale <= 0:
        raise ValueError("visual scores have zero total mass")
    total = sum(v / scale for v in scores.values())
    distribution = {a: (scores[a.name] / scale) / total for a in actions}
    return distribution, scores, obj.get("explanation")


class QwenVisualDecisionModel:
    """Only RGB/goal/actions plus camera and TCP proprioception reach the prompt."""

    def __init__(self, backend):
        self.backend = backend
        self.camera = self.tcp = None
        self.records = []

    def bind_context(self, camera, tcp):
        if camera is None:
            raise ValueError("visual decision requires calibrated camera")
        self.camera, self.tcp = camera, tuple(tcp)
        self.records = []

    def describe(self, action):
        camera = self.camera
        uv = camera.project(self.tcp)
        destination = tuple(p + d for p, d in zip(self.tcp, action.displacement, strict=True))
        end = camera.project(destination)
        vector = action.displacement
        axes = ", ".join(f"world {'XYZ'[i]} {v:+.3f} m" for i, v in enumerate(vector) if v)
        return {
            "id": action.name,
            "description": f"Move the gripper TCP by {axes}. World Z positive is up, negative is down.",
            "camera_motion_pixels": {"right": round(end[0] - uv[0], 2), "down": round(end[1] - uv[1], 2)},
        }

    def score(self, rgb, goal, actions):
        import numpy as np
        from PIL import Image, ImageDraw

        # A current-TCP overlay is allowed proprioception, not target information.
        image = Image.fromarray(rgb)
        x, y = self.camera.project(self.tcp)
        if not (0 <= x < image.width and 0 <= y < image.height):
            raise ValueError("current TCP outside policy camera view")
        draw = ImageDraw.Draw(image)
        draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=(255, 0, 255), outline=(255, 255, 255), width=1)
        descriptions = [self.describe(a) for a in actions]
        prompt = (
            f"Goal: {goal.label}\n"
            "Score the listed high-level actions as promising next moves for SAFE APPROACH reaching. "
            "The magenta dot marks the CURRENT gripper TCP, not a target. Identify the requested freestanding shape from the image. "
            "Do not touch or grasp: stop above the requested object with clearance. Lift in +Z ONLY while the current TCP is below the object top. Once above it, prefer horizontal alignment over the requested object; do NOT keep lifting indefinitely. "
            "avoid descending into them. Camera pixel origin top-left; X right, Y down. "
            "The supplied camera-motion vectors describe how each action moves the CURRENT TCP in this camera view; "
            "they do not describe target coordinates. World X/Y are horizontal, world +Z is vertically up. "
            "Give finite nonnegative preference scores, every exact action ID once. "
            "Include LOW or ZERO scores for unpromising actions; NEVER omit any ID. "
            "No world coordinates or executable commands.\nCandidates: "
            + json.dumps(descriptions)
            + "\nRequired JSON schema (replace every score with your preference, retain ALL keys): "
            + json.dumps({"scores": {a.name: 0 for a in actions}})
            + "\nReplace all template zeroes with your own numeric scores from 0 to 100 (unquoted JSON numbers). At least one score must be positive. Favor useful/safe directions over wrong or unsafe directions; do not assign equal scores unless genuinely equally useful. Return all listed scores only, no explanation or prose."
        )
        started = perf_counter()
        raw = self.backend.generate(np.asarray(image), prompt)
        probabilities, scores, explanation = parse_scores(raw, actions)
        self.records.append(
            {
                "goal": goal.label,
                "candidate_actions": descriptions,
                "presentation_order": [a.name for a in actions],
                "raw_response": raw,
                "parsed_scores": scores,
                "normalized_probabilities": {a.name: p for a, p in probabilities.items()},
                "explanation": explanation,
                "request_seconds_including_load": perf_counter() - started,
            }
        )
        return probabilities


class QwenVisualPolicy(VisualDecisionPolicy):
    """Existing root boundary + V1 trust/permutation utilities; fresh-image scope."""

    def __init__(self, backend, *, alpha=0.5, permutations=1, seed=0):
        model = QwenVisualDecisionModel(backend)
        super().__init__(model)
        self.raw_policy = VisualDecisionPolicy(model)
        self.averaged = PermutationAveragedPolicy(self.raw_policy, permutations, seed)
        self.mixed = MixedPolicy(self.averaged, alpha)
        self.alpha = alpha
        self.hits = self.misses = 0
        self.last = {}
        self._cache = {}
        self.inference_times = []

    def bind(self, observation, goal):
        super().bind(observation, goal)
        self.raw_policy.bind(observation, goal)
        self.model.bind_context(observation.camera, observation.robot.position)
        self.averaged._cache.clear()  # Never reuse old-image priors for a new observation.
        self._cache.clear()

    def probabilities(self, state, legal_actions):
        if self.observation is None:
            raise RuntimeError("bind a live observation before using the visual policy")
        if getattr(state, "depth", 0) != 0:
            raise ValueError("visual priors require root_policy; deeper nodes have no images")
        self.calls += 1
        legal = tuple(sorted(legal_actions, key=lambda a: a.name))
        if legal in self._cache:
            self.hits += 1
            self.last = {**self.last, "physical_inference_seconds": 0, "visual_policy_seconds": 0, "cache_hit": True}
            return dict(self._cache[legal])
        self.misses += 1
        self.model.records = []
        before = len(self.model.backend.inference_seconds)
        started = perf_counter()
        try:
            used = self.mixed.probabilities((self.observation.sequence, self.goal.label), legal)
        finally:
            self.inference_times.extend(self.model.backend.inference_seconds[before:])
        records = self.model.records
        visual = {a.name: sum(r["normalized_probabilities"][a.name] for r in records) / len(records) for a in legal} if records else None
        self.last = {
            "observation_sequence": self.observation.sequence,
            "goal": self.goal.label,
            "model_id": self.model.backend.config.model_id,
            "alpha": self.alpha,
            "model_top1": max(visual, key=visual.get) if visual else None,
            "normalized_visual_probabilities": visual,
            "actual_root_priors": {a.name: p for a, p in used.items()},
            "presentations": records.copy(),
            "physical_inference_seconds": sum(self.model.backend.inference_seconds[before:]),
            "visual_policy_seconds": perf_counter() - started,
        }
        self._cache[legal] = dict(used)
        return used

    def diagnostics(self):
        return {
            "logical_visual_requests": self.calls,
            "cache_hits": self.hits,
            "cache_misses": self.misses,
            "physical_visual_inference_calls": len(self.inference_times),
            "physical_visual_inference_seconds": sum(self.inference_times),
            "mean_visual_inference_seconds": sum(self.inference_times) / len(self.inference_times) if self.inference_times else 0,
            "backend": self.model.backend.diagnostics(),
            "permutations": self.averaged.samples,
            "deep_node_policy": "configured structured policy (UniformPolicy by default)",
            "last_decision": self.last,
        }
