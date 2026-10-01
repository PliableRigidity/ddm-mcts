"""VLM chooses WHICH object; shared foreground and calibration determine WHERE."""

from dataclasses import asdict, replace
from math import isfinite

import numpy as np

from .color_perception import ColorPlanePerception
from .observation import Entity
from .representation import WorldState


class VLMSemanticPerception:
    """Static-scene semantic cache, with fresh RGB metric checks every observation.

    refresh=0 reuses semantics until reset/goal/camera change; refresh=N reruns
    inference every N observations. All shapes share one foreground material.
    That material refines a VLM box but cannot choose the requested shape.
    """

    def __init__(self, backend, plane_z, foreground_rgb, *, refresh=0, debug=False, max_missed_frames=8):
        if not isfinite(plane_z) or refresh < 0 or max_missed_frames < 0:
            raise ValueError("invalid VLM perception plane/refresh/tracking configuration")
        color = np.asarray(foreground_rgb, dtype=float)
        if color.shape != (3,) or not np.isfinite(color).all() or np.any(color < 0) or color.sum() <= 0:
            raise ValueError("invalid shared foreground color")
        self.backend, self.plane_z = backend, plane_z
        self.foreground = color / color.sum()
        self.refresh, self.debug, self.max_missed_frames = refresh, debug, max_missed_frames
        self.reset()

    def reset(self):
        self.semantic = self.previous = self._key = None
        self._age = self._area = 0
        self.last = {}

    def perceive(self, observation, goal):
        if observation.rgb is None or observation.camera is None or goal is None:
            raise ValueError("VLM semantic perception requires RGB, calibration and a language goal")
        rgb, camera = observation.rgb, observation.camera
        if rgb.dtype != np.uint8 or rgb.shape != (camera.height, camera.width, 3):
            raise ValueError("expected calibrated uint8 RGB HxWx3")
        key = (goal.label, camera)
        refreshed = self.semantic is None or key != self._key or (self.refresh > 0 and self._age >= self.refresh)
        if refreshed:
            result = self.backend.infer(rgb, goal.label)
            self.semantic, self._key, self._age = result, key, 0
            self.previous, self._area = None, 0
        self._age += 1
        box = self.semantic.bbox
        x1, y1, x2, y2 = (v * (camera.width if i % 2 == 0 else camera.height) for i, v in enumerate(box))
        # Small padding tolerates approximate VLM grounding; it does not use scene
        # IDs, shape labels, true positions or goal-to-location mappings.
        px, py = (x2 - x1) * 0.2, (y2 - y1) * 0.2
        values = rgb.astype(float)
        intensity = values.sum(axis=2)
        chroma = values / np.maximum(intensity[..., None], 1)
        mask = (np.linalg.norm(chroma - self.foreground, axis=2) < 0.12) & (intensity > 80)
        rows, cols = np.indices(mask.shape)
        mask &= (cols >= x1 - px) & (cols <= x2 + px) & (rows >= y1 - py) & (rows <= y2 + py)
        components = ColorPlanePerception._components(mask)
        candidate = components[0] if components else []
        if len(components) > 1 and len(components[1]) >= max(12, 0.8 * len(candidate)):
            raise ValueError("VLM grounding overlaps multiple foreground objects; target ambiguous")
        if len(candidate) >= max(12, 0.65 * self._area):
            r, c = np.asarray(candidate).T
            pixel = (float(c.mean()), float(r.mean()))
            position = camera.intersect_plane(pixel, self.plane_z)
            entity = Entity(self.semantic.target_label, position, pixel, len(candidate))
            self._area = max(self._area, len(candidate))
        elif self.previous is not None and self.previous.missed_frames < self.max_missed_frames:
            entity = replace(self.previous, observed=False, missed_frames=self.previous.missed_frames + 1, pixels=len(candidate))
        else:
            raise ValueError("VLM-grounded target has no valid foreground pixels or tracking expired; no ground-truth fallback")
        self.previous = entity
        self.last = {
            "semantic_result": asdict(self.semantic),
            "semantic_refreshed": refreshed,
            "semantic_age": self._age,
            "localized_entity": asdict(entity),
        }
        return WorldState(observation.robot, (entity,), goal, "qwen-semantic-geometry", self.semantic.target_label)

    def configuration(self):
        return {
            "backend": asdict(self.backend.config),
            "plane_z": self.plane_z,
            "shared_foreground": self.foreground.tolist(),
            "refresh": self.refresh,
            "debug": self.debug,
            "max_missed_frames": self.max_missed_frames,
            "foreground_chromaticity_tolerance": 0.12,
            "bbox_padding_fraction": 0.2,
            "partial_visibility_ratio": 0.65,
        }

    def diagnostics(self):
        return {**self.last, "backend": self.backend.diagnostics()}

    def save_debug(self, folder, observation):
        if not self.debug or not self.last:
            return
        import json

        from PIL import Image, ImageDraw

        path = folder / "vlm_debug"
        path.mkdir(exist_ok=True)
        prefix = f"observation_{observation.sequence:04d}"
        image = Image.fromarray(observation.rgb)
        image.save(path / f"{prefix}_input.png")
        draw = ImageDraw.Draw(image)
        box = tuple(v * (image.width if i % 2 == 0 else image.height) for i, v in enumerate(self.semantic.bbox))
        draw.rectangle(box, outline="red", width=2)
        if self.previous.pixel is not None:
            x, y = self.previous.pixel
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill="yellow")
        image.save(path / f"{prefix}_grounding.png")
        (path / f"{prefix}.json").write_text(json.dumps(self.diagnostics(), indent=2) + "\n")
