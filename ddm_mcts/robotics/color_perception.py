"""Deterministic RGB segmentation and explicit known-plane target localization."""

from dataclasses import asdict, dataclass, replace
from math import isfinite

import numpy as np

from .observation import Entity
from .representation import WorldState


@dataclass(frozen=True)
class ColorTarget:
    label: str
    rgb: tuple[float, float, float]  # chromaticity prototype (0..1)


DEFAULT_COLORS = (ColorTarget("red", (1, 0, 0)), ColorTarget("blue", (0, 0, 1)))


class ColorPlanePerception:
    def __init__(self, plane_z: float, colors=DEFAULT_COLORS, tolerance=0.22, min_pixels=12, max_missed_frames=6):
        if not isfinite(plane_z) or not 0 < tolerance < 1 or min_pixels < 1 or max_missed_frames < 0:
            raise ValueError("invalid color perception configuration")
        if not colors or len({c.label for c in colors}) != len(colors):
            raise ValueError("unique nonempty target colors required")
        self.plane_z, self.colors, self.tolerance = plane_z, tuple(colors), tolerance
        self.min_pixels, self.max_missed_frames = min_pixels, max_missed_frames
        self._previous = {}
        self._areas = {}

    @staticmethod
    def _components(mask):
        remaining = {tuple(pixel) for pixel in np.argwhere(mask)}
        components = []
        while remaining:
            start = min(remaining)
            remaining.remove(start)
            stack, pixels = [start], []
            while stack:
                row, col = stack.pop()
                pixels.append((row, col))
                for neighbour in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
                    if neighbour in remaining:
                        remaining.remove(neighbour)
                        stack.append(neighbour)
            components.append(pixels)
        return sorted(components, key=len, reverse=True)

    def perceive(self, observation, goal=None):
        if observation.rgb is None or observation.camera is None:
            raise ValueError("color perception requires RGB and camera calibration")
        rgb = np.asarray(observation.rgb)
        if rgb.dtype != np.uint8 or rgb.shape != (observation.camera.height, observation.camera.width, 3):
            raise ValueError("expected calibrated uint8 RGB image")
        values = rgb.astype(float)
        intensity = values.sum(axis=2)
        chroma = values / np.maximum(intensity[..., None], 1)
        entities = []
        for color in self.colors:
            prototype = np.asarray(color.rgb, dtype=float)
            if prototype.shape != (3,) or not np.isfinite(prototype).all() or np.any(prototype < 0) or prototype.sum() <= 0:
                raise ValueError("invalid color prototype")
            prototype /= prototype.sum()
            mask = (np.linalg.norm(chroma - prototype, axis=2) <= self.tolerance) & (intensity > 80)
            components = self._components(mask)
            candidate = components[0] if components else []
            previous = self._previous.get(color.label)
            # Partial occlusion moves a centroid. Retain the last full detection,
            # explicitly tagged stale, rather than inventing unseen coordinates.
            visible = len(candidate) >= self.min_pixels and len(candidate) >= 0.65 * self._areas.get(color.label, 0)
            if visible and len(components) > 1 and len(components[1]) >= 0.8 * len(candidate):
                raise ValueError(f"ambiguous {color.label} targets")
            if visible:
                rows, cols = np.asarray(candidate).T
                pixel = (float(cols.mean()), float(rows.mean()))
                entity = Entity(color.label, observation.camera.intersect_plane(pixel, self.plane_z), pixel, len(candidate))
                self._areas[color.label] = max(self._areas.get(color.label, 0), len(candidate))
            elif previous is not None and previous.missed_frames < self.max_missed_frames:
                entity = replace(previous, observed=False, missed_frames=previous.missed_frames + 1, pixels=len(candidate))
            else:
                continue
            self._previous[color.label] = entity
            entities.append(entity)
        state = WorldState(observation.robot, tuple(entities), goal, "color-plane")
        state.target()
        return state

    def configuration(self):
        return {
            "plane_z": self.plane_z,
            "colors": [asdict(color) for color in self.colors],
            "tolerance": self.tolerance,
            "min_pixels": self.min_pixels,
            "max_missed_frames": self.max_missed_frames,
        }

    def reset(self):
        self._previous.clear()
        self._areas.clear()
