"""Structured contacts and phase-aware manipulation collision rules."""

from dataclasses import dataclass

import mujoco
import numpy as np


@dataclass(frozen=True)
class ContactInfo:
    geom1: str
    geom2: str
    body1: str
    body2: str
    distance: float
    normal_force: float

    def involves(self, first, second):
        return {self.body1, self.body2} == {first, second}


class ContactInspector:
    def __init__(self, backend):
        self.backend = backend

    def contacts(self):
        model, data = self.backend.model, self.backend.data
        result = []
        for i in range(data.ncon):
            contact = data.contact[i]
            force = np.zeros(6)
            mujoco.mj_contactForce(model, data, i, force)
            geoms = [int(contact.geom1), int(contact.geom2)]
            result.append(
                ContactInfo(
                    *(model.geom(g).name or f"geom-{g}" for g in geoms),
                    *(model.body(int(model.geom_bodyid[g])).name for g in geoms),
                    float(contact.dist),
                    float(force[0]),
                )
            )
        return tuple(result)

    def touching(self, first, second):
        return any(c.involves(first, second) and c.distance <= 0.0005 and c.normal_force > 0.01 for c in self.contacts())


@dataclass(frozen=True)
class ContactRules:
    target: str
    allow_fingers: bool = False
    maximum_penetration: float = 0.003

    def violations(self, contacts):
        violations = []
        fingers = {"left_finger", "right_finger"}
        for contact in contacts:
            pair = {contact.body1, contact.body2}
            objects = {b for b in pair if b.startswith("pickup_")}
            robot = pair - objects - {"world", "support_table"}
            if not robot:
                continue  # ordinary object/support contact
            if objects or "support_table" in pair or "world" in pair:
                allowed = self.allow_fingers and self.target in objects and bool(robot) and robot <= fingers
                if not allowed or contact.distance < -self.maximum_penetration:
                    violations.append(contact)
        return tuple(violations)
