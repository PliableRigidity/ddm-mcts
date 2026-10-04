"""V5 contact compliance for dynamic support changes; V4 scenes stay frozen."""

from .manipulation_scene import panda_pickup_scene


def panda_reasoning_scene(model_path, *, physics_steps=25):
    world = panda_pickup_scene(model_path, physics_steps=physics_steps)
    # A toppling cube falls ~50 mm. The V4 5 ms object contact response exceeded the
    # unchanged 3 mm penetration bound during impact; the V4 table uses 20 ms,
    # giving a measured object/table contact response of 12.5 ms. Use the safe minimum
    # 4 ms (= two 2 ms timesteps), preserving friction, gravity, mass and dynamics.
    # This is contact compliance, not adhesion, stabilization or state attachment.
    for name in ("pickup_cube_geom", "pickup_cylinder_geom", "pickup_table"):
        world.backend.model.geom_solref[world.backend.model.geom(name).id, 0] = 0.004
    return world
