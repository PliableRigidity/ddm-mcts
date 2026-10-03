# V4 Phase 2: deterministic physical pick-and-place

This extension composes the frozen V4 Phase 1 pickup executor with measured-transform transport, contact-based placement, release and verification. It uses the existing dynamic Panda scene and original six-DoF controller, gripper and contacts. Qwen, VLM perception, DDM and MCTS are not invoked by these deterministic manipulation primitives. V3's separate approach/planning pipelines remain available unchanged.

## Commands

```bash
export PANDA_MODEL=/home/ishaan/robot-arm-playground/mujoco_menagerie/franka_emika_panda/scene.xml

# XYZ is the desired OBJECT center, not the TCP; table top is Z=0.35 m.
.venv/bin/python -m ddm_mcts.robotics.cli --model "$PANDA_MODEL" \
  --task pick-place --goal cube --place-position 0.60 -0.10 0.37 --viewer --diagnostics

.venv/bin/python -m ddm_mcts.robotics.cli --model "$PANDA_MODEL" \
  --task pick-place --goal cylinder --place-next-to cube --viewer --diagnostics

# One viewer and one physical scene: cylinder next_to cube, then cube elsewhere.
.venv/bin/python -m ddm_mcts.robotics.cli --model "$PANDA_MODEL" \
  --task rearrange-demo --viewer --diagnostics

# Headless equivalents: omit --viewer; EGL avoids needing a display.
MUJOCO_GL=egl .venv/bin/python -m ddm_mcts.robotics.cli --model "$PANDA_MODEL" \
  --task pick-place --goal cube --place-position 0.60 -0.10 0.37 --diagnostics
MUJOCO_GL=egl .venv/bin/python -m ddm_mcts.robotics.cli --model "$PANDA_MODEL" \
  --task pick-place --goal cylinder --place-next-to cube --diagnostics

# Existing Phase 1 command is unchanged.
.venv/bin/python -m ddm_mcts.robotics.cli --model "$PANDA_MODEL" \
  --task pickup --goal cube --viewer --diagnostics

# Repeatable real physics acceptance, including continuous plans.
MUJOCO_GL=egl .venv/bin/python -m examples.robotics.validate_pick_place \
  --model "$PANDA_MODEL" --trials 3 --sequential-trials 2
# Real GUI smoke; each viewer closes after its complete plan.
MUJOCO_GL=glfw .venv/bin/python -m examples.robotics.validate_pick_place \
  --model "$PANDA_MODEL" --trials 1 --sequential-trials 1 --viewer --viewer-speed 8
```

Normal CLI holds the final scene open. Close the viewer or press Ctrl+C; closing early stops physical execution. `--viewer-speed` affects pacing only. One display-copy viewer persists across all operations; no speculative actions are animated. Traces print high-level phases and retain detailed physical substeps/contact transitions in the runtime outputs.

## API

```python
from ddm_mcts.robotics.manipulation_scene import panda_pickup_scene
from ddm_mcts.robotics.pick_place import PickPlaceAgent, PickPlace, ManipulationPlan
from ddm_mcts.robotics.placement import PlacementTarget, NextTo

world = panda_pickup_scene(model_path)
agent = PickPlaceAgent(world)
plan = ManipulationPlan((
    PickPlace("cylinder", NextTo("cube", gap=0.04)),
    PickPlace("cube", PlacementTarget((0.46, 0.10, 0.37))),
))
result = agent.run_manipulation_plan(plan)
assert result.success, result.trace["operations"][-1].get("failure")
# Single-operation API, on another fresh scene if desired:
# result = agent.pick_and_place("cube", PlacementTarget((0.60, -0.10, 0.37)))
```

Caller-provided destinations must fit the calibrated table/workspace and avoid other objects. Position is the desired object center, orientation is a normalized wxyz quaternion, support is `support_table`. Defaults: position tolerance 10 mm, orientation tolerance 0.10 rad. Other support surfaces are rejected. Upright geometry is required; cube orientation uses SO(3) error, cylinder orientation uses axis tilt because yaw is symmetric. No unrestricted instruction parser or Qwen manipulation reasoning is implemented.

## Geometry and transforms

`compose`, `inverse`, `measured_grasp` and `placement_tcp` operate on rigid `PoseTarget`s:

`T_tcp_object = inverse(T_world_tcp) * T_world_object`

`T_world_tcp_desired = T_world_object_desired * inverse(T_tcp_object)`

The pickup trace retains its measured transform immediately after grasp verification. Placement remeasures the actual held transform after verified lift/hold, accounting for the cylinder's known initial contact motion. It does not assume ideal centering. Subsequent drift is monitored against that actual transform; no object state is overwritten to correct it. A cylinder preserves its measured yaw instead of introducing an unnecessary twist to arbitrary yaw zero.

`NextTo` uses reference pose and directional shape half-extents, target extents and requested gap. Candidate world ±X/±Y positions must fit table bounds X=[0.34,0.70], Y=[-0.22,0.22], with a 15 mm edge margin, resting center height and obstacle separation. Among feasible candidates it chooses the nearest to the source. Narrow Y gaps (<60 mm) are excluded to preserve the reference's open Panda jaw sweep for a subsequent regrasp; the known grasp closes along world Y. The default 40 mm gap therefore selects an X direction (−X in the recorded acceptance scene). There is no hard-coded cylinder destination. This is a calibrated scene constraint, not arbitrary clutter planning.

Final `next_to` verification independently reads **actual final** object/reference poses, computes horizontal center distance, directional surface gap and overlap, and requires upright objects and a gap in [requested−10 mm, requested+10 mm]. Default acceptable gap is 30–50 mm, with no overlap. Support is verified by placement and reference scene state. Moving the reference in a later operation deliberately changes the earlier relation; the earlier success records the relation when that operation completed.

## Physical execution and retention

Pickup is delegated to `ManipulationAgent.run(PickupTask(...))`; Phase 1 geometry, grasp/lift criteria, scene contact response, friction, servo parameters and actuator limits are unchanged. Transport uses a vertical waypoint, horizontal motion at safe object-center height, and a separate pre-place pose. Height accounts for table, reference top, held-object extent and a 60 mm clearance allowance. The measured object transform produces the required TCP poses. Held-object translation uses 1 mm increments and rotation 0.01 rad increments; physics runs throughout. Four-millimeter increments from the pickup trajectory caused transient finger unloading during the first transport diagnostic, so Phase 2 uses smaller increments without changing the frozen controller.

Every executed carry/descent substep inspects finger geometry/forces, actual width, relative positional drift (<15 mm), rotational drift (<0.35 rad), forbidden contacts, and transport support clearance (≥15 mm). Box rotation uses full SO(3) drift. Cylinder retention uses **axis tilt** for that bound, because axial yaw rotation does not change its collision geometry; full SO(3) drift is also explicitly logged. This does not imply the cylinder is held rigidly. Contact force can unload for at most 20 ms only while both pads still geometrically touch, width remains nonempty and the drift bounds remain satisfied. Missing geometric contact or exceeded drift fails immediately; prolonged unloaded contact fails. Actual acceptance force gaps were at most 4 ms. This bounded filter is tested; it is not a substitute for contact verification.

Pre-place verifies actual TCP pose, predicted versus actual object position (<15 mm), table clearance (≥25 mm) and other-object clearance (≥15 mm). Descent uses 1 mm increments, stopping at actual force-bearing object/table support. A bounded 2 mm below-nominal TCP allowance handles contact compliance; release does not depend solely on a Z command. Once table support exists, unloading of finger force is expected, but relative drift and forbidden-contact checks still apply.

Release actuates the existing gripper open. Persistent force-bearing finger contact fails release. Retreat moves upward 100 mm, with finger/object contact forbidden, and natural physics settles the object for one second. Success requires supported, released, upright placement inside tolerances. In the final half-second window, linear speed must stay <0.01 m/s, angular speed <0.10 rad/s, position variation <2 mm, and placement/support/release checks must remain valid. No velocities are forced to zero.

## Contacts, persistence and recovery

During carry, finger/target contact is intended and target/table contact forbidden. During descent/release, target/table support becomes intended. During retreat/settling, fingers must remain separated. Robot/table/floor, hand/arm/target, wrong-object finger contact, target/reference contact and penetration >3 mm remain failures. No checks are globally disabled. The phrase **no forbidden contacts observed** allows legitimate grasp/support contact.

The same world, backend, controller, pickup executor, gripper and viewer persist across operations. Full MuJoCo integration snapshots plus robot/object position/velocity/time records at operation end exactly match the next operation's start. No home reset, scene reload, object reset or snapshot restore occurs inside the plan. Safe retreat/home-like motions are executed through actuators when needed.

Default recovery budget is one retry per operation before release. Recoverable pickup tolerance/grasp-verification failures physically reopen/retreat, regenerate from current geometry and retry. Pre-place convergence can repeat its bounded pose command if the retry budget remains. Collision, drop, invalid destination, exhausted retry and any post-release verification failure stop the plan. Later operations do not execute. No post-release autonomous regrasp recovery is attempted.

`summary.json` and `events.jsonl` live in unique ignored `robotics_runs/*_pick_place_*/` folders. Pickup logs are referenced/nested inside each plan's unique directory. Summaries include source/end state, every pickup attempt, measured transforms, destination/direction, waypoints, support, release, retreat, settling, relation, retries, timings, final pose and failure diagnostics. Contact accounting counts executed substeps and intended **contact observations per substep**, not distinct contact episodes. Minimum transport table clearance covers free transport; drift maxima include free carry/pre-place/descent until support. Failed diagnostics are retained separately and never overwrite old results.

## Honest limits

Known upright cube/cylinder, deterministic grasp/destination geometry, calibrated simple table scene, waypoint transport, bounded recovery and finite contact retention only. Cylinder transport has appreciable axial rotation and positional slip; measurements are reported, not erased. No general collision planner, clutter reasoning, force-control placement, unrestricted language, learned grasping, Qwen manipulation, stacking or real hardware is included. Contact parameters affect success and simulation results establish no hardware safety guarantee. V3's upward visual-policy bias, alpha=0.5/c_puct=0.05 and MCTS-recovery findings remain untouched.

## Manual acceptance

The user manually validated cube absolute placement, upright cylinder next-to-cube placement and the continuous two-operation rearrangement. Physical grasp, transport, support, release, retreat and stable final placement were observed; Panda continued from current state without robot/scene/object reset. No teleportation, fake attachment or obviously artificial motion was observed. Recorded results, contact-sensitive cylinder drift and earlier diagnostic failures remain in TEST_REPORT.md; acceptance does not imply real-hardware safety.

## Phase 3 integration

[PHYSICAL_AI_AGENT.md](PHYSICAL_AI_AGENT.md) adds a bounded language task interpreter, semantic observation and physically simulated placement-side selection around this frozen skill implementation. Placement/retention/contact/settling thresholds and historical measurements remain unchanged. Actual selected operations use the same measured transform, physical release, independent placement/relation checks and bounded recovery. Speculative MuJoCo branches have separate logs and restore state; the executed multi-operation episode does not reset.
