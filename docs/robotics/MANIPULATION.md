# V4 Phase 1: physical Panda pickup

This optional manipulation extension picks up a known cube or cylinder through **actual MuJoCo contact physics**. It does not change V3 approach tasks: those still stop above static, non-colliding semantic markers. Pickup uses a separate dynamic scene and deterministic geometry. No Qwen call is needed for grasp transforms, finger width or joint commands. No placement is implemented.

## Commands

Install the existing `robotics` extra and provide the external Menagerie Panda model. No new dependency, model download or asset copy is required.

```bash
export PANDA_MODEL=/home/ishaan/robot-arm-playground/mujoco_menagerie/franka_emika_panda/scene.xml

# Cube pickup, fingers close physically, lift, hold; final viewer stays open.
.venv/bin/python -m ddm_mcts.robotics.cli \
  --model "$PANDA_MODEL" --task pickup --goal cube --viewer --diagnostics

# Cylinder pickup.
.venv/bin/python -m ddm_mcts.robotics.cli \
  --model "$PANDA_MODEL" --task pickup --goal cylinder --viewer --diagnostics

# Headless pickup (repeat with cylinder).
MUJOCO_GL=egl .venv/bin/python -m ddm_mcts.robotics.cli \
  --model "$PANDA_MODEL" --task pickup --goal cube --diagnostics

# Existing V3 approach behavior remains available.
HF_HUB_OFFLINE=1 .venv/bin/python -m ddm_mcts.robotics.cli \
  --model "$PANDA_MODEL" --task semantic-reach \
  --goal 'Reach the cylindrical object.' --perception vlm --viewer --diagnostics
```

`--lift-distance` defaults to 0.10 m (allowed 0.05–0.15 m). `--viewer-speed` controls execution pacing. Drag/scroll with normal MuJoCo camera controls. The red debug sphere indicates the active TCP waypoint. Close the viewer or press Ctrl+C after completion. The validation utility automatically closes its viewers:

```bash
.venv/bin/python -m examples.robotics.validate_pickup --model "$PANDA_MODEL" --trials 3
.venv/bin/python -m examples.robotics.validate_pickup --model "$PANDA_MODEL" --trials 1 --viewer --viewer-speed 4
```

## Python API and composition

```python
from ddm_mcts.robotics.manipulation_scene import panda_pickup_scene
from ddm_mcts.robotics.manipulation import ManipulationAgent, PickupTask

world = panda_pickup_scene(model_path)
agent = ManipulationAgent(world)
result = agent.run(PickupTask("cube", lift_distance=0.10), "robotics_runs")
assert result.success, result.trace.get("reason")
```

The scene exposes `ManipulationObject` (label, shape, full dimensions, actual pose). `ParallelJawGraspGenerator` produces a `GraspPose` with distinct object-centered grasp, pre-grasp and lift poses, opening allowance and approach direction. A generator can be injected for another calibrated shape; unsupported or oversized/tilted objects are rejected. `PoseController`, `PandaGripper`, `ContactInspector`, `ContactRules`, `verify_grasp` and `verify_lift` are usable independently. Existing `PhysicalAgent.run_task`, MCTS and single-goal APIs are unchanged. This pickup executor is an explicit deterministic physical primitive, **not another search planner** and not a claim that a VLM decides contact physics.

## Six-DoF TCP control

The controller continues to use `panda_gripper_tcp`, 103.4 mm along hand-local Z. Pose targets contain XYZ and a normalized **wxyz** quaternion mapping local axes into world coordinates. TCP local Z points downward for top-down grasps; local Y is the finger closing axis. Cube yaw follows the upright object's horizontal principal axes; cylinders use a radial pinch, yaw-invariant for the demonstrated upright geometry.

Pose IK uses the world-frame rotation-vector error `Log(R_target R_current.T)` rather than Euler subtraction. It stacks MuJoCo translational and rotational TCP Jacobians and solves weighted damped least squares:

`dq = Jw.T @ solve(Jw @ Jw.T + damping² I, ew)`

`Jw = W J`, `ew = W [position_error; rotation_error]`. Defaults: position weight 1, orientation weight 0.3, damping 0.02, per-joint update limit 0.06 rad, 100 scratch iterations. Joint and actuator bounds apply. Live `qpos` is never written by the solver: scratch kinematics produce actuator commands, followed by real physics. Position-only `CartesianController.execute` is unchanged and inherited by the pose controller.

Trajectories use 4 mm position and 0.04 rad orientation increments with quaternion SLERP, followed by bounded settling. The pickup factory defaults to 25 physics steps per block at Menagerie's 2 ms timestep; its Python `physics_steps` argument can configure that block size. Verification requires actual position error ≤2 mm and orientation error ≤0.025 rad. Unreachable poses fail verification rather than being treated as completed. These are demonstration tolerances, not a general feasibility or collision planner.

## Physical fingers and contacts

Menagerie has `hand`, `left_finger`, `right_finger`, slide joints `finger_joint1/2` in [0,0.04] m, and `actuator8` on the fixed `split` tendon. The original equality couples the finger joints; **no object weld is added**. Width is the sum of joint travels (nominal 0–80 mm); meshes/pads can affect the precise surface gap. `set_width`, `open`, `close` write actuator control in [0,255] and advance physics. State reports commanded/actual width, each joint and motion. Closure against an object leaves nonzero measured width despite a zero-width command.

Contacts report geom/body names, signed contact distance and normal force. Bilateral contact requires positive force (>0.01 N), not just overlapping broad-phase geometry. World-model snapshots include the new free bodies, finger motion and actuator state.

## Scene configuration and diagnosed tuning

The separate scene adds a table with top Z=0.35 m, a 40 mm cube at XY=(0.50,-0.08), and an upright radius-20 mm, height-50 mm cylinder at XY=(0.50,0.08). Each object has mass 0.08 kg, a free joint, gravity, collision and friction. Initialization extends the home keyframe with free-body poses and then settles physically. No resets occur inside a pickup.

Default soft contact/actuation allowed cylinder slip and eventual drop. Pickup-only configuration uses pad/object `solref=(0.005,1)` instead of the usual 20 ms response, finger servo stiffness 400 N/m instead of 100, damping 20 N s/m instead of 10, with the gain scaled to preserve the original width mapping. Original force limits and friction `(1,0.005,0.0001)` remain unchanged. Dynamic objects use `condim=4`. A compliance-only trial still slipped during a 15 s hold; the increased finger stiffness kept that longer diagnostic hold within the positional drift limit. These are explicit task-scene changes, not sticky objects or modifications to external Menagerie assets. Cube/cylinder acceptance uses the specified one-second hold; it does not certify indefinite retention.

## Pickup stages and verification

1. Select the explicit known object and generate a shape-aware grasp.
2. Lift/retreat vertically, translate and orient at a safe transit height, descend to pre-grasp (100 mm above object center).
3. Open fingers, verify TCP pose, sufficient opening and ≥20 mm gripper/target clearance.
4. Descend along the controlled approach axis to the grasp pose and verify actual pose.
5. Close physically; require bilateral force-bearing finger contact, nonzero held width and object near TCP.
6. Lift 100 mm while retaining orientation, verify actual TCP pose, then hold and recheck every physics block.
7. Require object height gain ≥75% of commanded lift, relative positional drift <15 mm, object orientation drift <0.35 rad, ongoing bilateral contact and no table support. Only then return success.

On any failure the sequence stops, records the failure stage/reason and preserves actual physical state. It does not retry infinitely or pretend that finger closure alone is pickup success. Deliberately offset grasps, bad pre-grasp poses, forbidden contact and dropped-object tests exercise failure paths.

Before grasp, finger-target contact is forbidden. During approach/closure/lift, intended target contact is allowed **only for fingers** with penetration ≤3 mm. Hand/arm-target, robot-table/floor and finger-other-object contacts remain forbidden. Object-table support is legitimate before lift. Monitoring runs on every executed physics substep, alongside viewer callbacks. It never welds, teleports, reparents, changes gravity or forces target velocities.

## Logging and limitations

Unique `robotics_runs/<timestamp>_pickup_<id>/` folders contain `summary.json` and `events.jsonl`: initial/final robot and object poses, grasp poses, stage transitions/timings, requested/actual trajectory poses and errors, gripper commands/widths, contact transitions, verification, hold checks, relative object/TCP transforms, penetration and failure reasons. The small validation utility writes a separate unique aggregate manifest. These files are ignored, not committed. Historical V3 logs remain untouched.

The supported set is a known upright cube/cylinder in a simple calibrated simulation scene. No learned grasp detection, Qwen geometry prediction, unrestricted manipulation language, placement, whole-arm collision planning, force controller, real hardware or hardware-safety claim is provided. Contact monitoring detects configured undesirable contacts; it is not a collision-free planning guarantee. Retention is tested for a bounded hold; cylinder contact remains more sensitive than box-face contact. V3's Qwen upward bias and MCTS-recovery findings remain unchanged.

## Subsequent V4 Phase 2 extension

The preceding sections preserve the Phase 1 pickup checkpoint and its historical measurements. [PICK_AND_PLACE.md](PICK_AND_PLACE.md) adds placement, relative geometry, measured transforms, continuous rearrangement and bounded recovery by composing the unchanged pickup executor. The Phase 1 command and API remain valid; no grasp/scene/controller/contact tuning was changed for Phase 2.
