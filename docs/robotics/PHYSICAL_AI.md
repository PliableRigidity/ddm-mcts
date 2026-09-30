# Perception-aware physical planning

Phase 3 adds observations, perception, semantic goal resolution, and a closed-loop agent to the existing Phase 2 toolkit. It uses the same MCTS, snapshot-backed simulator, Cartesian actions, IK controller, execution loop, and viewer. No model, camera hardware, or network service is required.

## Run the demonstrations

From the repository root, set the installed external Menagerie model:

```bash
export PANDA_MODEL=/home/ishaan/robot-arm-playground/mujoco_menagerie/franka_emika_panda/scene.xml
```

A. Existing Phase 2 reaching, with its original direct simulator state and viewer:

```bash
.venv/bin/python -m ddm_mcts.robotics.cli \
  --model "$PANDA_MODEL" --target 0.5945 0.02 0.6245 --viewer
```

B. Phase 3 camera/color perception, reach red:

```bash
.venv/bin/python -m ddm_mcts.robotics.cli \
  --model "$PANDA_MODEL" --task visual-reach --goal red \
  --observation camera --perception color --viewer --diagnostics
```

C. Same scene, planner, and perception, change only the semantic goal:

```bash
.venv/bin/python -m ddm_mcts.robotics.cli \
  --model "$PANDA_MODEL" --task visual-reach --goal blue \
  --observation camera --perception color --viewer --diagnostics
```

D. Headless Phase 3 with EGL and no viewer:

```bash
MUJOCO_GL=egl .venv/bin/python -m ddm_mcts.robotics.cli \
  --model "$PANDA_MODEL" --task visual-reach --goal red \
  --observation camera --perception color --diagnostics
```

`--task visual-reach` defaults to camera/color. `--goal` defaults to red. Do not pass XYZ `--target` for the semantic task. For an explicit privileged observation/perception baseline:

```bash
.venv/bin/python -m ddm_mcts.robotics.cli \
  --model "$PANDA_MODEL" --task visual-reach --goal blue \
  --observation ground-truth --perception ground-truth
```

The original XYZ task also accepts `--observation ground-truth --perception ground-truth` to route through the explicit observation path. Omitting observation options preserves the Phase 2 direct-state path. Existing policy/search/action options still work: `--policy uniform|laya|mica`, `--simulations`, `--horizon`, `--seed`, `--increment`, `--epsilon`, `--max-steps`, `--physics-steps`, and `--planar`.

Use `--camera inspection`, `--width 640 --height 480`, `--save-images`, `--output <robotics-directory>`, and `--viewer-speed 0.5` as needed. The camera is an offscreen sensor, independent of the viewer's freely movable camera. Images are not saved by default. `--diagnostics` enables true simulator state/position-error logging, strictly after perception.

## What the visual task does

The example adds static red and blue disks to an in-memory Menagerie model, plus a named perspective camera. Disk centers lie on the calibrated horizontal plane z=0.6245 m. Targets default to red (0.6045, 0.08, 0.6245) and blue (0.5045, -0.08, 0.6245); those coordinates configure the scene, not the visual perception algorithm. The disks are world geoms with contype=conaffinity=0 and no robot mass contribution. They do not collide with the robot. Panda XML and assets on disk are unchanged.

The camera renders both robot and targets. Color perception finds connected colored regions in the actual image, rejects tiny regions and comparable ambiguous components, computes their centroids, and intersects calibrated camera rays with the known target plane. `--goal red` selects the entity labeled red; `--goal blue` selects blue. A resolved estimated target becomes the reach objective. MCTS explores physical futures from the current simulator snapshot and selects a Cartesian action. The original controller applies it through actuator commands and physics. The agent takes a new image and resolves the goal again after every action, including the final success check.

Success is within the configured epsilon of the currently estimated target; max_steps is failure termination. With diagnostics, target-estimation errors can be compared to the true scene positions. Simulation truth never corrects perception output.

## Developer API

```python
from ddm_mcts.robotics import (
    PhysicalAgent, RoboticsPlanner, SemanticGoal, SemanticReachTask,
)
from ddm_mcts.robotics.camera import MujocoCameraObservationProvider
from ddm_mcts.robotics.color_perception import ColorPlanePerception
from ddm_mcts.robotics.visual_scene import panda_visual_reach, TARGET_PLANE_Z
from ddm_mcts.policies.random_policy import UniformPolicy
from ddm_mcts.search.mcts import MCTSConfig

world = panda_visual_reach(model_path)
observer = MujocoCameraObservationProvider(world, camera="inspection", width=320, height=240)
perception = ColorPlanePerception(TARGET_PLANE_Z)
task = SemanticReachTask(SemanticGoal("red"), epsilon=0.012, max_steps=30)
planner = RoboticsPlanner(world, UniformPolicy(), MCTSConfig(60, seed=0), horizon=3)

with PhysicalAgent(world, observer, perception, task, planner,
                   diagnostics=world.target_entities) as agent:
    red_run = agent.run()
    agent.reset(SemanticReachTask(SemanticGoal("blue")))
    blue_run = agent.run()  # Same components and planner; unique output directory.
```

`agent.prepare()` acquires and perceives a live observation and binds its goal; `agent.plan()` delegates to the configured RoboticsPlanner. The ordinary run loop handles execution and re-observation. Manual component use is also possible: `observation = observer.observe()`, `state = perception.perceive(observation, goal)`. Use `agent.set_task(task)` to change goals and invalidate a prepared decision context. `agent.reset(task=None)` resets robot and perception history. A finished run is detached from the agent, so later manual observation/plan calls cannot append to its files. Close the agent/provider when done; camera GL resources are owned by the creating thread.

To combine the facade with visualization programmatically, call `agent.prepare()` before opening `VisualInspection(world, planner)`, then `agent.run(visualization=visual)` and `visual.wait_until_closed()`. Release camera resources before closing the viewer when both share a GLFW backend; the CLI handles this ordering.

## Component boundaries and extension

| Component | Responsibility |
|---|---|
| ObservationProvider | Acquire timestamped raw input, with optional RGB and calibrated camera metadata |
| GroundTruthObservationProvider | Explicit privileged simulator/proprioception/entities input |
| MujocoCameraObservationProvider | Read-only uint8 RGB plus simulator robot proprioception, without target positions |
| PerceptionProvider | Convert input and semantic goal into a structured WorldState |
| GroundTruthPerception | Identity-style structured baseline and goal resolution |
| ColorPlanePerception | Image segmentation and calibrated known-plane target localization |
| WorldState | Robot state, labeled entity estimates, semantic goal, source, visibility/staleness |
| SemanticReachTask | Resolve a semantic entity into the existing ReachTask objective |
| RoboticsPlanner / SimulatorAdapter | Reuse original MCTS and branch safely over the world model |
| PhysicalAgent | Coordinate observe/perceive/resolve/plan; reuse execution/logging |
| Controller | Existing bounded IK and actuator execution, owned by ControlledRobot |

An observation provider can be replaced without MCTS changes. A perception provider implements `perceive(observation, goal)`; provide a prediction projection for its structured representation. A task resolver implements `resolve(state)` and supplies the task's goal/context. A simulator implements the existing robotics contracts. `WorldState`/`SemanticReachTask` are small reaching helpers, not a universal ontology required by search. Other robot/task representations may supply their own projection and resolver while reusing the same core planner.

## Perceived context versus world-model state

This is simulation-based physical AI, not fully observed-state reconstruction from pixels. The RGB observation includes explicit robot proprioception from the simulator. Target positions are estimated from RGB. MCTS begins from the privileged live robot dynamics snapshot (joint positions/velocities, controls, etc.). The structured root contains perceived target/entity context. Future nodes use `WorldState.predict_observation` to predict robot motion from physics while holding perceived target estimates fixed. This is valid for the static-target demo. No speculative cameras run, and no image is treated as something MuJoCo can step.

A future robot/learned world model must provide a suitable dynamics-state initializer or belief-state representation. A perception estimate is not automatically a physically complete simulator state. Calibration or dynamics mismatch can cause confident planning errors even when image detections are correct.

## Camera geometry and perception assumptions

RGB is HxWx3 uint8, RGB channel order, top-left image origin. Integer (u,v) denotes a pixel center; u increases right, v down. MuJoCo camera local axes are +X right, +Y up, -Z forward. The provider supports named perspective cameras parameterized by vertical fovy, square pixels, and a centered principal point; orthographic and sensor-size/intrinsic cameras are explicitly rejected.

With H=image height and θ=vertical fovy, f=H/(2 tan(θ/2)), cx=(W-1)/2, cy=(H-1)/2. Camera-to-world rotation R and optical center C are included with each observation. The pixel ray is `r = R [(u-cx)/f, -(v-cy)/f, -1]`. Intersection with z=z_plane is `p = C + ((z_plane-Cz)/rz) r`; parallel rays or intersections behind the camera fail explicitly. Projection applies R transpose to (p-C), then divides X and Y by -Z with the appropriate image-Y sign. See the [official camera frame description](https://mujoco.readthedocs.io/en/stable/programming/visualization.html).

Known plane height is explicit calibration, not a hidden per-target answer. RGB alone cannot recover arbitrary 3D depth. For other scenes, supply an appropriately calibrated plane or implement an RGB-D/multiview perception provider. Camera intrinsics/extrinsics must be accurate. The target surface is thin and planar; segmentation centroids represent its center. Distinct saturated matte/emissive target colors and adequate pixel area are assumed. Colors, chromaticity tolerance, pixel count, plane height, and missed-frame limit are configurable in the Python API.

Partial occlusion can shift a centroid. If the visible component is below 65% of its previous full area, the baseline retains the last estimate for at most six missed frames, tagged `observed=False` with a warning. This is an explicit static-target tracking assumption, not a recovered invisible position. Missing/expired selected targets fail before execution. Reset perception history between episodes. Moving targets and uncertain pose estimation need a different tracker/belief model; no such capability is claimed here.

## Optional models: VLM and VLDM are distinct

`ModelPerceptionAdapter(predictor)` accepts an injected callable `(Observation, SemanticGoal) -> WorldState`. A future VLM can parse image+goal into structured entities and robot state, then the existing DDM and MCTS consume that representation. Schema/goal validation is performed. There is no built-in model service, automatic download, or inference dependency.

`VisualDecisionModel.score(rgb, goal, actions) -> probabilities` instead describes a future direct image+language-goal+candidate-choice model. `VisualDecisionPolicy` adapts it to the original Policy interface. Pass it as `visual_policy=` to PhysicalAgent: it becomes MCTS's root_policy, while ordinary uniform/structured DDM priors handle future nodes. The live image is bound afresh for each decision. No predicted future images exist, so using this policy on non-root search states is rejected. This is a mock-tested interface, not a trained VLDM or claim of a working neural visual decision model.

Existing TextLayaPolicy/MicaPolicy work with `state.observation.position`, `.goal`, and `.entities`; the CLI's state renderer exposes hand and perceived goal coordinates. A custom renderer can include entity source/visibility information. UniformPolicy requires no service. Model inference remains optional.

## Viewer and logs

The viewer shows Panda, both scene targets, and a debug sphere at the selected perceived goal. The camera observation uses the physical visual scene, not the viewer's debug sphere. Terminal output lists perceived entities, observed/tracked status, semantic goal, and existing planning/execution progress. Search stays invisible; only selected actions are displayed. The final pose remains open for inspection until closure/Ctrl+C. Camera motion through the viewer does not change the observation camera calibration.

Each run creates `robotics_runs/<UTC>_<unique ID>/` with the existing config/steps/summary artifacts plus `observations.jsonl`. Configuration records providers, semantic task, camera size/name, perception thresholds, world-model initialization, search/policy/seed, image/diagnostics flags, and actions/controller. Each observation records source, simulation timestamp, sequence, estimated entities/state, calibration, visibility warnings, acquisition/perception time, and optional diagnostic true-state/entities/errors. Step records include planning and execution times, selected action, predicted/actual post-action state, and root statistics. Summary records final perception and total planning time. Physical inference diagnostics are retained where provided, including structured and direct visual priors.

`--save-images` writes one binary RGB PPM per live observation under `images/`. No rollout images are generated/saved. Diagnostics are opt-in and never injected into perception. Ground-truth/proprioception is explicitly present in its own mode; camera mode uses simulator robot telemetry by design. Errors preserve a summary and stop before executing an unresolved target. Direct per-component failures before a run starts may not create run artifacts.

## Tests and troubleshooting

```bash
MUJOCO_GL=egl PANDA_MODEL="$PANDA_MODEL" PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  .venv/bin/python -m pytest -ra
```

Camera integrations require a supported offscreen OpenGL backend, not graphical interaction. EGL worked with DISPLAY and WAYLAND_DISPLAY unset locally. Select MUJOCO_GL before Python imports MuJoCo; the toolkit does not change your environment globally. OSMesa is an alternative if installed. For viewer mode, WSLg/X11 must be available; the tested viewer can coexist with an EGL observation camera. On macOS use MuJoCo's mjpython launcher. Tests that need an unavailable external Panda model skip with a reason. Tiny-scene rendering tests skip when offscreen GL cannot initialize; perception/geometry and mock-model tests remain deterministic without network access.

A missing target often indicates occlusion, insufficient resolution, changed lighting/colors, or incorrect calibration. Inspect optional saved frames and observed/stale flags. No perception fallback silently substitutes true coordinates. Reach tolerance should account for action resolution and perception error. Check the test report for measured local errors; there is no claim of robust general vision, real-world calibration, collision safety, or broad-workspace reaching.

## Small acceptance audit

Run six real closed-loop episodes across three nearby target placements, using the same agent components for red then blue in each scene:

```bash
MUJOCO_GL=egl .venv/bin/python -m examples.robotics.validate_visual \
  --model "$PANDA_MODEL"
```

Each unique run gets `acceptance.json` containing goal, perceived and true positions, maximum selected-target perception error, true final reaching error, action/observation counts, and planning time. This is a development acceptance check, not a benchmark. It exercises both signs of X and Y displacement and varying nearby distances. Target positions configure scene generation only.

The audit also attempted targets hidden by the arm at startup and a farther target that remained occluded beyond six frames. These fail explicitly before another action; target visibility is a documented precondition, and bounded static tracking cannot solve persistent occlusion. The automated suite includes a poisoned privileged-target observation to prove that image perception ignores supplied true entity/task coordinates.
