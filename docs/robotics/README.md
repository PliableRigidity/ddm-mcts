# Robotics toolkit usage

V3 Phase 3 adds [ordered physical-AI tasks](MULTI_STEP_AGENT.md), using `PhysicalAgent.run_task` and `--task multi-semantic-reach --instruction 'Approach the cylinder, then the sphere, then the cube.'`. One scene, robot, camera, controller, viewer and loaded Qwen session persist across goals. Existing single-target commands below remain valid. See the guide for exact perception/visual-policy commands, physical verification, trace and bounded language grammar.

Phase 2 adds headless physical planning while retaining the historical experiments. Python 3.11+ is required. Install optional dependencies with:

```bash
python -m pip install -e '.[dev,robotics]'
```

Core imports and generic tests do not require MuJoCo. Uniform planning does not require Laya, Mica, network access, or a viewer.

## External Panda model

Use an existing MuJoCo Menagerie checkout or clone it outside this repository:

```bash
git clone https://github.com/google-deepmind/mujoco_menagerie.git /path/to/mujoco_menagerie
git -C /path/to/mujoco_menagerie checkout c96a32d28fb5da84da38c1da4d749e7a13212855
export PANDA_MODEL=/path/to/mujoco_menagerie/franka_emika_panda/scene.xml
```

This revision was validated locally. Keep its assets alongside its XML and follow Menagerie's model licensing. No playground code or model assets are copied into this repository. The factory uses the inspected `hand` body origin as the end effector, not a hypothetical tool-tip site. It uses joints 1–7, actuators 1–7, and the `home` keyframe. Gravity compensation is enabled for this example.

## Headless reach

```bash
python -m ddm_mcts.robotics.cli --target 0.5945 0.02 0.6245
python -m ddm_mcts.robotics.cli --target 0.5945 -0.02 0.6245 --planar
python -m examples.robotics.panda_reach --target 0.5145 0 0.6445
```

`ddm-robotics` is also installed as a console command. Configure `--simulations`, `--horizon`, `--seed`, `--increment`, `--epsilon`, `--max-steps`, and `--physics-steps`. `--planar` is the second example configuration, using only X/Y actions with the same robot, task, controller, and planner. Exit code 0 means success and 1 means the execution limit was reached. Headless mode performs no viewer initialization or GUI synchronization; optional viewer mode is documented below.

## Python API

```python
from ddm_mcts.robotics import ReachTask, RoboticsPlanner, panda_reach
from ddm_mcts.search.mcts import MCTSConfig
from ddm_mcts.robotics.run import run_episode

world = panda_reach(model_path, ReachTask((0.5945, 0.02, 0.6245)))
planner = RoboticsPlanner(world, config=MCTSConfig(60, seed=0), horizon=3)
result = planner.plan()       # live world preserved
world.step(result.action)    # explicit live execution
folder = run_episode(world, planner)  # continue, logging each decision
```

`world.snapshot()` / `world.restore(snapshot)` preserve physics and action count. `reset()` restores the model's home keyframe. `world.task.evaluate(world.get_state())` provides the objective; `is_success` and `is_terminal` distinguish successful and exhausted episodes.

## Custom environments, tasks, actions, and policies

Subclass `RoboticsEnvironment` in `ddm_mcts/robotics/core.py`. Supply observation, action enumeration, execution, snapshot, restore, and reset. Include every mutable controller/task runtime variable and external RNG in snapshots. Keep observations and snapshots immutable/hashable for policy caching. Provide a `Task` with evaluate, is_success, and is_terminal; expose goal/context in observations. A minimal simulator-independent example is `tests/robotics/test_core.py`.

For reaching, reuse `ReachTask`, `cartesian_actions(increment, axes)`, `ControlledRobot`, and a replacement controller. A different task can implement the same methods without changing MCTS. A different robot can configure `CartesianController` with body/joint/actuator names; it must use scalar joints and position controls. The generic environment contract does not require Cartesian actions or a reach observation.

Pass any existing `Policy` to `RoboticsPlanner`. It receives `SearchState`; use `state.observation` to access robot state and goal. UniformPolicy is the default. Existing `MicaPolicy` and `TextLayaPolicy` accept a state renderer and objective. The CLI supports `--policy uniform|laya|mica`; Laya needs `.[laya]` and local weights, while Mica needs its existing service. Model calls are unnecessary for normal tests. `MixedPolicy` and `PermutationAveragedPolicy` may wrap these policies. A predictor injected into TextLayaPolicy tests the real option-probability adapter offline.

## Logs and tests

Each run creates `robotics_runs/<UTC timestamp>_<unique ID>/` with configuration/initial state, per-action `steps.jsonl`, `summary.json`, and `SUMMARY.md`. Physics/action/search configuration, goal, seed, evaluations, chosen actions, search visits, planning time, policy diagnostics, and execution errors are retained. Configure `--output` for another robotics location; the repository's historical `results` directory is rejected. Runs never reuse a directory. Runtime artifacts are ignored by git.

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest
PANDA_MODEL=/path/to/scene.xml PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/robotics -ra
python -m ruff check .
```

The small MuJoCo XML tests require only the optional dependency. Panda integrations skip with an explicit reason when PANDA_MODEL is absent. There are no network/model/GUI dependencies in tests. The full suite includes all historical tests. See TEST_REPORT.md for recorded validation and ROBOTICS_CHANGELOG.md for decisions and limitations.

## Troubleshooting

Use Python 3.11+; the system WSL interpreter may be older. ROS can inject unrelated pytest plugins through PYTHONPATH; disable plugin autoload for this suite. Missing MuJoCo requires the robotics extra. Missing meshes means XML has been separated from its assets. Missing hand/joint/actuator/home names means the model differs from the validated Menagerie revision. Start with small nearby reachable targets. Orientation is unconstrained, targets may be unreachable, and discrete increments impose a precision floor. Adjust epsilon/increment together; more search cannot repair a physically unreachable goal. A planar action set cannot intentionally change height. The toolkit uses one world sequentially and does not support concurrent access to that instance.

## Visual inspection / viewer mode

Run the actual Phase 2 planner with a visible MuJoCo viewer from the repository root:

```bash
.venv/bin/python -m ddm_mcts.robotics.cli \
  --model /home/ishaan/robot-arm-playground/mujoco_menagerie/franka_emika_panda/scene.xml \
  --target 0.5945 0.02 0.6245 \
  --viewer
```

The observer uses the existing ControlledRobot, snapshot-backed MCTS, IK controller, and physics backend. **MCTS search branches are NOT visualized. Only selected actions are displayed.** The robot holds its last live pose while search runs; the terminal prints `Planning...`, then the chosen action and planning time. Execution is shown at approximately real simulation time, with display updates near 60 Hz. `--viewer-speed 0.5` makes execution slower, while `--viewer-speed 2` speeds it up. Search has no real-time pacing.

A small orange-red sphere marks the actual task target. This is visualization-only geometry in the viewer's user scene; it adds no physical body, collision, mass, or force. Both the model and data given to the viewer are display copies, so GUI edits cannot change the planner's physical model or state. No Panda XML is edited. Normal mouse camera rotation, panning, and scrolling/zoom remain available. Tab/Shift+Tab toggle viewer panels. Viewer simulation controls do not control the autonomous planner; this mode is an observer.

The terminal shows the target, policy, budget, horizon, action increment, each decision, and final error/action count/total planning time. Defaults remain UniformPolicy, 60 simulations, horizon 3, 2 cm increments, and 12 mm success tolerance. Structured logs still go to unique directories under `robotics_runs/` and now include final state and total planning time.

After success or the step limit, physics stops and the final pose stays visible. Close the window normally to exit; Ctrl+C in the terminal also closes the viewer. Closing early stops execution and retains an error summary for the interrupted run. The process waits for the viewer's render thread to finish on shutdown, avoiding a native-resource teardown race observed under WSL.

Headless mode remains the default: omit `--viewer`. No GUI imports, display setup, or pacing are needed for headless planning. Viewer mode needs a working graphical display (WSLg/X11); on macOS MuJoCo requires its `mjpython` launcher for passive viewing. See the [official passive viewer API](https://mujoco.readthedocs.io/en/stable/python.html#passive-viewer). Automated viewer integration tests use fake handles and never open a window.

## Phase 3: perception-aware physical AI

The toolkit now supports explicit ground-truth observations and real MuJoCo camera observations, deterministic calibrated color perception, structured semantic goals, and a closed-loop PhysicalAgent using the same planner/controller. The [physical-AI guide](PHYSICAL_AI.md) explains the API, camera geometry, logs, model boundaries, and full demo commands.

```bash
.venv/bin/python -m ddm_mcts.robotics.cli \
  --model /home/ishaan/robot-arm-playground/mujoco_menagerie/franka_emika_panda/scene.xml \
  --task visual-reach --goal red --observation camera --perception color --viewer
```

Change only `--goal blue` to select the other target. Omit `--viewer` for headless operation; set `MUJOCO_GL=egl` before starting Python when an offscreen backend is needed. Camera resolution/name are configurable, `--diagnostics` compares estimated positions against simulation truth, and `--save-images` optionally retains one RGB frame per live observation. Targets are genuinely detected from pixels using camera calibration and a known plane. Robot proprioception and dynamics initialization explicitly remain simulator-native. Model-based perception and direct visual decision interfaces are optional, mock-tested boundaries; no real VLM/VLDM is installed or required.

The original Phase 2 reach command and headless/viewer defaults remain supported. Both phases retain unique logs under `robotics_runs/`; camera observations and viewer updates never run inside speculative search.

## Real local VLM perception — V3 Phase 1

The optional `vlm` extra adds Qwen3-VL-4B-Instruct to the existing pipeline. The VLM identifies/grounds a requested shape; deterministic geometry localizes it; the original policy/MCTS/IK performs physical reaching. Cube, sphere and cylinder share one color, so semantic selection comes from the image/model. Ground-truth and red/blue color commands above remain unchanged.

See [LOCAL_VLM.md](LOCAL_VLM.md) for installation, GPU/model cache, standalone sanity check, cylinder/cube/sphere viewer commands, refresh/debugging options, logs and limitations. The default backend loads one local model per backend; normal tests use fakes and never download weights. V3 work remains uncommitted on `v3-phase1-vlm` pending manual inspection.

V3 Phase 2 optionally adds a [VLM-backed visual decision policy](VISUAL_DECISION.md) via `--decision-policy visual`. Perception remains separate; fresh-image priors guide only the MCTS root, with the existing configured policy at deeper nodes. The safe approach waypoint remains above the object.

## V4 Phase 1: physical pickup

[Manipulation guide](MANIPULATION.md) adds optional six-DoF TCP IK, physical Panda finger control, structured contacts and deterministic cube/cylinder grasp-and-lift. Use `--task pickup --goal cube` or `--goal cylinder`, with optional `--viewer --diagnostics`. This uses a separate dynamic scene; all existing reach/perception/visual-policy/multi-step commands retain their behavior. Qwen is not required for pickup. No placement is implemented.

## V4 Phase 2: pick-and-place and continuous rearrangement

[PICK_AND_PLACE.md](PICK_AND_PLACE.md) documents `--task pick-place --goal cube --place-position 0.60 -0.10 0.37`, `--task pick-place --goal cylinder --place-next-to cube`, and `--task rearrange-demo`. Each supports the existing viewer/diagnostics flags. Measured TCP/object transforms, physical contact release, independent placement/relation verification and a bounded retry extend frozen pickup. Rearrangement keeps one Panda, scene and viewer throughout; Qwen is not used by these deterministic operations.

## V4 Phase 3: language-conditioned closed-loop manipulation

[PHYSICAL_AI_AGENT.md](PHYSICAL_AI_AGENT.md) documents `--task manipulate --instruction 'Pick up the cylinder and place it next to the cube.'`, optional `--perception vlm`, and continuous `Move the cylinder next to the cube, then move the cube to the target location.`. The composed API supports fresh observations, semantic binding, actual MuJoCo skill-future MCTS choices, frozen V4 execution and physically verified progress. Existing reach, pickup, pick-place and rearrangement modes are unchanged.

## V5 goal composition

[COMPOSITIONAL_PHYSICAL_REASONING.md](COMPOSITIONAL_PHYSICAL_REASONING.md) documents the additive `physical-reason` mode, physical predicates, support surfaces, simulation-time wait, Panda pushes and genuine MuJoCo-backed MCTS. Frozen V4 `manipulate` remains unchanged.
