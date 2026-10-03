# V3 Phase 3: ordered physical-AI tasks

Final acceptance: the user manually validated both continuous cylinder → sphere → cube viewer modes (semantic perception and visual policy), including ordered completion, persistent Panda state, continuous safe transitions and the TCP waypoint marker. The finalization suite passed all 154 tests. See the final checkpoint in [TEST_REPORT.md](TEST_REPORT.md).

`PhysicalAgent.run_task` coordinates an ordered task using the existing observation, perception, policy, MCTS, MuJoCo and controller components. It is a new physical capability: the Panda visits multiple objects in order **without resetting between goals**. Single-goal commands and APIs remain supported.

The task remains safe **approach reaching**. The hand-local 103.4 mm `panda_gripper_tcp` offset, configured 40 mm object-radius allowance, 50 mm standoff, +Z approach and lift-before-translate logic are unchanged. The red viewer marker is the current TCP waypoint, normally above the object. It is not a target-identity hint sent to Qwen. No grasping or finger closure occurs.

## Commands

From the repository root, using the existing virtual environment and cached model:

```bash
export PANDA_MODEL=/home/ishaan/robot-arm-playground/mujoco_menagerie/franka_emika_panda/scene.xml
```

Phase 1 semantic perception, uniform MCTS, full-task viewer:

```bash
HF_HUB_OFFLINE=1 .venv/bin/python -m ddm_mcts.robotics.cli \
  --model "$PANDA_MODEL" --task multi-semantic-reach \
  --instruction 'Approach the cylinder, then the sphere, then the cube.' \
  --perception vlm --viewer --diagnostics
```

Phase 2 visual root priors plus the same semantic perception, with one shared model:

```bash
HF_HUB_OFFLINE=1 .venv/bin/python -m ddm_mcts.robotics.cli \
  --model "$PANDA_MODEL" --task multi-semantic-reach \
  --instruction 'Approach the cylinder, then the sphere, then the cube.' \
  --decision-policy visual --viewer --diagnostics
```

Second order, headless:

```bash
HF_HUB_OFFLINE=1 MUJOCO_GL=egl .venv/bin/python -m ddm_mcts.robotics.cli \
  --model "$PANDA_MODEL" --task multi-semantic-reach \
  --instruction 'Go to the cube and then the cylinder.' --perception vlm --diagnostics
```

Existing Phase 1 and Phase 2 single-target viewer commands remain:

```bash
HF_HUB_OFFLINE=1 .venv/bin/python -m ddm_mcts.robotics.cli \
  --model "$PANDA_MODEL" --task semantic-reach \
  --goal 'Reach the cylindrical object.' --perception vlm --viewer --diagnostics

HF_HUB_OFFLINE=1 .venv/bin/python -m ddm_mcts.robotics.cli \
  --model "$PANDA_MODEL" --task semantic-reach \
  --goal 'Reach the cylindrical object.' --decision-policy visual --viewer --diagnostics
```

`--max-steps` is the action budget **per subgoal** in multi-step mode; defaults remain 30 actions, 60 simulations, horizon 3 and 2 cm Cartesian actions. The robot's cumulative action counter is not reset. `--epsilon`, `--standoff`, `--object-radius`, camera, policy, output and existing VLM options also apply. Use `--instruction`, not `--goal`, for the ordered task. Headless rendering uses EGL; leave `MUJOCO_GL` unset for viewer runs in the tested WSL display environment.

The viewer opens once and stays open across all subgoals. The robot remains stationary during model inference and search; only selected live executions animate. Terminal output shows `TASK`, `SUBGOAL i/N`, physically verified completion, and the final result. On final success/failure the scene remains open for inspection. Close normally or use terminal Ctrl+C. Intermediate subgoal completion does not hold or reopen the viewer.

## Bounded task language and Python API

`parse_physical_task` accepts an initial **Approach**, **Visit**, or **Go to**, followed by cube/cylinder/sphere names (including cylindrical/spherical object), separated by commas, `then`, `and`, or `and then`. Optional `the` and repeated approach verbs are supported. Examples:

- `Approach the cylinder, then the sphere, then the cube.`
- `Go to the cube and then the cylinder.`
- `Visit the sphere, cube, and cylinder.`

Duplicates are allowed, with distinct stable occurrence IDs. Revisiting an already-achieved waypoint may require zero actions, but it still requires a fresh observation and verification. Unknown objects, empty clauses, negation, disjunction and unsupported operations are rejected. This is a deterministic grammar, **not general natural-language planning**.

```python
from ddm_mcts.robotics import parse_physical_task
from examples.robotics.validate_multi_step import make_agent

task = parse_physical_task("Approach the cylinder, then the sphere, then the cube.")
with make_agent(model_path, visual=False) as agent:
    result = agent.run_task(task)
print(result.success, result.folder)
```

The example's `make_agent` shows explicit component construction. The reusable library boundary is `PhysicalAgent(environment, observer, perception, task, planner, ...)`, followed by `run_task`. Ground-truth observation/perception can be substituted directly, using semantic scene entities as an explicit baseline. Uniform or existing structured DDM policies remain independent choices. `OrderedTaskRunner` accepts an injected verifier; the current default is `ApproachVerifier`. A custom backend/task must supply equivalent reaching and verification capabilities; this small representation is not a universal workflow or robot-state ontology.

## Task lifecycle, verification and information boundaries

`PhysicalTask` contains immutable `ApproachGoal` definitions with stable IDs, semantic targets, tolerance, action budget and extent/standoff criteria. The trace tracks each occurrence as pending, active, succeeded or failed. Failure stops the task; later goals stay pending and earlier successful results are retained. There is no silent skip or infinite retry.

For each subgoal, the runner clears goal-dependent perception history, changes the task with `set_task` (not `reset`), obtains a fresh observation, and invokes the existing single-goal receding-horizon run loop. After the final observation, verification checks:

1. The resolved semantic entity matches the requested subgoal.
2. The actual simulator TCP is within the same tolerance of the **final** outside-object waypoint, not an intermediate vertical-lift waypoint.
3. Where ground-truth entities are exposed, the TCP is also within that tolerance of the requested object's true approach waypoint. A mislabeled/wrongly localized image cannot pass solely by reaching its own incorrect estimate.
4. Where geometry diagnostics exist, signed gripper distances to **all** scene objects are positive at the final pose and throughout selected execution substeps.

Ground truth in checks 3/4 is explicit evaluation, never an input to inference or a correction of the estimated target. If geometry/true-entity capabilities are absent, the respective diagnostic is unavailable, not manufactured. This generic fallback does not establish physical collision safety. The Panda semantic scene exposes both capabilities. A substep clearance violation halts the current execution and preserves the failed trace.

Qwen semantic perception receives RGB plus the active semantic name. It grounds that object; the existing common-material mask and calibrated center plane localize it. The visual decision adapter separately receives RGB plus that goal and stable candidate action descriptions, with generic calibration/current-TCP context. It never receives target XYZ, simulator body IDs, true verification coordinates, or a correct action. Its validated priors guide only the MCTS root. Deeper nodes use the configured structured policy, uniform by default. MuJoCo remains the world model; MCTS remains the final action selector.

## What persists and what refreshes

Robot joints, velocities, actuator state, integration time, cumulative action count, scene, controller, camera, planner object and Qwen backend persist across goals. Full integration snapshots are checked before/after goal preparation, and each previous end state is recorded alongside the next start state. Nothing calls the Panda/scene reset between goals.

MCTS creates a new tree on **every decision**, including after goal changes: old objective statistics do not carry over. Its configured RNG stream persists; the seed is recorded. Perception reset clears semantic grounding/tracking while preserving the loaded backend and cumulative metrics. Fresh camera observations rebind the visual policy and invalidate old-image/old-goal priors. The same Qwen backend is shared by semantic perception and visual scoring; one model load per new backend is expected, not one per goal.

The existing +Z waypoint staging lifts before lateral motion when below the next safe approach height. The demonstration objects share a height plane, so transitions between already-cleared approach poses travel above the objects. Continuous gripper clearance is verified against all three geometries. This reuses the established approach strategy; it does not add arbitrary-scene collision planning or constrain robot orientation.

## Trace and accounting

Each unique run creates `robotics_runs/<timestamp>_task_<id>/task_trace.json`. Per-subgoal original run logs live below `subgoals/goal-N/<run-id>/`; they retain observations, selected actions, root statistics, raw Qwen outputs and summaries. The compact task trace references those logs rather than duplicating every response/tree statistic.

Trace fields include instruction/order/status, component configuration, search settings, start/prepared/end physical states, preserved-snapshot checks, first/final observation sequence, action and Qwen top-choice sequences, MCTS disagreements, perceived/true waypoint errors, final all-object distances, minimum transit distance, monitored substeps, per-goal timings, failure reason and overall result. Counters distinguish logical perception/visual requests, cache hits/misses, physical semantic/visual calls, inference time, model load time/count, observation/perception time, MCTS search time and controller execution. Per-subgoal counters are differences of cumulative session counters. Shared-session totals also include any caller activity before the task. Inference time excludes load time; task wall time includes orchestration/logging overhead.

Normal tests never load/download Qwen or open a GUI. Explicit real acceptance utility:

```bash
HF_HUB_OFFLINE=1 MUJOCO_GL=egl .venv/bin/python -m examples.robotics.validate_multi_step --model "$PANDA_MODEL"
```

Add `--visual` for visual root priors; add `--viewer` **without EGL** for a bounded smoke that closes the one viewer automatically after execution. Use the CLI for manual final-scene inspection.

## Limits retained from previous phases

The scene and object set are simple; the parser language is bounded. VLM localization uses a known center-height plane and configured object extent. The controller is primarily position-based and safe approach is not grasping. There is no general collision planner, uncertainty-aware safety system, manipulation, real hardware support or model training. Simulation success does not establish real-hardware safety.

Qwen's visual scores still show a persistent upward-motion bias. Phase 2's failed `c_puct=1.4` trials remain documented and untouched. Visual multi-step mode retains **alpha=0.5, c_puct=0.05**. Successful lateral travel is recovered by MCTS despite those priors; it does **not** demonstrate that Qwen improves search. The automated bad-prior override proof remains in the unchanged Phase 2 tests. See the [test report](TEST_REPORT.md) for actual multi-step measurements and the [visual-policy guide](VISUAL_DECISION.md) for earlier results.
