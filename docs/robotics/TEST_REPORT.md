# Phase 2 validation report

Date: 2026-09-30 (Europe/London). Status: COMPLETE.

## Baseline and protection

Initial git status was clean. Baseline on Python 3.12.14: **41 passed in 0.74s**, no failures/skips. Baseline Ruff: all checks passed. The system Python is 3.10 and cannot import datetime.UTC; inherited ROS pytest plugins also interfered with collection. These were environment setup failures, not repository regressions. Validation disables automatic third-party plugin loading.

All original source, nine original test modules, scripts, and `results/.gitkeep` remain byte-identical to HEAD. Only existing README, pyproject.toml, and gitignore receive additive toolkit changes. No historical report/results directories were moved, written, or removed. The checkout has no historical numerical reports to reinterpret.

## Added coverage

`tests/robotics/test_core.py`: simulator-independent contract, task evaluation/success, finite search horizon, seeded uniform search, legal actions, reset, independent branching, invalid-action exception restoration, simulator-independent structured run logging.

`tests/robotics/test_mujoco.py`: optional backend initialization with tiny local XML, state capture including controls/applied forces, exact repeated restoration and same-action replay, distinct-action branching, owner validation, reset, actual Panda axis motion, control/displacement bounds, robot snapshot/action-count restore, three reachable 3D targets, planar alternate action set, uniform and mock-DDM planning, swappable policies, prior inspection, live-world preservation, headless execution, structured output contents, error logging, historical-output rejection, CLI success, task/action validation.

The mock DDM injects a deterministic offline predictor into the existing TextLayaPolicy adapter and scores candidate displacements toward the goal. No model downloads, service calls, GUI, or network access occur in tests. Python commands installed dependencies during setup only.

## Final gates

| Gate | Evidence | Result |
|---|---|---|
| 1 repository/baseline | clean status, 41 tests, baseline Ruff | pass |
| 2 generic contracts | dependency-free line-world tests | pass |
| 3 MuJoCo snapshots | exact full integration replay and branching | pass |
| 4 Panda controller | positive movement on X/Y/Z through physics, bounds | pass |
| 5 physical MCTS | original search, three uniform-policy targets | pass |
| 6 policies | offline Laya choice adapter and policy swapping | pass |
| 7 generality | planar X/Y action configuration reaches target | pass |
| 8 APIs/logs | JSON/JSONL summaries, failure record, output guard | pass |
| 9 documents | architecture, usage, changelog, report | pass |
| 10 regression | full suite and original tests independently | pass |

Final complete suite: **56 passed**, zero failures/skips with PANDA_MODEL set. Robotics: **15 passed** (12 MuJoCo/integration cases plus 3 generic cases). Historical V1 separately: **41 passed in 0.70s**. Without external model configured: robotics **5 passed, 10 skipped**, each Panda skip explains PANDA_MODEL requirement. Core-only fresh installation without MuJoCo: **44 passed, 1 module skipped**; generic contracts and logging remain usable. No real Laya/Mica service was available/required for validation.

Formatting applies only to new robotics/example/test files. `ruff check .` passes for the entire repository. Existing historical source formatting is preserved.

## Reproduce

```bash
export PANDA_MODEL=/path/to/mujoco_menagerie/franka_emika_panda/scene.xml
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -ra
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest tests/robotics -ra
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest tests --ignore=tests/robotics
.venv/bin/ruff check .
.venv/bin/ruff format --check ddm_mcts/robotics tests/robotics examples/robotics
```

Local validation: MuJoCo 3.14.0, NumPy 2.5.3, pytest 9.1.1; Menagerie revision `c96a32d28fb5da84da38c1da4d749e7a13212855`. External model path was `/home/ishaan/robot-arm-playground/mujoco_menagerie/franka_emika_panda/scene.xml`; no playground scripts were copied.

## Standalone run evidence

All four commands returned exit 0 / success true with 60 simulations, horizon 3, seed 0, 2 cm increments, epsilon 0.012 m:

| Configuration | Target XYZ | Local run directory under robotics_runs |
|---|---|---|
| 3D | (0.5945, 0.02, 0.6245) | 20260929T235555_61a20851c275 |
| 3D | (0.5145, 0, 0.6445) | 20260929T235556_d2df020e05e4 |
| example module, 3D | (0.5545, -0.04, 0.6045) | 20260929T235558_3f7a83d0729a |
| planar | (0.5945, -0.02, 0.6245) | 20260929T235559_609b20417f73 |

Run IDs use UTC; these dates are Sep 30 in the local timezone. Each contains config.json, steps.jsonl, summary.json, SUMMARY.md. Files are local runtime artifacts ignored by git. Snapshot reproducibility is asserted exactly, not merely by end-effector tolerance.

## Regressions and limits

No V1 regressions. Development failures exposed enum comparison and uncompensated servo sag; both fixed before declaring the physical gates complete. Reach tests cover nearby targets, not the whole workspace or collision safety. Position-only IK can terminate without convergence; the task checks actual simulated position. Integration snapshots do not capture arbitrary Python callbacks or external plugin resources. Simulator warnings are not currently collected into structured records; execution exceptions are. Tests run without a viewer.

Final integrity audit compared SHA-256 content of all 49 protected tracked files with HEAD: **zero changes**. Only README, packaging, and gitignore differ among pre-existing files.

Final implementation smoke rerun: `robotics_runs/20260929T235915_a17b1fd075cb`, target (0.5945, 0.02, 0.6245), uniform policy, exit 0/success true. Final logs additionally record IK configuration, physical substeps per action, action definitions, and compiled gravity-compensation body count. Final full suite: 56 passed in 8.76s; explicit robotics/integration run: 15 passed in 9.36s; core-only install: 44 passed/1 optional module skipped in 1.13s. Final Ruff, new-file formatting, and git diff whitespace checks pass.

## Phase 2 visual-mode validation — 2026-09-30

Status: COMPLETE. Baseline before viewer changes: existing 56-test suite passed. New tests: 8 in `tests/robotics/test_visual.py`, all GUI-free. Final complete suite **64 passed in 14.42s**; robotics suite **23 passed in 14.96s**; historical V1 suite **41 passed in 0.79s**. No failures/skips in runs with PANDA_MODEL configured. Ruff, new-code formatting, and git diff whitespace validation passed.

Verified:

- `--viewer` parsing and headless default; positive finite execution speed.
- Physics callback invoked once for each of 150 selected execution steps, then removed, including exception cleanup.
- Batched headless physics and observed single-step physics produce exactly equal integration snapshots.
- Speculative MCTS produces zero viewer synchronizations and preserves the live snapshot.
- Target marker matches task coordinates within float32 rendering precision; no physical geometry/model/state added.
- Display model/data are distinct; fake GUI changes to display controls/gravity cannot alter live physics.
- Complete observed and headless episodes have identical actions, observations, root priors/visits/Q, and final integration snapshot.
- Progress output and normal logging remain, final display refresh does not advance live physics.
- CLI routing through the observer, early viewer closure, rendering-thread shutdown, and Ctrl+C summary retention.

Real GUI smoke tests were performed on WSL DISPLAY=:0/Wayland support. The first run opened and successfully reached but segfaulted during asynchronous render teardown. This was diagnosed and fixed by waiting for render-thread exit after close. Subsequent direct-observer smoke completed with exit 0. Exact `python -m ddm_mcts.robotics.cli ... --viewer` smoke reached in 3 actions, held the final scene, then received controlled SIGINT; it printed clean interruption and returned expected exit 130 with no stderr. No orphan GUI/process remains. Initialization/rendering/execution were smoke-tested programmatically; visual appearance was not manually judged by a person.

Artifacts:

- Clean viewer smoke: `robotics_runs/20260930T000832_c2dd178c1acd`.
- Exact CLI + final-scene hold + Ctrl+C: `robotics_runs/20260930T000941_c53b3277af3b`.
- Final headless smoke: `robotics_runs/20260930T000940_dbfbb8714411`.

All successful runs used target (0.5945, 0.02, 0.6245), uniform policy, 60 simulations/horizon 3, and ended at approximately (0.5923, 0.0192, 0.6243), error 0.0023 m. Live trajectory was MOVE_X_POS, MOVE_X_POS, MOVE_Y_POS. Search pauses are visible as a stationary pose while terminal prints Planning; rollout branches are never published.

Final SHA-256 integrity audit against HEAD: **49 protected historical tracked files checked, zero differences**. No V1 results/reports/source/tests changed. Main README/packaging/gitignore retain their prior Phase 2 edits and were not edited by the viewer stage. No Phase 3 work was introduced.

## Phase 3 final audit and validation — 2026-09-30

Status: **COMPLETE**. Baseline: clean `main` at Phase 2 checkpoint `65009c4a934d19fc45637e41e61f1e9c64fcf3f9`; complete suite **64 passed in 11.99s** (41 V1, 23 Phase 2). Work finalized on `phase3-perception-toolkit`; main remains at that checkpoint. No merge or history rewrite is authorized/performed.

### Definition-of-done audit

Each PASS is supported by tests or real runtime validation, rather than code presence alone.

| Acceptance item | Status | Evidence |
|---|---|---|
| V1 intact | PASS | protected source/tests/artifact hash audit |
| Phase 2 intact | PASS | original 23 tests unmodified and passing |
| Existing Panda Reach | PASS | original CLI headless and real viewer success |
| Existing viewer | PASS | Phase 2 GUI smoke, hold and clean shutdown |
| Observation abstraction | PASS | immutable schema/provider tests |
| Ground-truth mode | PASS | provider, semantic and coordinate integration tests |
| MuJoCo RGB observations | PASS | real tiny-scene and Panda rendering |
| Rendering preserves simulation | PASS | exact integration snapshots and image replay |
| Perception abstraction | PASS | deterministic, ground-truth and injected model tests |
| Deterministic visual perception | PASS | masks/components, noise, ambiguity and tracking tests |
| Structured representation | PASS | WorldState goal resolution/projection tests |
| Genuine observation-derived output | PASS | relocated targets and poisoned privileged fields |
| Ground-truth diagnostic comparison | PASS | opt-in true entities/errors after inference |
| Closed-loop execution | PASS | exact observe/perceive/plan/execute event-order assertion |
| Proper MCTS/world-model futures | PASS | original MCTS, six action priors/root statistics, physics snapshots |
| Visual target task | PASS | six real physical acceptance episodes |
| Semantic goal switching | PASS | same agent/planner/controller/perception red then blue in three scenes |
| Uniform policy | PASS | all standalone acceptance episodes |
| Existing DDM path | PASS | TextLaya adapter with injected predictor; existing Mica tests/CLI preserved |
| Mock DDM | PASS | priors flow through original MCTS expansion |
| Optional VLM boundary | PASS | injected ModelPerceptionAdapter with schema/goal validation |
| Optional direct VLDM boundary | PASS | mock image/goal/action scorer as root policy |
| No real model needed | PASS | no network/services/downloads in tests |
| Headless visual task | PASS | EGL red/blue runs, including display variables unset |
| Watchable visual task | PASS | actual WSL GLFW viewer red and blue smoke |
| Invisible speculative branches | PASS | zero speculative viewer syncs/camera requests; exact live-state preservation |
| Structured Phase 3 logs | PASS | config, observations, steps, summaries and opt-in images/diagnostics |
| Architecture guide | PASS | implementation/formula and camera-boundary review |
| Practical guide | PASS | actual CLI and component API examples |
| Changelog | PASS | appended Phase 3 history below prior entries |
| Test report | PASS | this persistent audit and measured evidence |
| Complete suite | PASS | 83 passed, no skips with Panda and EGL |
| V1 regression | PASS | 41 passed separately |
| Phase 2 regression | PASS | 23 passed separately |

Human judgment of rendered appearance remains recommended: automation verifies viewer initialization, execution, final-scene hold and shutdown, not subjective visual correctness. No required architectural feature is PARTIAL/FAIL; real VLM/VLDM/hardware are deliberately outside scope.

### Automated coverage and compatibility

New Phase 3 tests: **19 passed** across `test_observation.py`, `test_perception_camera.py`, and `test_physical_agent.py`. Final suite **83 passed**: 41 V1 + 23 original Phase 2 + 19 Phase 3. No failures/skips in fully configured final validation. Final results are rerun after source hardening. Ruff, robotics/test/example formatting and git whitespace checks pass.

Before the additional post-implementation acceptance case, full compatibility suites passed on MuJoCo **3.14.0 (82 passed)** and **3.3.7 (82 passed)**. The older binding lacks `mj_copyData`; a tested official integration-state copy/forward/reapply fallback preserves inputs. Camera projection-field changes are handled explicitly. Core-only installation without MuJoCo passed **46 tests**, with four optional modules skipped for missing MuJoCo; no inference dependencies are mandatory. Panda tests skip with a PANDA_MODEL reason when the external model is unavailable. Tiny-scene rendering tests explicitly skip if offscreen GL cannot initialize. No skips occurred in final local Panda/EGL validation.

Tests prove image dimensions/type/read-only format, width/height and Y-axis conventions, projection/plane intersection and rotated-camera roundtrips, calibrated geometry, rendering independence, semantic selection, stale expiry, no hidden-coordinate fallback, exact closed-loop event order, one new observation after every selected action, mock policy swapping, root-only direct visual priors, invisible search, visual geometry dynamics equivalence, failed perception logs without execution, completed-log immutability, and original Phase 2 root-statistics equivalence.

### Physical acceptance runs

Camera/color, uniform policy, seed 0, 60 simulations, horizon 3, 2 cm actions, 150 physics steps/action, epsilon 12 mm. Each scene uses the same objects/components for red then blue; only semantic goal changes and episode state resets. z=0.6245 m is explicit plane calibration. The table records selected-target errors, not all-entity aggregate errors. Positions are meters; errors are millimeters. Planning time is measured local latency, not a performance claim.

| Scene / goal | True target | Final perceived target | Max perception error mm | True final reach error mm | Actions / observations | Planning s | Unique run |
|---|---|---|---:|---:|---|---:|---|
| 0 / red | (0.6045, 0.08, 0.6245) | (0.604379, 0.07945, 0.6245) | 0.563 | 6.781 | 7 / 8 | 4.26 | 20260930T012327_e508ff329b7d |
| 0 / blue | (0.5045, -0.08, 0.6245) | (0.504467, -0.079637, 0.6245) | 0.364 | 11.287 | 6 / 7 | 3.29 | 20260930T012333_ba495698eab1 |
| 1 / red | (0.5945, -0.06, 0.6245) | (0.594718, -0.060161, 0.6245) | 0.271 | 3.818 | 5 / 6 | 2.72 | 20260930T012337_cd94bad84341 |
| 1 / blue | (0.5145, -0.08, 0.6245) | (0.516723, -0.083941, 0.6245) | 4.524 | 3.535 | 6 / 7 | 3.09 | 20260930T012341_3af732d5e8ab |
| 2 / red | (0.6245, 0.04, 0.6245) | (0.625511, 0.044998, 0.6245) | 5.099 | 5.913 | 6 / 7 | 3.20 | 20260930T012345_2b6baa2ac6a5 |
| 2 / blue | (0.5045, -0.1, 0.6245) | (0.504755, -0.09992, 0.6245) | 0.267 | 11.219 | 7 / 8 | 3.72 | 20260930T012349_2aa61661de12 |

All six succeeded. Placements cover positive/negative X and Y displacements and differing nearby distances. `examples/robotics/validate_visual.py` reproduces this small acceptance check; each run includes `acceptance.json`. Initial independent red/blue CLI runs also succeeded (7 actions each), with maximum all-entity errors 0.563 mm and 1.576 mm respectively. Original Phase 2 headless final audit run: `20260930T012438_b9b03b587f5c`, success true.

Stress attempts also exposed expected visibility limits: blue at (0.45,0.09,0.6245) and (0.5145,0.06,0.6245) can be hidden by the initial arm, while (0.4945,-0.10,0.6245) remained hidden beyond the six-frame tracker allowance. These attempts failed explicitly, with retained failure logs; they were not silently converted to successful perception. The acceptance placements require initial visibility and bounded occlusion. This is not broad-workspace or arbitrary-occlusion validation.

### Real viewer validation

Fresh final audit exercised original Phase 2, Phase 3 red and Phase 3 blue with the real WSL display and default GLFW camera. All reached, printed SUCCESS, held the final scene, then received controlled SIGINT and exited **130** cleanly. Captures: `/tmp/ddm-phase3-audit-viewer-{phase2,red,blue}.log`. Earlier independent EGL-camera/red and GLFW-camera/blue viewer smokes also succeeded. No orphan viewer/process remains. Speculative invisibility and physics equivalence are asserted by GUI-free instrumentation; visual appearance still warrants personal inspection.

### Logging, integrity and finalization

Logging adds provider/perception/camera configuration, semantic goal, calibration, estimated entities and staleness, acquisition/perception/planning/execution times, root statistics, warnings, optional truth/errors and optional PPM images. True target entities are read only after inference for diagnostics. Default logging saves no image frames. Failed perception preserves error/success=false and executes no unresolved action. Completed run files are detached from later observations.

All **57 pre-existing runtime/result files** match their pre-Phase-3 SHA-256 hashes. All **51 protected tracked source/test/script/result files** match the Phase 2 checkpoint, including original tests; this broader count includes files beyond the earlier 49-file audit. No historical run/result/report is changed. Main remains the Phase 2 commit. Generated robotics runs, images, caches, dependencies and Menagerie assets are ignored/external and excluded from the commit. SSH remote configuration is reused without credentials changes. A normal Phase 3 branch push is authorized; no merge is performed.

Known limitations: RGB known-plane localization; named perspective fovy cameras only; explicit simulator robot proprioception and privileged dynamics snapshots; static targets and bounded six-frame tracking; color/lighting/visibility assumptions; position-only IK and nearby reaching, without collision/hardware safety guarantees; sequential GL/backend ownership. VLDM priors are root-only because future images are not modeled. No real model inference is validated.

Deliberately deferred: real VLM/VLDM backends/training, arbitrary-depth perception, learned dynamics/belief tracking, real robots, ROS, grasping, RL, large downloads, cloud services, dashboards and any next phase. Recommended next action: personally inspect the pushed Phase 3 branch using PHYSICAL_AI.md before merging.
