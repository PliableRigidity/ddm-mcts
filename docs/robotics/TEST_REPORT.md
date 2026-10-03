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

## V3 Phase 1 — real local VLM — 2026-10-01

Status: **COMPLETE; awaiting manual inspection, deliberately UNCOMMITTED/UNPUSHED** on `v3-phase1-vlm`. Baseline HEAD/main `fa9e489` merges the completed V2 checkpoint. Initial git status clean. Baseline complete suite: **83 passed in 46.14s**. PyTorch/Transformers/Pillow were absent in .venv; existing MuJoCo/numpy packages were preserved. NVIDIA RTX 4090 Laptop GPU, 16 GB VRAM, driver 572.83 (WSL NVIDIA-SMI 570.133.07), CUDA capability 12.8.

### Validation gates and acceptance

| Gate | Status | Verified evidence |
|---|---|---|
| 1 baseline | PASS | inspected branch/interfaces/docs/tests and 83 baseline tests |
| 2 optional dependencies | PASS | uv install and package compatibility check; core-only suite |
| 3 local GPU model | PASS | official Qwen3-VL 4B loaded on CUDA/bfloat16 |
| 4 real standalone inference | PASS | MuJoCo RGB + language goal -> actual model JSON |
| 5 structured parsing | PASS | valid/fenced/invalid schema and grounding tests |
| 6 semantic scene | PASS | cube/sphere/cylinder with identical material, anonymous geom names |
| 7 actual grounding | PASS | real Qwen recognized all three shape goals |
| 8 metric geometry | PASS | image foreground refinement + calibrated ray-plane intersection |
| 9 existing policy | PASS | Uniform real acceptance and mock TextLaya physical integration |
| 10 actual MCTS | PASS | unchanged original search, snapshot/restored futures and root statistics |
| 11 physical reach | PASS | actual Qwen-selected cylinder reaches headlessly |
| 12 goal switch | PASS | same components/model/planner/controller, cylinder then cube then sphere |
| 13 viewer | PASS | real WSL cylinder/cube viewer, selected-action execution and clean hold/shutdown |
| 14 existing modes | PASS | old coordinate and red/blue CLI success; ground-truth tests retained |
| 15 complete suite | PASS | 105 passed: 41 V1 + 42 V2 + 22 V3 |
| 16 docs/logs | PASS | practical commands, conceptual guide, metrics/debug logs, changelog/report |

Every definition-of-done item maps to these passing gates. Real VLM acceptance was performed, not inferred from mocks. No optional cloud/backend fallback is used. VLM semantics do not emit robot actions or policy priors; unchanged MCTS remains the final planner. Viewer automation verifies operation, not subjective appearance; human visual inspection remains recommended.

### Model and performance

Installed optional validation stack: torch **2.7.1+cu128**, torchvision **0.22.1+cu128**, transformers **4.57.6**, accelerate **1.15.0**, Pillow **12.3.0**. `uv pip check` reports all 57 installed packages compatible. Model: `Qwen/Qwen3-VL-4B-Instruct`, snapshot `ebb281ec70b05090aa6165b016eac8ec08e71b17`, normal cache `/home/ishaan/.cache/huggingface/hub/models--Qwen--Qwen3-VL-4B-Instruct/`. Only this model was downloaded; weights remain outside Git/repository.

First standalone real sanity: cache-backed load **7.346 s**, first inference **3.313 s**, CUDA:0/bfloat16; raw response identified cylinder and bbox [543,400,595,466] in the initial more occluded camera view. This exposed geometry sensitivity; the final scene uses a clearer, higher oblique camera. Final three-goal acceptance: **one load, 7.941 s**, three inferences **3.393 / 2.692 / 2.695 s**, total **8.779 s**. Model remains loaded between goals/observations. These are local measurements, not promises; Hugging Face cache checks can add startup latency, and HF_HUB_OFFLINE=1 avoids checks after download.

Generation uses official processor chat templating, input-token trimming, SDPA, greedy bounded output, no quantization/FlashAttention/vLLM. Normal suite has no real-model download/load requirement. The standalone/real acceptance utilities are explicitly invoked separately.

### Real physical results

Scene object centers share explicit calibration plane z=0.6245 m. Same teal material for every object; no text labels in RGB. VLM gets real RGB and goal, not simulator identities/positions. Refined estimates come from the selected box/material pixels and calibration. The table separates true diagnostic position from the visual estimate. Errors are mm; positions meters.

| Goal | Diagnostic true center | Initial estimated center | Max localization error mm | True final reach error mm | Actions / observations | Planning s | Headless execution s | Unique run |
|---|---|---|---:|---:|---|---:|---:|---|
| cylinder | (0.6045, 0.08, 0.6245) | (0.604346, 0.080947, 0.6245) | 2.534 | 6.782 | 7 / 8 | 3.049 | 0.012 | 20261001T184534_7500a6cc39f6 |
| cube | (0.5045, -0.08, 0.6245) | (0.5043, -0.079958, 0.6245) | 6.426 | 8.887 | 7 / 8 | 2.990 | 0.011 | 20261001T184550_15b2a4d35d08 |
| sphere | (0.6245, -0.08, 0.6245) | (0.624418, -0.07993, 0.6245) | 5.941 | 5.999 | 8 / 9 | 3.328 | 0.013 | 20261001T184557_827d6acdb5ed |

All succeeded with UniformPolicy, 60 MCTS simulations, horizon 3, seed 0, 2 cm actions, 150 physics steps/action and 12 mm estimated-goal success tolerance. Real raw responses and parsed boxes are in observations.jsonl/vlm_debug JSON and vlm_acceptance.json. Final object boxes: cylinder [565,404,625,500], cube [370,500,445,610], sphere [488,610,550,683]. All returned correct semantic labels. Self-reported confidence 0.92 is diagnostic, not a calibrated guarantee.

Natural-language CLI acceptance also succeeded: `Reach the spherical object.` headlessly (`20261001T184856_cb8678dd61b2`); cylinder and cube viewer commands succeeded (`20261001T184735_3059f548e692`, `20261001T184752_3afd0e45486e`). Final scenes held, then controlled SIGINT returned expected **130** with clean shutdown. No orphan processes remain. Target marker is display-only and absent from observation frames. Speculative rollouts have no camera/model calls and no viewer publication, enforced by existing execution callback scopes and tests.

Original Phase 2 headless command: `20261001T184857_1b765ccf1810`, success. Original red/blue color commands: `20261001T184859_f62058178225` / `20261001T184904_0a13ad568712`, both success.

### Tests and hardening

New V3 tests: **22 passed**. Cover dependency-light semantic JSON/fenced JSON normalization; malformed/missing/unknown values; invalid/non-finite/out-of-range boxes/confidence; lazy once-only runtime loading; RGB/goal validation; missing optional dependencies; simulated OOM; official runtime PIL conversion/chat template/input trimming through fakes; semantic caching/refresh/reset/goal changes; deterministic geometry; poisoned true entities/task coordinates; no lookup fallback; shared object material; old CLI defaults/new flags; mock VLM -> real TextLaya option priors -> unchanged physical MCTS; exact live snapshot preservation/no speculative observations; physical success; metrics/logs/debug images and default no-image behavior.

Complete suite **105 passed**, no skips with Panda and EGL. Historical V1 separately **41 passed in 0.66s**; existing V2 robotics separately **42 passed in 41.45s**; new V3 separately **22 passed in 5.41s**. Core-only environment without torch/transformers/MuJoCo/numpy: **62 passed, 5 skipped** (four optional MuJoCo modules, new optional numpy/MuJoCo VLM module). Core installation remains dependency-free. Formatter/linter/whitespace pass. Normal model tests use an injected runtime factory and never call Hugging Face.

Development findings: initial broad foreground threshold admitted blue floor pixels and biased localization up to ~16.5 mm. Tightening chromaticity tolerance to 0.12 and adding a blue-background regression fixture reduced final measured max errors to 2.53–6.43 mm. The first camera angle partly hid cylinder; final oblique pose reveals shapes more clearly. Greedy generation clears irrelevant sampling flags. Natural-language goals needed an explicit resolved object label for diagnostics/task matching; optional WorldState.resolved_target_label preserves V2 behavior while avoiding a language-to-simulator-name lookup. Missing DDM objective in a test fixture was corrected; no underlying DDM change was needed. No ground-truth repair/fallback was introduced.

### Logging, limits, integrity and handoff

Logs separate load versus inference time/count, include raw/parsed semantics, grounding, fresh localization/staleness, model/device/dtype/configuration, semantic cache age, goal, optional true-state/entity errors, search/action/execution data and success/failure. Default saves no image frames. Debug is opt-in per live observation. Exceptions retain summary diagnostics and stop rather than substituting privileged coordinates.

Known limits: generated static same-material objects, configured known center-height plane, calibrated perspective camera, approximate silhouette center localization, initial visibility and bounded eight-frame static tracking. Refresh 0 caches semantics but still observes/localizes/plans after every action; dynamic objects/viewpoints need appropriate refresh/tracking. Qwen can produce semantically wrong but schema-valid boxes. CUDA acceptance only; CPU inference not performance-validated. Robot proprioception/dynamics initialization remain explicitly privileged simulator state. No grasping/collision/hardware safety claim.

Integrity: all **235 pre-existing runtime/result files** match pre-V3 hashes; all **54 protected historical source/test/script/result files** remain byte-identical. MCTS, policies, MuJoCo backend and IK controller unchanged. Only authorized API/CLI/logging/docs/packaging additions modify stable toolkit files. No model weights, caches, generated images or logs are tracked. Git remains on v3-phase1-vlm with understandable source/docs/tests changes, and **no commit/push/merge** is performed.

Exact sanity, cylinder/cube/sphere viewer, headless, original Phase 2 and original color commands are in LOCAL_VLM.md. Recommended next action: manually run the semantic viewer before checkpointing. V3 Phase 2/direct visual action scoring, training/RL, hardware/ROS, manipulation, extra models, cloud APIs, navigation and dashboards were deliberately not implemented.

Final standalone rerun on the final camera/implementation: `robotics_runs/20261001T185450_vlm_sanity_4287db5f`, real cylinder grounding, offline cached load 4.596 s / inference 3.465 s. Final full suite rerun: **105 passed in 45.31s**. Optional editable packaging build and installed dependency compatibility check passed. Final git diff whitespace check passed; HEAD and main remain fa9e489, with all V3 changes uncommitted.

## 2026-10-01 — Semantic reach penetration correction

Baseline: 105 passed. Cause: localized object centroid was used directly as the goal and IK controlled the hand body, not the gripper TCP. Added configurable outside-object approach (40 mm radius prior + 50 mm standoff), lift-first receding-horizon waypoints, an invisible hand-local 103.4 mm TCP site with matching site Jacobian, and approach/frame/signed-clearance diagnostics. VLM and deterministic localization are unchanged. Legacy Phase 2 frame and CLI remain compatible.

Regression coverage: four new tests (three physical shape cases plus invalid configuration), including non-centroid outside-object goals and positive gripper/target clearance at every executed physics substep. Existing centroid/hand API test now explicitly requests its legacy frame. Full suite: 109 passed (41 V1, 42 V2, 26 V3/current correction). No model download is required by normal tests.

Real Qwen viewer acceptance, all three successful; windows held at final state and closed via Ctrl+C with no orphan processes:

| Goal | Final TCP target error | Final minimum gripper clearance | Minimum selected-step clearance | Run |
| --- | --- | --- | --- | --- |
| cylinder | 5.12 mm | 29.35 mm | 29.32 mm | 20261001T201901_87f47f1dd15e |
| cube | 8.13 mm | 39.61 mm | 27.73 mm | 20261001T201928_4e4a2f7d173f |
| sphere | 4.37 mm | 38.90 mm | 30.42 mm | 20261001T201956_dd027565e27b |

Viewer validation is automated initialization/execution/shutdown plus numerical clearance, not a substitute for human visual assessment. Diagnostics include center, extent prior, approach vector, standoff, final target/current waypoint, per-action requested endpoint, site/frame, final pose/error and optional true geometry sizes/signed distances. Unique ignored robotics_runs directories retain records. Historical outputs are preserved. Known limits: fixed conservative extent prior, position-only IK, lift-first staging rather than general obstacle avoidance, non-contact reaching only. No grasping, VLM changes or subsequent-phase features. Implementation remains uncommitted for manual inspection.

## 2026-10-01 — V3 Phase 1 final checkpoint

User acceptance: real Qwen viewer and corrected approach behavior manually validated. Finalization scope is source/tests/documentation only; no new phase or merge into main. Earlier uncommitted status and centroid-reaching measurements above describe prior chronological stages; final behavior is the outside-object gripper-TCP approach described in the correction section and LOCAL_VLM.md.

Final checks: repository-wide Ruff lint passes. Robotics/VLM source/examples/tests formatting passes (34 files). Full-repository formatting reports 19 unchanged pre-existing baseline files; an isolated HEAD archive reproduces exactly 19 failures. Historical files are preserved rather than reformatted. Git whitespace checks pass. Original coordinate Panda reach and camera/color red and blue CLI commands succeeded again in unique runs 20261001T203001_d95607206e01, 20261001T203003_f22656e54380 and 20261001T203004_7cf3894af8f0.

Real model remains Qwen/Qwen3-VL-4B-Instruct in external Hugging Face cache /home/ishaan/.cache/huggingface/hub/models--Qwen--Qwen3-VL-4B-Instruct/, snapshot ebb281ec70b05090aa6165b016eac8ec08e71b17. CUDA:0/bfloat16 on RTX 4090 Laptop. Recorded final cylinder viewer: load 11.551 s, inference 3.713 s, one model load and one cached semantic inference. Earlier shared-model three-goal validation measured load 7.941 s and first/subsequent inference 3.393/2.692/2.695 s. Model loading and inference are separate, environment-dependent measurements.

All 235 protected pre-existing output files remain hash-identical. No V1 source/results/reports are changed. Stable V2 APIs remain compatible; the optional site-controller path is limited to semantic-scene use. Models, images, robotics_runs, virtualenvs and caches are excluded from the commit. Commit message: Complete V3 Phase 1 local VLM perception. Push current v3-phase1-vlm branch only, normal SSH push; no merge, force push or subsequent phase. Known limitations and deliberately deferred features remain as documented above.

Final complete suite rerun before commit: **109 passed in 79.00 s**, no failures/skips: **41 V1 + 42 V2 + 26 V3/approach**. Real viewer cylinder/cube/sphere errors and positive clearances remain recorded in the correction table; human manual acceptance is now confirmed.

## 2026-10-02 — V3 Phase 2 visual decision policy

Status: implementation and real headless/viewer acceptance complete; uncommitted on v3-phase2-vldm for manual inspection. The initially clean Phase 2 branch pointed to V2 fa9e489; it was advanced to the existing Phase 1 ac50199 checkpoint before implementation, without creating a new commit or changing main. Baseline: **109 passed in 98.71 s**. Final regression suite so far: **127 passed in 121.92 s** (41 V1, 42 V2, 26 Phase 1/approach, 18 new Phase 2). No normal test needs Qwen, network or GUI. New tests cover strict/fenced/prose JSON, normalization, missing/unknown/duplicate IDs, zero/negative/nonfinite/nonnumeric scores, stable identity/order, calibrated action mapping, poison-answer exclusion, unchanged RGB, alpha 0/0.5/1, permutation averaging/remapping, fresh-image cache scopes, root-only guidance, honest uniform deeper nodes, model reuse, closed-loop observations, exact root-prior logs and CLI separation.

Architecture: QwenVisualPolicy reuses the V2 VisualDecisionPolicy boundary, V1 MixedPolicy and PermutationAveragedPolicy. Qwen3VLBackend gains reusable generate(rgb,prompt); infer preserves its Phase 1 semantic schema and loader. One lazy model is shared across perception and visual decisions. Semantic perception supplies the structured evaluator target; the visual prompt receives no target XYZ, simulator body/geom ID, selected waypoint or answer. Camera/TCP proprioception supplies a magenta current-TCP overlay and calibrated local projected action deltas. Stable IDs are lexicographically ordered; K=1 default, optional K=2/3. Only the root has fresh RGB. Deep nodes use --policy (uniform default). No speculative rendering/model calls, direct argmax execution, new planner, new controller or physics changes.

A deterministic snapshot-backed LineWorld test deliberately makes MOVE_X_NEG the model top choice with 100:1 weight. Actual MCTS selects MOVE_X_POS after 600 simulations, leaves live state unchanged and calls Qwen once; deeper priors are uniform. Physical fake-backend tests prove one shared load for semantic+visual adapters, a fresh observation after each executed action, no per-simulation inference, and actual root probabilities equal logged priors.

Default alpha=0.5; alpha=0 skips visual scoring while preserving semantic perception. Visual CLI c_puct=0.05 matches meter-valued negative-distance rewards; existing modes retain 1.4. Initial real runs with 1.4 failed by over-lifting for all three goals despite positive clearance (runs 20261002T170436_de051b341f37, 20261002T170630_5b9df33527c7, 20261002T170843_fb3d2bd8b00d). These diagnostic runs remain intact. The simulator/objective were correct; oversized exploration plus persistent upward priors dominated limited-budget search. Calibrating PUCT enabled search recovery without leaking target coordinates, changing dynamics, removing standoff or overriding the selected action outside MCTS. Prompt hardening also fixed omitted IDs, copied uniform template values and quoted numeric outputs; invalid output is still rejected, not repaired silently.

Real model: cached Qwen/Qwen3-VL-4B-Instruct, CUDA:0/bfloat16, external /home/ishaan/.cache/huggingface/hub/models--Qwen--Qwen3-VL-4B-Instruct/, snapshot ebb281ec70b05090aa6165b016eac8ec08e71b17. No weights were downloaded again. Standalone valid scoring: load 4.721 s, inference 2.723 s, six legal scores, initial MOVE_Z_POS top-1 (robot begins below clearance). Run visual_decision_sanity_3eacb5bbaed8. This is a prompted VLM-backed policy, not a specially trained robotics VLDM.

### Real acceptance

Settings: UniformPolicy at deeper nodes, 60 simulations, horizon 3, seed 0, 2 cm Cartesian actions, 150 physics substeps, alpha 0.5, c_puct 0.05, K=1. Original panda_gripper_tcp (103.4 mm hand-local Z), 40 mm configured radius, 50 mm standoff and lift-before-translate remain unchanged. Qwen repeatedly favors MOVE_Z_POS even after lifting; all actual horizontal motions below were selected by MCTS, not model argmax. There is no claim of improved semantic action quality or speed over uniform search. This limitation is deliberately exposed in raw outputs and diagnostics.

| Goal | Headless actions / visual calls | MCTS overrides | Mean visual inference s | Headless search s | Final TCP error mm | Final clearance mm | Minimum selected-step clearance mm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cylinder | 17 / 17 | 7 | 2.466 | 3.159 | 5.118 | 29.351 | 29.324 |
| cube | 16 / 16 | 6 | 2.166 | 3.117 | 8.097 | 39.615 | 27.731 |
| sphere | 18 / 18 | 8 | 3.070 | 4.750 | 4.345 | 38.898 | 30.421 |

Headless acceptance: robotics_runs/visual_policy_acceptance_94be1e07dbd2/acceptance.json. One model load (15.172 s), 54 total backend calls = 3 semantic + 51 physical visual calls. Logical visual requests=51, cache hits=0, misses=51. All succeeded. Full action/top-1 sequences and priors are in each run's steps.jsonl and acceptance.json.

Final-prompt real viewer acceptance: robotics_runs/visual_policy_acceptance_501c66c86b9e/acceptance.json. One model load **5.754 s**, 54 backend calls, 51 logical/physical visual requests, no cache hits. All three succeeded with the same physical endpoint metrics. Viewer mean visual inference cylinder/cube/sphere: **3.000/3.579/4.280 s**; search **4.029/5.137/6.414 s**. Runs: 20261002T173042_312a823eecd2, 20261002T173149_1e96c7f732e0, 20261002T173302_3b1b8ae36f8f. Only selected execution publishes frames; Qwen/MCTS thinking leaves the displayed robot stationary. Automated utility holds final scene two seconds then closes; public CLI holds until normal close/Ctrl+C. Human visual correctness remains recommended. No orphan viewer processes are retained.

Logging distinguishes shared model load/count, backend total calls, logical visual requests, cache hits/misses, physical visual calls/time/mean, policy wall time, MCTS search time excluding root-policy wall time, controller time and component-summed decision latency. Scores, descriptions/order, raw response, normalized visual distribution, alpha/mixed priors, top-1, visits/Q, selected action/agreement and final TCP/clearance are retained. No images are saved by default. New observations invalidate prior caches; only duplicate requests within one bound image can hit.

Compatibility smoke: original Phase 2 coordinate task succeeded (20261002T173204_058d6b87ef24); red and blue color modes succeeded (20261002T173207_0b589da5a4eb, 20261002T173216_55bc5e894799). All 534 pre-existing outputs match pre-Phase-2 hashes. Historical V1 code/results/reports are untouched. Approach task, semantic scene, controller and per-substep clearance tests are unchanged. Ruff lint/changed Python formatting/whitespace pass; full repository formatting retains its 19 historical baseline failures.

Files added: robotics/qwen_visual_policy.py; examples/robotics/visual_decision_sanity.py and validate_visual_policy.py; tests/test_visual_scores.py and tests/robotics/test_qwen_visual_policy.py; docs/robotics/VISUAL_DECISION.md. Files modified: robotics/cli.py, physical_agent.py, run.py, visual.py, vlm_backend.py; architecture guide, robotics README, LOCAL_VLM guide, this report and changelog. No new dependencies. No model/cache/image/run/virtualenv/credential artifacts are staged. No commit, push or merge.

Known limitations: persistent upward visual bias; heuristic uncalibrated scores; generated static known-plane scene and approximate silhouette localization; root-only images; position-only approach; configured radius prior; no general collision safety. Alpha/budget/PUCT sensitivity remains material. No grasping, gripper closure, contact manipulation, training/fine-tuning, new model/cloud server, ROS/hardware, RL or Phase 3. Next action: manually inspect the visual-policy CLI before checkpointing. Exact commands and component/information boundaries: VISUAL_DECISION.md.

Final rerun after all code hardening: **127 passed in 115.38 s**, no skips/failures. The permutation seed/cache state now uses only observation sequence and user goal (not privileged rollout snapshots); model loading, score generation and root selection remain unchanged. Exact public CLI visual mode succeeded headlessly with the natural-language cylindrical-object goal: 20261002T173459_ee7e1f977ec9. Original Phase 1 shared-model uniform-MCTS semantic acceptance was rerun for cylinder/cube/sphere and all succeeded (runs 20261002T173608_5bd9f3980790, 20261002T173635_dccaf2c5d390, 20261002T173651_0dd513d5010e).

| Validation gates | Status | Evidence |
| --- | --- | --- |
| 1 baseline | PASS | 109 pre-change tests |
| 2 existing visual boundary | PASS | VisualDecisionPolicy root binding reused |
| 3 shared backend | PASS | one load across semantic/visual tests and real utility |
| 4 standalone | PASS | real six-score numeric Qwen result |
| 5–6 stable actions/parser | PASS | 14 schema tests and semantic remapping tests |
| 7–8 root injection/deep fallback | PASS | logged priors equal root statistics; deep priors uniform |
| 9–10 override/no argmax execution | PASS | actual MCTS overturns bad 100:1 root prior |
| 11–12 fresh observations/no per-simulation Qwen | PASS | closed-loop tests; real 51 decisions/51 scoring calls |
| 13–15 real goals | PASS | all three headless and viewer episodes succeed |
| 16 TCP/standoff | PASS | unchanged source; per-substep clearance regressions; positive real clearances |
| 17 viewer | PASS | real all-goal viewer execution/final hold/clean close |
| 18 headless | PASS | all-goal utility and exact public CLI |
| 19–21 regressions | PASS | 41 V1 + 42 V2 + 26 Phase 1 in full suite; legacy CLI/real Phase 1 reruns |
| 22 docs/logging/accounting | PASS | source-matched guides, root stats/latency logs, persistent report/history |

All definition-of-done items are covered by these passing gates. Meaningful visual-policy reliability remains a known limitation: the model often returns an upward prior independent of desired lateral direction; success requires the explicit search/evaluator, not a claim that Qwen is a trained or reliable robot controller.

Exact public CLI viewer smoke: visual mode and original Phase 1 structured mode both succeeded, held final scenes, then returned expected Ctrl+C exit 130 with clean shutdown. Records: robotics_runs/cli_viewer_acceptance_ff977aed439b/acceptance.json. No acceptance Python/viewer process remains.

## 2026-10-02 — V3 Phase 2 final checkpoint

User manually accepted the real visual-policy cylinder viewer and safe TCP approach. Finalization retains all prior real results, including failed c_puct=1.4 runs, persistent Qwen upward bias, successful c_puct=0.05/alpha=0.5 search recovery, and the automated bad-prior override proof. Earlier uncommitted status describes the pre-checkpoint stage. Only appropriate source, tests and documentation are checkpointed on v3-phase2-vldm using “Complete V3 Phase 2 visual decision MCTS integration”; no merge into main, force push or new phase.

Ruff lint, all ten changed/new Python files' formatting and git whitespace checks pass. TCP/controller/approach/semantic-scene/MCTS sources and original per-substep clearance tests are byte-identical to Phase 1 HEAD. All 534 pre-Phase-2 historical outputs remain hash-identical; all 802 current output files were additionally protected at finalization. Qwen weights remain in the external Hugging Face cache. No runtime/model/image/cache/virtualenv/credential artifacts belong in the commit. Documentation retains the distinction between VLM perception, prompted visual priors, structured DDM priors, MCTS selection and MuJoCo transitions.

Final pre-commit suite: **127 passed in 101.73 s**, no failures/skips: **41 V1 + 42 V2 + 26 V3 Phase 1/approach + 18 V3 Phase 2**. Ruff lint, changed-code formatting, staged artifact/security scan and git whitespace checks pass. Existing results and limitations are preserved. Main remains fa9e489; only the current Phase 2 branch is pushed.


## 2026-10-03 — V3 Phase 3 final implementation validation

**Status: COMPLETE; intentionally uncommitted on `v3-phase3-agent` for manual inspection.** Baseline main/HEAD: `2d13da0622c4be70bb8f5c00804eebe3be3a234c`, including V3 Phase 1 `ac50199` and Phase 2 `67444ac`. Baseline: **127 passed in 72.05 s**, no skips/failures. Final complete suite: **154 passed in 126.50 s**, no skips/failures: **41 V1 + 42 V2 + 26 V3 Phase 1/approach + 18 V3 Phase 2 + 27 new Phase 3**. Normal tests do not load Qwen, require network or open a GUI.

New tests: `tests/test_ordered_tasks.py` (13) and `tests/robotics/test_ordered_agent.py` (14). They cover bounded grammar, ordering, duplicates, rejection, criteria/budgets, status/advancement, failure stops, retained completed results, independent verification, false episode success, mislabeled localization, true-object evaluation, state/snapshot persistence, forbidden resets, fresh trees, semantic/visual cache invalidation, model sharing/accounting, actual camera/visual priors/MCTS execution, no future-image calls, all-object substep clearance, alternate order, one viewer session and CLI final hold. Existing bad-visual-prior MCTS override proof passes unchanged.

### Real cached-Qwen acceptance

Model: `Qwen/Qwen3-VL-4B-Instruct`, CUDA:0 / torch.bfloat16, existing external cache `/home/ishaan/.cache/huggingface/hub/models--Qwen--Qwen3-VL-4B-Instruct/`. Offline mode was used; no weights were downloaded or copied into Git. Defaults: 60 simulations, horizon 3, seed 0, 20 mm Cartesian actions, 150 physics substeps, TCP site `panda_gripper_tcp` at hand-local Z=103.4 mm, bounding radius 40 mm, standoff 50 mm and +Z lift-first approach. Uniform MCTS uses c_puct=1.4. Visual mode retains alpha=0.5, c_puct=0.05, K=1 and uniform deeper-node policy.

The following are physical multi-step episodes, **not reset single-target episodes**. Each task preserved robot/scene/camera/model/planner, observed afresh on transitions, and verified the actual TCP near both the estimated final waypoint and the requested object's true waypoint. Ground truth was used only for verification/diagnostics. All start/prepared states and full preparation snapshots match; each end state matches the next start state. The main three-object task executed 32 actions and observed 35 times, including final checks and next-goal observations. All 4,800 selected physics substeps had positive gripper clearance against all three geometries.

| Mode | Ordered target | Actions | Estimated TCP error mm | True waypoint error mm | Final selected-object clearance mm | Minimum all-object transit clearance mm | MCTS overrides |
|---|---|---:|---:|---:|---:|---:|---:|
| Phase 1 semantic / uniform | cylinder | 17 | 5.12 | 4.35 | 29.35 | 27.69 | 0 |
| Phase 1 semantic / uniform | sphere | 9 | 4.71 | 4.72 | 40.49 | 29.35 | 0 |
| Phase 1 semantic / uniform | cube | 6 | 6.09 | 5.06 | 40.41 | 30.50 | 0 |
| Phase 2 visual / shared semantic | cylinder | 17 | 5.12 | 4.34 | 29.35 | 27.69 | 7 |
| Phase 2 visual / shared semantic | sphere | 9 | 4.71 | 4.72 | 40.49 | 29.35 | 9 |
| Phase 2 visual / shared semantic | cube | 6 | 6.09 | 5.06 | 40.40 | 30.50 | 6 |

All six subgoals succeeded. Visual Qwen top-1 was **MOVE_Z_POS on all 32 decisions**. MCTS disagreed on **22/32** actual actions (7 cylinder, 9 sphere, 6 cube). The later lateral travels succeeded entirely through search recovery. This is not evidence that Qwen is a better robot policy or improves search. The Phase 2 c_puct=1.4 failed runs and explanation remain intact; no tuning, prompt leakage or result rewriting was used to hide the upward bias.

| Accounting | Phase 1 headless | Phase 2 continuous viewer |
|---|---:|---:|
| Model load count | 1 | 1 |
| Model load seconds | 4.945 | 5.772 |
| Logical perception requests | 35 | 35 |
| Semantic inference calls | 3 | 3 |
| Semantic cache hits | 32 | 32 |
| Semantic inference seconds | 8.932 | 10.433 |
| Logical visual requests | 0 | 32 |
| Physical visual calls | 0 | 32 |
| Visual cache hits | 0 | 0 |
| Visual cache misses | 0 | 32 |
| Visual inference seconds | 0 | 86.713 |
| Shared physical model calls | 3 | 35 |
| Total model inference seconds | 8.932 | 97.147 |
| Mean physical visual inference seconds | unavailable | 2.710 |
| MCTS search seconds | 13.413 | 7.046 |
| Controller execution seconds | 0.541 | 3.213 |
| Task wall seconds | 30.831 | 116.750 |

These timings are diagnostic measurements, not a controlled performance comparison: runs used different pacing and shared machine load. Model load time is distinct from inference. Phase 1 first/subsequent inferences were 3.311 / 2.836 / 2.785 s. Visual mode shared one backend for semantic and visual inference: 3 semantic + 32 visual = 35 physical calls, with one load. Per-subgoal action sequences, raw responses, priors, root visits/Q and Qwen/MCTS disagreements remain in the referenced run logs.

Second-order acceptance `Go to the cube and then the cylinder.` succeeded with the same component configuration: cube 16 actions, TCP error 8.13 mm, clearance 39.61 mm; then cylinder 13 actions, TCP error 5.55 mm, clearance 30.70 mm. No reset; one model load, two semantic calls, 31 observations. Final true-waypoint errors were 7.33 / 5.07 mm. This rules out a hard-coded cylinder/sphere/cube ordering.

### Viewer, regressions, outputs and repository protection

Both actual WSL viewer paths completed cylinder -> sphere -> cube in one continuous viewer session, using the real cached Qwen model, and closed cleanly through the bounded acceptance utility. No orphaned viewer/model process remains. The viewer observes display copies and selected execution only; search branches stay invisible. Automated no-GUI tests additionally verify one launch across goals, nested callback monitoring and CLI hold only after the whole task. The normal CLI retains the final scene until manual closure; the validation utility closes automatically. This verifies initialization/execution/lifecycle, not independent human judgement of every visual detail; manual final inspection is recommended before checkpointing.

Original public CLI headless smokes passed for coordinate reach, red and blue color perception, natural-language cylinder VLM perception, and natural-language cylinder visual policy. All original tests are retained and pass. Controller, TCP site configuration, semantic scene, approach logic and MCTS are unchanged. The only backend change composes nested execution callbacks so clearance monitoring and viewer synchronization both run; existing deterministic callback/physics tests pass.

Runs (UTC timestamps):

- Final Phase 1: `robotics_runs/20261002T234331_task_1c77126da611/task_trace.json`.
- Real Phase 2 viewer: `robotics_runs/20261002T233135_task_7cda20bec9cb/task_trace.json`.
- Real Phase 1 viewer: `robotics_runs/20261002T233032_task_d640886be221/task_trace.json`.
- Headless visual acceptance: `robotics_runs/20261002T232646_task_721d4f1baef1/task_trace.json`.
- Final second order: `robotics_runs/20261002T233940_task_9594e9e4cf10/task_trace.json`.
- Coordinate/red/blue/single VLM/single visual: `20261002T233826_9a783d667d1b`, `20261002T233828_df734b80908f`, `20261002T233832_c26ae7844893`, `20261002T233836_376fd39b484c`, `20261002T233852_694996f4a8c9` under `robotics_runs/`.

Ruff repository lint, formatting of all nine changed/new Python files, git whitespace and new-file whitespace checks pass. All **802 pre-existing protected result/report/run files** remain hash-identical. No old tests/results/logs were removed or overwritten. Generated runs, caches, images and the virtual environment remain ignored; no weights/runtime/credential artifacts are tracked or staged. Main and HEAD remain `2d13da0`; no commit, push or merge was performed.

### Definition-of-done audit

| Requirement | Status | Evidence |
|---|---|---|
| Existing baseline | PASS | 127 tests before edits |
| Unified API / interchangeable components | PASS | PhysicalAgent.run_task; original interfaces; ground-truth and camera tests |
| Ordered representation / supported language | PASS | PhysicalTask, ApproachGoal, parser grammar tests |
| Tracked status / physical completion | PASS | Pending/active/succeeded/failed trace; independent waypoint/identity/clearance checks |
| Failure stops and preserves results | PASS | Budget, perception, false-success and later-failure tests |
| Panda and scene persist / no hidden reset | PASS | Forbidden reset tests; real state continuity and full snapshot comparisons |
| Fresh MCTS goal state | PASS | New roots per decision; original MCTS unchanged |
| Perception and visual caches refresh | PASS | Reset/reobserve per subgoal; goal-change/fresh-bind tests |
| One shared Qwen load | PASS | Real one-load tasks and mock shared-backend accounting |
| Safe transit | PASS | All-object signed substep checks; minimum distances above |
| Real Phase 1 cylinder/sphere/cube | PASS | Final headless and continuous viewer traces |
| Second ordering | PASS | Real cube/cylinder trace and reordered/duplicate tests |
| Real Phase 2 multi-step exercised | PASS | Real root priors, MCTS and shared semantic path; headless + viewer |
| Honest Qwen limitation | PASS | 32 upward top choices; 22 MCTS disagreements; old failed runs retained |
| Continuous viewer | PASS | Real full-task execution/clean closure; fake-viewer API and CLI hold tests |
| Structured trace / per-goal metrics | PASS | task_trace.json plus original referenced per-step logs |
| Single-goal Phase 1 / Phase 2 | PASS | Public CLI smokes and unchanged regression tests |
| V1 / V2 / V3 regressions | PASS | 41 / 42 / 26 / 18 tests |
| No runtime/model artifacts committed | PASS | Ignored/external outputs; no staged files; no commit |
| Accurate architecture/docs | PASS | Architecture, multi-step guide, practical guides, changelog and this report |

Limits: simple fixed scene and bounded object grammar; calibrated-plane VLM localization; configured extent priors; primarily position-based IK; approach is not grasping; no general collision planner; gripper clearance sampling is not whole-robot/hardware safety certification; Qwen upward bias and prior sensitivity persist. Successful search recovery is not evidence of model superiority. No new model/download, training, grasping, ROS/hardware, RL, unrestricted language planning, multi-agent system or V4 work was added. Exact manual commands and programmatic composition: [MULTI_STEP_AGENT.md](MULTI_STEP_AGENT.md). Next action at this implementation checkpoint: manually inspect both multi-step viewer paths before checkpointing.

### Final checkpoint — manual acceptance and pre-commit validation

The user manually accepted both full cylinder → sphere → cube viewer modes. They confirmed requested ordering, no reset or teleport, persistent physical state, continuous safe transitions, correct TCP waypoint marker, one continuous viewer and successful task completion.

The fresh complete suite passed **154 tests in 163.17 s**: V1 **41**, V2 **42**, V3 Phase 1/approach **26**, V3 Phase 2 **18**, new Phase 3 **27**. Ruff passed, all nine changed Python files passed formatting checks, and `git diff --check` passed. Existing single-goal compatibility is covered by the unchanged regression tests and the recorded CLI smoke runs above.

The real results above are unchanged: all **32** visual-policy top choices were **MOVE_Z_POS**, and MCTS disagreed on **22/32** actions, with **alpha=0.5** and **visual c_puct=0.05**. The earlier c_puct=1.4 failure remains documented. Successful execution demonstrates MCTS recovery, not superior Qwen action reasoning. Real-model measurements were not rerun or altered for finalization.

All **802** inventoried historical result/report/runtime files remained byte-for-byte unchanged. The source/documentation/test manifest contains no runtime logs, task traces, screenshots, weights, caches, environments or credentials. Qwen remains in the external Hugging Face cache; no model download or dependency change occurred. Finalization is confined to `v3-phase3-agent`; main remains `2d13da0`. The authorized checkpoint commit is `Complete V3 multi-step physical AI agent`; only that branch is to be pushed, without merging. The limitations listed above remain applicable.

## 2026-10-03 — V4 Phase 1 final implementation validation

**Status: COMPLETE implementation, left uncommitted on `v4-phase1-grasping` for manual viewer inspection.** The branch began clean at main merge `a8f0717`. Baseline **154 passed in 135.84 s**. Final full suite **200 passed in 137.48 s**, no skips with the external Panda model available:

| Regression group | Passed |
|---|---:|
| V1 | 41 |
| V2 | 42 |
| V3 Phase 1 / approach | 26 |
| V3 Phase 2 | 18 |
| V3 Phase 3 | 27 |
| New V4 manipulation | 46 |

Command: `MUJOCO_GL=egl PANDA_MODEL=/home/ishaan/robot-arm-playground/mujoco_menagerie/franka_emika_panda/scene.xml PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q -o addopts=''`. The normal tests need no Qwen download, inference, network or GUI. Tests requiring the external Panda explicitly skip if PANDA_MODEL is absent; optional MuJoCo/NumPy tests skip if those extras are absent.

### Six-DoF and physical validation

SO(3) tests cover zero/small/90°/180° rotation, quaternion sign/normalization, and invalid poses. Real pose tests exercise translational and rotational Jacobians, multiple target orientations, tolerances, live-state non-teleportation, joint/actuator limits and position-only compatibility. Additional real pose checks around world X (+0.2 rad), Y (-0.2 rad) and Z (+90°) converged to position errors **0.391 / 0.285 / 0.107 mm** and orientation errors **0.00190 / 0.00175 / 0.00316 rad**. Unreachable poses fail actual verification rather than claiming success.

The gripper physically opens to approximately **80 mm**, closes empty to near zero and stops at nonzero width against the object. Every selected substep is monitored; normal force is required to identify load-bearing contact. No object attachment, added weld, post-initialization object qpos write, forced velocity or disabled gravity is used. Only Menagerie's original finger-coupling equality remains. Pose IK scratch updates do not mutate live joints.

### Real MuJoCo acceptance — three trials per target

Final manifest: `robotics_runs/20261003T003930_pickup_validation_dfc49631543e/validation.json`. Three independently initialized trials per object were identical under this deterministic configuration:

| Metric | Cube | Cylinder |
|---|---:|---:|
| Strategy | Top-down box-face pinch | Top-down radial pinch |
| Successes | 3/3 | 3/3 |
| Pre-grasp position error | 0.403 mm | 0.399 mm |
| Pre-grasp orientation error | 0.000265 rad | 0.000261 rad |
| Grasp position error | 0.382 mm | 0.384 mm |
| Grasp orientation error | 0.000264 rad | 0.000262 rad |
| Load-bearing finger contact | Bilateral | Bilateral |
| Final measured width | 39.986 mm | 39.610 mm |
| Object elevation | 99.079 mm | 97.520 mm |
| Relative positional drift | 1.295 mm | 2.022 mm |
| Object orientation drift | 0.000699 rad | 0.170918 rad |
| Table contact after lift | None | None |
| Checked hold duration | 1.0 s | 1.0 s |
| Executed physics substeps monitored | 5,725 | 5,700 |
| Maximum target-contact penetration | 0.679 mm | 0.565 mm |
| Forbidden contacts | None | None |

The cylinder rotated about 9.8° while held, within the explicit 0.35 rad verification bound; this is not hidden or described as rigid attachment. Longer retention remains contact-sensitive. The configured 100 mm lift succeeds only when actual object height increases at least 75 mm, relative drift is below 15 mm, bilateral force-bearing contact persists and table support is absent. All hold blocks are checked; later success cannot erase a transient failed hold.

Default contact/actuation allowed cylinder slip and drop. Compliance-only changes still slipped during a 15 s diagnostic hold. The final pickup-only scene uses 5 ms pad/object contact response, finger stiffness 400 N/m and damping 20 N s/m, preserving original force limits and friction. Full changes, limits and rationale are in MANIPULATION.md. External Menagerie assets and V3 static scene/controller/approach behavior are unchanged.

### Failure and viewer acceptance

Tests deliberately offset the grasp by 80 mm: closure without bilateral contact fails at verify_grasp and never lifts. Bad pre-grasp pose/clearance fails before contact. Dropped/unsupported/no-force objects and forbidden hand/table or wrong-object contact are rejected. Logs preserve failure stage/reason, actual state and already executed motion. Snapshot tests include free-object and finger state.

Final WSL viewer manifest: `robotics_runs/20261003T003933_pickup_validation_cb2c5c7b4cf1/validation.json`, one complete cube and one cylinder pickup, both successful and cleanly closed. The bounded utility closes automatically; normal CLI keeps the final scene open. No orphaned viewer/model process remains. A rendered final cube scene was inspected and shows the cube suspended between the fingers above the support table. Normal tests use a fake viewer to verify one session, display-copy isolation and nested execution callbacks without a GUI. Final personal viewer inspection is still recommended before the checkpoint.

### V3 regression runtime validation

The original coordinate CLI succeeded: `robotics_runs/20261003T003604_d74c9fcb0a3c/summary.json`. Real cached-Qwen V3 regression smokes also succeeded:

- Continuous semantic-perception cylinder → sphere → cube: `robotics_runs/20261003T003603_task_635316610f48/task_trace.json`.
- Visual-policy cylinder approach: `robotics_runs/20261003T003648_task_0025235ae38a/task_trace.json`.

Both shared one existing Qwen backend/model load, with 21 physical inferences total. No download or new model was needed. The last real visual response still assigned all score to MOVE_Z_POS; prior upward-bias and MCTS-recovery findings remain intact. Original color, single-goal, visual-policy and ordered-task tests pass unchanged. The original controller, semantic scene, approach, MCTS, perception/model adapters and multi-step agent are byte-for-byte unchanged.

### Repository and definition-of-done audit

All required implementation gates pass: six-DoF control; measured physical finger motion; structured contacts; distinct shape-aware grasps; safe transit/pre-grasp/approach; phase-aware contact monitoring; bilateral grasp verification; physical lift; independent elevation/relative-pose/support/hold verification; invalid-grasp failure; real cube/cylinder trials; complete bounded viewers; CLI/API/logging; all regressions and documentation.

Ruff, formatting of all ten changed/new Python files and whitespace checks pass. **960 inventoried pre-existing result/report/runtime files remain hash-identical**. Existing tests were not weakened or changed. Generated pickup traces, validation manifests, screenshots and model/cache artifacts are excluded/ignored; staging remains empty. No commit/push/merge occurred. The public Python API and exact manual commands are documented in [MANIPULATION.md](MANIPULATION.md).

Known limits: known upright cube/cylinder in a simple simulated table scene; deterministic simulator geometry; configured trajectory/tolerance/contact rules; finite hold and contact sensitivity; no full collision planner, placement, learned grasping or hardware safety claim. No V4 Phase 2/3, another model, training, RL, ROS or real hardware was implemented.

### Phase 1 checkpoint finalization

The authorized Phase 1 finalization reran the full suite: **200 passed in 134.29 s**, no skips (41 V1, 42 V2, 71 V3, 46 V4 Phase 1). Ruff, all ten changed-code formatting checks and whitespace checks passed. The complete source/test/documentation audit found no Phase 2 placement or rearrangement implementation. Physics uses scratch-only arm IK, actual finger actuation and force-bearing contacts; object free-body positions are supplied only by scene/keyframe initialization before the episode, never written to fake grasping or lift.

All previously measured cube/cylinder results and the cylinder rotation limitation above remain unchanged. All 960 protected historical files are hash-identical; previous V3 source/tests are unchanged and documentation is preserved with appended additions. Only the 15 Phase 1 source/test/example/documentation files belong to this checkpoint; runtime logs, validation manifests, images, weights, caches, environments and credentials are excluded. The authorized commit is `Complete V4 Phase 1 physical grasping`, pushed only to `v4-phase1-grasping`; main remains `a8f0717`, with no merge or Phase 2 work. The earlier uncommitted status describes the implementation checkpoint preceding this finalization.

## V4 Phase 2 — physical placement and continuous rearrangement (2026-10-03)

### Baseline and regression scope

The prerequisite Phase 1 merge `da9884ac0e2d2fbd21702e88ededdbf912215703` was authorized, committed and pushed to main. `v4-phase2-pick-place` was created from that clean merge with Phase 1 `c05339d` in history. Baseline: **200 passed in 132.79 s**. Phase 2 adds **71 tests**; all historical tests remain unchanged. The final suite contains 271 tests: V1 41, V2 42, V3 71 (26 perception/approach, 18 visual policy, 27 ordered agent), Phase 1 manipulation 46, Phase 2 placement 71.

Tests cover offset rigid transforms/inversion/composition, desired-object to TCP pose, geometry-derived destinations/extents/gap/workspace, pre-place/place/retreat, transport height, contact semantics, force-bearing retention and drift, cylinder symmetry/tilt, independent final-pose relation including hovering/tipped/overlapping cases, stable/unsupported/unreleased/fast/tipped/incorrect placement, actual physical releases, invalid-destination stop, one bounded retry/limit, no-retry drop, substep contact-loss stop, finite unloading grace, physical state continuity, trace contents, continuous viewer display isolation and legacy/new CLI parsing. Real MuJoCo tests run without Qwen or GUI; external PANDA_MODEL is required for physical tests. Tests do not weaken existing grasp verification.

Full command:

```bash
MUJOCO_GL=egl PANDA_MODEL=/home/ishaan/robot-arm-playground/mujoco_menagerie/franka_emika_panda/scene.xml \
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q -o addopts=''
```

### Repeated real-physics acceptance

Manifest: `robotics_runs/20261003T020336_placement_validation_0d9a68e954ec/validation.json`. Every requested trial was successful. Earlier diagnostics and failures are preserved separately; the manifest does not erase them. Trials were physically deterministic under this configuration.

| Metric | Cube absolute (3/3) | Cylinder next_to (3/3) | Sequential cube (2/2) |
|---|---:|---:|---:|
| Pre-grasp position error | 0.403 mm | 0.399 mm | 0.397 mm |
| Pre-grasp orientation error | 0.000265 rad | 0.000261 rad | 0.000261 rad |
| Bilateral force-bearing grasp | Yes | Yes | Yes |
| Grasp width | 39.986 mm | 39.594 mm | 39.986 mm |
| Object lift | 99.079 mm | 97.520 mm | 99.069 mm |
| Free transport path | 112.266 mm | 181.717 mm | 195.821 mm |
| Minimum free-transport table clearance | 99.036 mm | 90.143 mm | 99.029 mm |
| Maximum relative positional drift through carry/descent | 2.626 mm | 13.780 mm | 3.480 mm |
| Maximum retention rotation drift | 0.031084 rad | 0.016524 rad (axis tilt) | 0.040039 rad |
| Maximum full SO(3) rotation drift | 0.031084 rad | **0.748772 rad** | 0.040039 rad |
| Placement position error | 0.267 mm | 0.132 mm | 0.090 mm |
| Placement orientation error | 0.0000299 rad | 0.0001826 rad (upright tilt) | 0.00000474 rad |
| Settling-window maximum linear speed | 1.58e-10 m/s | 0.000125 m/s | 1.61e-10 m/s |
| Settling-window maximum angular speed | 1.16e-8 rad/s | 0.005195 rad/s | 6.70e-9 rad/s |
| Supported/released/stable/upright | Yes | Yes | Yes |
| Physics substeps | 15,725 | 16,700 | 17,100 |
| Forbidden contacts | None observed | None observed | None observed |

Both sequential trials completed cylinder placement followed by cube placement without reset. Operation 1 matches the cylinder metrics above. Exact end/start robot/object records match; all Panda/scene/object reset counts are zero. The final read-only checks independently confirm both objects are supported, released, upright and slow. Final representative centers: cylinder `(0.420100,-0.079918,0.374971)`, cube `(0.460069,0.099949,0.369972)` m. The cylinder is next_to the cube when operation 1 finishes; moving the cube in operation 2 intentionally changes that relation.

The independently observed cylinder/cube relation after operation 1 has horizontal center distance **79.902 mm**, surface gap **39.878 mm**, overlap **0 mm**, within the documented [30,50] mm gap interval. Both actual support contacts are checked. `next_to` is derived from geometry and feasible X/Y candidates; it is not a magic fixed destination.

Across the eight acceptance plans (ten operations): **164,875 executed physics substeps**, **2,169,022 intended contact observations** (per substep, not distinct episodes), **zero forbidden contacts**, maximum observed penetration **0.679 mm**. Timings and every desired/actual pose are in the manifest; headless operation times varied with concurrent validation load. No collision-free claim is made because grasp/support contacts are intentional.

### Diagnostics, recovery and failure

Initial 4 mm transport increments triggered finger unloading. Reducing held-motion increments to 1 mm maintained the cube grasp without changing frozen contact/servo parameters. Supported descent correctly permits finger-force unloading only after real object/table contact. Cylinder yaw is symmetric, so retention bounds axis tilt while reporting full rotation separately. Axial drift is substantial and positional drift approaches the 15 mm limit; no rigid or indefinite retention claim is made. A bounded 20 ms force-unloading filter requires ongoing bilateral pad geometry, nonempty measured width and valid drift; acceptance gaps were ≤4 ms. Tests stop immediately for geometry loss or excessive drift and stop prolonged force unloading.

An initially nearer Y-side cylinder destination obstructed the next cube grasp's open finger, producing a real forbidden-contact failure. Geometry now excludes narrow Y gaps that occupy the reference's open-jaw sweep. That failed trace remains preserved; no collision rule was disabled.

Controlled recovery run: `robotics_runs/20261003T015718_pick_place_a64afd4f8631/summary.json`. A deliberately offset first grasp physically closed empty and failed bilateral verification. The system opened, retreated through actuation, recomputed and succeeded with exactly one retry, then physically placed the cube. Controlled invalid-destination run: `robotics_runs/20261003T015730_pick_place_d54271bd5141/summary.json`. The first operation failed with `invalid_destination`, executed zero physics substeps, preserved state and never started the second operation. Unit tests also enforce the retry limit, drop stop and no infinite loops. No post-release autonomous regrasp recovery is implemented.

### Viewer, compatibility and audit

Real WSL/GLFW viewer manifest: `robotics_runs/20261003T020447_placement_validation_43cde92aa216/validation.json`. Both single placements and the continuous two-operation plan completed successfully; each plan used one scene/viewer with no reset. The utility closes after the plan; normal CLI keeps the final scene open for manual inspection. Fake-viewer tests check display-copy isolation, nested callbacks and one continuous session.

Fresh unchanged Phase 1 pickup regression: cube and cylinder succeeded, `robotics_runs/20261003T015722_pickup_validation_b8e43162bd03/validation.json`. Original coordinate CLI succeeded, `robotics_runs/20261003T020226_367a663f9744/summary.json`. Complete original V1/V2/V3/Phase 1 regressions are preserved; Qwen inference need not be rerun because its backend/adapters and V3 controllers/scenes/agent are unchanged. The frozen upward bias and alpha=0.5/c_puct=0.05 MCTS recovery findings remain documented exactly.

The Phase 1 manipulation scene, controller, gripper, contact implementation, grasp generator, pickup state machine, tests, example and external Menagerie assets are unchanged. No new friction/servo/contact tuning, object qpos writes, added weld/attachment, gravity removal, forced following, velocity zeroing or freezing is used. Full snapshots are read solely for continuity logging, never restored inside manipulation. Documentation additions preserve historical sections verbatim. All **1,032 inventoried historical result/runtime files** remain hash-identical; new ignored runtime output uses unique directories. No Phase 2 changes are staged, committed, pushed or merged. Scope remains deterministic known-scene placement/rearrangement with bounded recovery, no Qwen manipulation or V4 Phase 3, general collision planner or hardware safety claim.

Final validation after the complete trace/continuity audit: **271 passed in 198.74 s**, no skips. Repository-wide Ruff, all seven changed/new Python formatting checks, tracked/untracked whitespace checks and the artifact allowlist pass. Both new single-operation headless CLI paths succeeded as well. Nine earlier physical diagnostic episodes stopped under development configurations (eight retention-condition stops and one wrong-object collision); these are retained rather than counted as successes. The separate invalid-destination acceptance is an expected tenth failed trace. The final repeated acceptance manifest reports all eight requested trials, not the earlier diagnostic experiments. New Phase 2 source/tests/documentation remain uncommitted on the requested branch pending manual viewer acceptance.

### Phase 2 manual acceptance and checkpoint finalization

The user manually accepted cube absolute pick-and-place, upright cylinder next-to-cube placement, and continuous cylinder-then-cube rearrangement. They observed physical grasp/lift/transport/descent/release/retreat, stable final objects, and continuity from the current Panda state without robot/scene/object reset. No teleportation, fake attachment or obviously artificial object motion was observed. This manual acceptance supplements the automated contact/transform/state audits; it does not change the measured results, cylinder drift limits or nine retained development failures above.

The user authorized checkpoint `Complete V4 Phase 2 physical pick and place` on `v4-phase2-pick-place`, pushed only to that branch. Main remains the Phase 1 merge `da9884ac0e2d2fbd21702e88ededdbf912215703`; no Phase 2 merge or Phase 3 work is included. Earlier uncommitted/manual-inspection statements describe the implementation checkpoint before this finalization.

Fresh checkpoint validation: **271 passed in 185.56 s**, no skips (41 V1 / 42 V2 / 71 V3 / 46 V4 Phase 1 / 71 V4 Phase 2). Repository Ruff, all seven changed-code format checks and staged/unstaged whitespace checks passed. All 1,242 inventoried existing result/runtime files, including the original 1,032 protected files and Phase 2 acceptance/diagnostic traces, remain hash-identical. The reviewed 13-file checkpoint contains only source, tests, example and documentation; no runtime/model/cache/environment/credential files are included. The physics audit confirms measured-transform placement, final-physical-pose relation checks, actual Panda actuation and natural dynamics, with no object-state writes or fake attachment/freezing/velocity mechanisms.

## V4 Phase 3 — integrated language manipulation (2026-10-03)

Branch `v4-phase3-physical-ai-agent`, base/main merge `f7d6d801eeec20bb3357f20f0cf829b5828b6d4d`; frozen Phase 2 checkpoint `2fbfe00493e9cf57f0a03cc457ef2440226ad448` is an ancestor. Clean baseline **271 passed in 180.54 s**. First complete implementation regression: **317 passed in 256.91 s**. Counts: 41 V1, 42 V2, 71 V3, 46 V4 Phase 1, 71 V4 Phase 2, **46 new Phase 3**. Ordinary tests use fakes for model inference and real MuJoCo where PANDA_MODEL is configured; no Qwen weights are needed by pytest.

New coverage includes bounded instruction parsing, ordered task decomposition, fixed-action honesty, calibrated read-only observations, known-scene/VLM grounding, missing/wrong semantic targets, invalid destinations, goal/scene cache invalidation, shared fake backend loading, bounded fresh-view recovery, physically verified pickup, actual high-level MuJoCo search override, continuous physical two-operation execution, one viewer object, failure stop and pending later operations. Frozen Phase 1/2 recovery/contact/placement tests remain unchanged.

### Designated end-to-end acceptance

Every requested trial is retained:

- Deterministic manifest: `robotics_runs/20261003T024553_agent_validation_9adb9d823917/validation.json`: single cylinder next_to cube **1/1**, continuous language cylinder-then-cube **2/2**.
- Real Qwen manifest: `robotics_runs/20261003T024542_agent_validation_55e9924996d6/validation.json`: same single **1/1**, continuous multi-step **2/2**.
- Real GUI smoke manifest: `robotics_runs/20261003T024630_agent_validation_b7955ecc19a7/validation.json`: deterministic single and continuous multi-step **2/2**; one viewer per entire task, no speculative branches animated. Normal CLI keeps final scene open.

Both modes execute actual frozen physical skills; corresponding final metric results are deterministic under this configuration. No physical threshold, grasp geometry, contact/friction/servo parameter or historical output was changed.

| New integrated acceptance metric | Cylinder placement | Subsequent cube placement |
|---|---:|---:|
| Placement position error | 0.030927 mm | 0.123368 mm |
| Orientation error | 0.000114683 rad (axis tilt) | 0.000431900 rad |
| Carry travel | 181.854 mm | 195.805 mm |
| Minimum free-transport table clearance | 90.337 mm | 99.049 mm |
| Maximum positional retention drift | 13.603 mm | 3.482 mm |
| Maximum axis/full retention rotation | 0.004916 / **0.724041 rad** | 0.040713 rad |
| Supported/released/stable/upright | Yes | Yes |
| Final linear speed | 0.0001433 m/s | 1.78e-10 m/s |
| Final angular speed | 0.005950 rad/s | 1.44e-8 rad/s |
| Executed substeps per operation | 16,675 | 17,025 |
| Forbidden contacts observed | 0 | 0 |

Cylinder relation at its completion: actual horizontal center distance **79.993881 mm**, surface gap **39.989184 mm**, overlap **0**, verified inside [30,50] mm. MCTS chose the +X candidate in these finite-budget runs; there is no guaranteed minimal-cost/optimal-search claim. The later cube move intentionally changes the earlier relation. Final representative centers: cylinder `(0.579994,-0.079992,0.374971)` and cube `(0.460111,0.099955,0.369972)` m.

All continuous tasks preserve exact operation-end/next-start integration, robot and object records. Panda, scene and object resets: **0**. For the six designated headless acceptance tasks (ten executed operations): **168,150 executed physics substeps**, **2,119,408 intended contact observations**, **0 forbidden contacts**, maximum penetration **0.678795 mm**. Separately, their twelve speculative skill branches accounted for **200,250** physics substeps and **1,396,512** intended contact observations, also with no forbidden contacts. Speculative snapshot restoration is not counted as real task execution or a robot reset. GUI/diagnostic/regression runs are additional separate manifests.

### Real semantic model and honest failures

Cached `Qwen/Qwen3-VL-4B-Instruct`, CUDA:0/bfloat16, **one shared model load** across the three designated VLM tasks. Load **3.949631 s**; **18 physical semantic calls**, total inference **53.199199 s**, mean **2.955511 s**. Per-task calls: **4 / 7 / 7**. No visual-policy calls or speculative inference. Total task times: **27.400 / 49.697 / 49.652 s** under concurrent validation load. Model diagnostics are cumulative for that shared session; new task traces additionally record the starting counters and per-task deltas.

Each multi-step task's first post-cube grounding at the 45-degree front view selected the wrong object region despite high confidence. The agent did not trust it. One new 25-degree observation resolved the cube, leaving all physical thresholds unchanged. Model descriptions also misinterpreted a shadow as a liquid-like substance; such prose never controls geometry or verification. These semantic limitations are not action-policy improvement evidence.

Earlier failed attempts remain: `robotics_runs/20261003T023716_manipulation_agent_8430a6879ab3/` (physical single placement succeeded, post-placement semantic binding failed) and `robotics_runs/20261003T023927_manipulation_agent_9ca7916ef412/` (both physical operations succeeded, final semantic binding exhausted the earlier views). The read-only camera/grounding diagnostics are retained separately. These failures motivated a bounded alternate view, not tolerance weakening or geometry leaked to Qwen. All nine Phase 2 diagnostic failures and original Phase 1/2 measurements remain unchanged.

### Genuine MCTS override and controlled failure

`robotics_runs/20261003T025103_agent_override_f7b17e4bcf0d/validation.json` records a physical setup move of the cube to `(0.54,-0.08,0.37)`. A synthetic prior assigns +X probability **0.99**, −X **0.01**. Actual MuJoCo skill transitions verify both placements. Measured carry paths: −X **167.806815 mm**, +X **202.803085 mm**. Existing MCTS (60 simulations, test c_puct=0.005, one full-skill horizon) selects −X with **59 versus 1 visits**, saving **34.996270 mm**. The selected action is then executed and physically verified in the live scene. Snapshot equality proves search preserved live state. This is a physical cost override, not an invented symbolic transition or a claim about Qwen priors. Default manipulation search c_puct remains 0.05 with uniform priors.

An invalid outside-workspace destination fails structurally before physics; the second operation stays pending and the complete physical snapshot is unchanged. The unique failure task trace records its reason/state. Unit tests also cover missing perception, exhausted single-view recovery, failed skill stop, and successful fresh-view recovery. Frozen Phase 2 tests continue to prove bounded physical grasp retry and fatal-drop/collision handling.

Old command smokes passed: pickup cube, pickup cylinder, cube absolute pick-place, continuous rearrange-demo, and real-Qwen V3 cylinder semantic approach (`robotics_runs/20261003T024853_6832b0e80a2a/`). Existing V3 visual-policy behavior, upward bias, alpha=0.5/c_puct=0.05 and MCTS-recovery findings are untouched. Manipulation visual-prior extension/experiment was not performed; the Cartesian policy is not misapplied to skill actions.

Known limits remain: explicit privileged known-scene metric calibration, bounded grammar, upright cube/cylinder, deterministic geometry, finite contact-sensitive cylinder retention, waypoint transport, bounded recovery, no general collision planner and no hardware-safety claim. Historical Phase 2 13.780 mm positional drift and 0.749 rad full rotation versus 0.0165 rad axis tilt remain preserved. Phase 3 stays uncommitted/unpushed for manual acceptance.

### Final model-protocol and GUI audit

A subsequent final-source regression passed **317 tests in 250.95 s**. Three additional model-protocol cases now exercise invalid JSON, missing required fields and an explicit unknown object through the actual shared backend adapter. Each exhausts at most one fresh-view retry, loads the fake backend once, issues no physical motion and leaves later operations pending. The final suite therefore contains **49 Phase 3 cases** (320 total); the complete final run is recorded below.

Real-Qwen GUI smoke also completed both single and continuous multi-step tasks: `robotics_runs/20261003T025245_agent_validation_674944ab2f2f/validation.json`, **2/2**, one shared backend load, per-task semantic calls **4 / 7**, exact state continuity. These final-source traces include explicit per-task inference deltas. Automated viewer smokes do not substitute for the user's pending manual acceptance.

Controlled invalid-destination acceptance: `robotics_runs/20261003T024941_manipulation_agent_f8cb6f31584a/task_trace.json`; structured failure, second operation pending, complete physical state unchanged. All 1,242 previously inventoried historical/runtime files and all 92 protected source/test files remain byte-identical; modified historical guides only append the new Phase 3 sections.

**Final complete suite: 320 passed in 214.54 s** — 41 V1, 42 V2, 71 V3, 46 V4 Phase 1, 71 V4 Phase 2 and 49 V4 Phase 3. Repository Ruff, all eight changed Python format checks and `git diff --check` pass. Staging is empty; no tracked runtime/model artifacts were added. Branch/main HEAD remain the original Phase 2 merge; Phase 3 is left as source/test/example/documentation working-tree changes.

### Phase 3 manual acceptance and finalization

The user manually validated all four integrated viewer demonstrations: deterministic single-step, real-Qwen single-step, deterministic multi-step and real-Qwen multi-step. Physical grasp, lift, transport, placement, release, retreat and continued execution were observed without visible teleportation, fake attachment, scene reset or object reset. Qwen supplied semantic grounding and deterministic geometry supplied metric manipulation. The earlier semantic misidentifications and bounded alternate-view recovery remain documented; prior acceptance metrics, visual-action bias and Phase 2 cylinder drift findings are unchanged.

Authorized checkpoint: `Complete V4 Phase 3 physical AI agent`, normal SSH push of `v4-phase3-physical-ai-agent` only. Main remains `f7d6d801eeec20bb3357f20f0cf829b5828b6d4d`; no Phase 3 merge or V5 work. Earlier uncommitted/pending-acceptance statements record the prior implementation stage.

Fresh finalization suite: **320 passed in 272.49 s**, no skips, with counts independently confirmed as 41 V1 / 42 V2 / 71 V3 / 46 V4 Phase 1 / 71 V4 Phase 2 / 49 V4 Phase 3. Repository Ruff, all eight changed-code formatting checks and staged/unstaged whitespace checks pass. The cylinder next_to regression also passed again (`robotics_runs/20261003T224953_pick_place_653eb286cf7e/`); previous pickup, absolute placement, continuous rearrangement and real-Qwen V3 semantic approach acceptance traces were verified. All 1,242 protected historical/runtime files and 92 frozen source/test files remain hash-identical. The 16-file checkpoint contains only source, tests, example and documentation; no runtime/model/cache/image/environment/credential files are staged.
