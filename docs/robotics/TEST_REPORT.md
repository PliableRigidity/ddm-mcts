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
