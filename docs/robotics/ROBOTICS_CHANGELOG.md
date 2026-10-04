# V2 robotics development log

## 2026-09-30 — Phase 2 core toolkit — COMPLETE

Repository inspection: existing package has immutable deterministic environments, generic policies, eager PUCT expansion, injected evaluation, experiment CLI, dataclass configuration, and nine historical test modules. All pre-existing implementation/tests/scripts/results are V1 historical material, including experiment files named v2/v4. Git status was initially clean. `results` contains only `.gitkeep`; no historical numerical reports are available in this checkout.

Baseline: system pytest executable absent; system Python 3.10 cannot import datetime.UTC. Local Python 3.12 with inherited ROS pytest plugins also failed before collection. Isolated optional-dependency installation and `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` resolved environment issues. Baseline 41 passed; Ruff clean.

Added: simulator-independent mutable environment and task contracts, immutable search observations/snapshots, adapter preserving live simulator state, and receding-horizon planner using the original MCTS unchanged. Added optional MuJoCo backend with full integration snapshots, restore recomputation, reset/keyframes, bounded position IK controller, composable actions/reach task, and Panda factory. Added uniform/Laya/Mica policy selection through existing integrations, planar second configuration, separate headless CLI and example, unique structured robotics logs, tests, and conceptual/practical documentation.

Changed: packaging adds optional robotics dependencies and a separate console entry point. README leads with toolkit usage while retaining historical research content. Gitignore excludes robotics runtime logs. No features removed, no original tests changed, no historical output files changed, no search/policy refactor.

Bugs found/fixed: NumPy scalar membership against MuJoCo enums required explicit integer conversion. Planar reaching revealed servo gravity sag. Runtime body_gravcomp mutation did not update compiled gravity-compensation metadata; recompiling MjSpec with compensation configured fixes this. Controller IK is computed in scratch data, and only actuator controls drive live physics. Forward recomputation may update integration inputs; restore reapplies captured inputs afterward.

Design decisions: single-agent player convention already exists in V1; no new MCTS implementation. Task evaluation at finite-horizon leaves bounds search. Physics model/configuration is fixed during search; snapshots bind to a backend owner. Task and controller state need snapshots when mutable. External Menagerie is referenced, not vendored. Existing service adapters remain optional and no network inference is required by tests.

Validation: complete suite 56 passed (41 historical + 15 robotics); robotics with external Panda 15 passed; robotics without model path 5 passed/10 explicit skips. Four standalone headless episodes succeeded, covering three 3D targets and planar reach. Formatting and Ruff passed. Detailed commands and artifacts are in TEST_REPORT.md.

Known limitations: position-only reaching, no orientation constraints/collision-aware planner, local short-horizon examples rather than broad workspace guarantee, scalar joint position-actuator controller, no concurrent use of one backend, no real-service inference validation, and no hardware/perception/ROS. Diagnostic simulator warnings and external callback state are not automatically snapshotted/logged. Runtime artifacts are local ignored files.

Next logical step: extend user-defined tasks/controllers against the existing contracts, add contact/task-specific failure criteria when needed, and validate wider reachable targets before a perception or hardware phase. No Phase 3 features were implemented.

Handoff files added: `ddm_mcts/robotics/{__init__,core,mujoco_backend,controller,reach,run,cli}.py`; `tests/robotics/{test_core,test_mujoco}.py`; `examples/robotics/panda_reach.py`; `docs/ARCHITECTURE_AND_CONCEPTS.md`; `docs/robotics/{README,TEST_REPORT,ROBOTICS_CHANGELOG}.md`. Files modified: `.gitignore`, `pyproject.toml`, `README.md`. V1 source/results unchanged.

Final API refinement: run recorder handles generic dataclass/primitive state/action representations and optional backend/controller metadata. Dependency-free line-world episode logging is tested. Final suite 56 passed; explicit robotics 15 passed; core-only fresh install 44 passed/1 optional module skipped. Final standalone rerun retained as `robotics_runs/20260929T235915_a17b1fd075cb`. Integrity audit: 49 protected tracked files match HEAD exactly. Final status COMPLETE.

## 2026-09-30 — Phase 2 visual inspection/debug mode — COMPLETE

Added optional `--viewer` and `--viewer-speed` to the existing robotics CLI; headless remains default. New `visual.py` observes the same ControlledRobot/planner/controller/backend. Viewer-specific code stays out of MCTS and the generic environment. The observer has isolated model/data copies, a visualization-only target sphere, sensible initial camera, approximately 60 Hz presentation, per-decision progress, and a frozen final scene until normal window closure or terminal Ctrl+C.

Backend changes: execution-scoped optional physics-step observation. Without a callback the existing batched physics call is retained. With a callback the same number of individual physics steps run; no extra live forward pass, altered control command, or animation interpolation is introduced. The run loop enables the callback only around the selected action's `world.step`, so speculative search never publishes. Structured logging remains and adds final state/total planning time; interrupts preserve a summary.

Found/fixed during validation: render geometry uses float32 coordinates, so marker assertions use a rendering-appropriate 1e-7 m tolerance. Real WSL initialization/reaching succeeded but the first shutdown segfaulted: MuJoCo passive close requests exit asynchronously on a daemon rendering thread. Shutdown now requests closure and joins threads created during launch before native resources/interpreter teardown. Two later real-viewer runs exited cleanly, including the exact CLI and Ctrl+C after success.

Tests: 8 new GUI-free tests cover CLI/default parsing, callback count/physics equivalence/cleanup, invisible search, isolated marker/model/data, headless-versus-observed exact action/state/root-statistics equivalence, progress, CLI routing, early closure, render-thread teardown, and interrupt logging. Final complete suite 64 passed; robotics 23 passed; historical V1 separately 41 passed. Ruff, new-file formatting, and git diff whitespace checks pass.

Real WSL smoke: target (0.5945, 0.02, 0.6245), default uniform/60 simulations/horizon 3/1x execution. Selected MOVE_X_POS, MOVE_X_POS, MOVE_Y_POS; final error 0.0023 m. Clean automatic-close run: `robotics_runs/20260930T000832_c2dd178c1acd`. Exact CLI kept final scene open and exited cleanly on SIGINT with expected code 130: `robotics_runs/20260930T000941_c53b3277af3b`. Headless smoke also succeeded: `robotics_runs/20260930T000940_dbfbb8714411`. No GUI/process left running.

Files changed for this stage: `ddm_mcts/robotics/{cli,mujoco_backend,run}.py`; added `ddm_mcts/robotics/visual.py` and `tests/robotics/test_visual.py`; updated `docs/robotics/{README,ROBOTICS_CHANGELOG,TEST_REPORT}.md`. No existing V1 file changed in this stage; all 49 protected historical tracked files still match HEAD. No features removed or Phase 3 features implemented.

Limitations: viewer observes execution only; GUI physics/pause/reset widgets do not control the autonomous planner. While MCTS thinks, the pose stays still; closing during search is handled when the search returns. The final pose does not continue integrating. Viewer requires display support; automated tests use fake handles. Next step: personal visual inspection using the documented command, then decide the next phase.

## 2026-09-30 — Phase 3 perception-aware toolkit and final acceptance audit — COMPLETE

Baseline: clean main at `65009c4a934d19fc45637e41e61f1e9c64fcf3f9`, complete suite 64 passed (41 V1/23 Phase 2). Finalization branch: `phase3-perception-toolkit`; main remains the known-good Phase 2 checkpoint. Phase 3 is committed/pushed separately for manual inspection, without merge or history rewrite.

Added: observation/provider contracts; explicit ground-truth observations/perception; independent MuJoCo RGB camera with calibration; deterministic connected-color/known-plane perception; structured WorldState/semantic goals; task resolution; reusable closed-loop PhysicalAgent facade; optional injected VLM-to-state and direct visual-choice/VLDM protocols; real colored-target scene; semantic camera/ground-truth CLI options; observation/diagnostic logs and opt-in PPM images; developer guide and acceptance utility.

Changed: existing planner accepts perceived root context and a predicted-observation projection while physics remains snapshot-backed. Existing execution loop integrates optional agent hooks and timing without duplicating MCTS or IK. Viewer updates the selected perceived marker and terminal context, still publishing selected execution only. Original no-option Phase 2 path remains. README now introduces the toolkit; architecture guide preserves V1 math/history while explaining observations, geometry, structured/model distinctions and closed-loop behavior. No features removed.

Design decisions: visual perception receives RGB/calibration and explicit simulator robot proprioception, never target coordinates. MuJoCo dynamics snapshots initialize physical rollout state; perceived entities remain fixed within one static-scene search. Ground truth is optional post-inference diagnostics only. Plane height is explicit calibration; no arbitrary RGB depth recovery is claimed. Live RGB direct visual priors are root-only, with structured/uniform deeper priors. Offscreen sensors and passive display use separate model/data copies. Agent reset supports semantic switches with the same components and clears tracking history; completed runs are detached from later observations.

Bugs found/fixed: camera projection field naming differs by MuJoCo version; older 3.3.7 bindings do not expose mj_copyData. Version-compatible camera validation and official full integration-state copy/recompute/reapply solve both, with exact input-preservation tests. Optional coordinate goals needed null-safe serialization. Later manual observations could otherwise append to a completed run; run detachment prevents that. Errors cannot report success merely because the preceding state was near the goal. Diagnostic logs now include explicit true entities separately from estimates. Test-scene materials need matte/emissive surfaces for stable generated-color assumptions; this is documented scene configuration, not privileged perception.

Acceptance hardening: poisoned privileged entity/task fields do not change perception, and initially occluded selected targets fail rather than being fabricated. Instrumented exact event order proves observe/perceive/plan/execute-one/re-observe. A small utility runs three scenes, switching red then blue with the same agent/planner/policy/perception/controller. Six accepted configurations succeeded, covering both signs of X/Y and modest distance variation, with 5–7 actions, maximum selected-target perception error 5.100 mm, and true final reach errors 3.535–11.287 mm. Persistent acceptance.json records accompany each unique run. Stress failures from startup/persistent occlusion are retained and documented; six-frame static tracking is intentionally bounded.

Tests: 19 new Phase 3 cases; final full suite 83 passed (41 V1 + 23 original Phase 2 + 19 Phase 3). Separate regression suites pass. Prior 82-case full suites passed on both MuJoCo 3.14.0 and 3.3.7; core-only optional dependency validation passed 46 tests with four MuJoCo-module skips. Ruff, formatting and whitespace checks pass. Final details and acceptance measurements are in TEST_REPORT.md.

Runtime validation: original Phase 2 CLI/headless remains successful. Camera red/blue succeeds headlessly, including EGL without display variables. Real WSL viewers for Phase 2/red/blue initialize, reach, hold final scene and close cleanly on SIGINT (expected exit 130). No GUI/process left running. Automation does not substitute for human assessment of appearance; personal inspection remains recommended. Tests confirm search invisibility, zero speculative camera observations and identical physical transitions with visual annotations.

Files added: `ddm_mcts/robotics/{observation,representation,perception,camera,color_perception,physical_agent,visual_policy,visual_scene}.py`; `tests/robotics/{test_observation,test_perception_camera,test_physical_agent}.py`; `examples/robotics/{visual_reach,validate_visual}.py`; `docs/robotics/PHYSICAL_AI.md`.

Files modified: `ddm_mcts/robotics/{__init__,core,mujoco_backend,controller,reach,run,cli,visual}.py`; `README.md`; `docs/ARCHITECTURE_AND_CONCEPTS.md`; `docs/robotics/{README,ROBOTICS_CHANGELOG,TEST_REPORT}.md`. MCTS/search/policy core and original tests are unchanged. All 57 old runtime/result artifacts and 51 protected tracked source/test/script/result files match baseline. No generated logs/images, Menagerie, virtualenv, caches, credentials or models are committed.

Limitations/deferred: static colored planar targets, calibrated perspective camera, simulator proprioception/dynamics initialization, bounded tracking, sequential GL use, local position-only reach and no hardware/collision-safety guarantees. Model boundaries are mock-tested, with no actual VLM/VLDM backend. No ROS, real robot, RL/training, large downloads, arbitrary 3D perception, cloud/dashboard, grasping or new phase added.

Next action: manual visual inspection of the pushed Phase 3 branch before any merge. Exact original Phase 2, red/blue viewer and headless commands are in PHYSICAL_AI.md. Runtime outputs remain unique under ignored robotics_runs/.

## 2026-10-01 — V3 Phase 1 real local VLM perception — COMPLETE, uncommitted

Baseline: clean `v3-phase1-vlm` at merged V2 checkpoint `fa9e489`; complete suite 83 passed. Inspected original observations/perception, world state, PhysicalAgent, tasks/scenes, CLI, policies, original MCTS, docs/tests and packaging. Local GPU is RTX 4090 Laptop, 16 GB. Torch/Transformers were absent; installed optional torch 2.7.1+cu128/torchvision 0.22.1+cu128, transformers 4.57.6, accelerate 1.15.0, Pillow 12.3.0. Existing MuJoCo/numpy preserved; dependency check passes.

Added: lazy once-loaded official Qwen3-VL backend, validated JSON semantic/relative image grounding schema, deterministic grounded foreground/plane localization, shared-material cube/sphere/cylinder scene, configurable semantic caching/refresh and bounded tracking, standalone real inference and three-goal physical validation utilities, opt-in live input/annotated-grounding debugging, optional VLM extra, CLI semantic-reach/VLM configuration, dependency-light and fake-runtime tests, and practical LOCAL_VLM.md guide.

Changed: WorldState gains an optional resolved_target_label so language goals can resolve entities without matching simulator names. PhysicalAgent/run logger add perception diagnostics/debug hooks; original policy, MCTS, simulator/controller implementations are reused unchanged. Existing ground-truth/color/coordinate modes remain. Conceptual/practical docs extend V1/V2 history with actual implemented VLM semantics -> deterministic geometry -> structured state -> policy -> MCTS -> IK. No features removed.

Design: Qwen chooses which visible object and returns label/description/box, never robot commands or guessed XYZ. All three shapes share one teal material; common masking only refines a VLM-selected box. Geom identifiers are anonymous; semantic mappings exist only in explicit ground-truth diagnostics. Metric localization uses existing calibrated ray-plane geometry and known center-height plane z=0.6245 m. Model confidence is diagnostic self-report. Perception reset clears grounding, not the model; default refresh 0 reuses static semantics while fresh images/metric checks and receding-horizon search continue after every action. Invalid/missing/expired/ambiguous perception fails without privileged fallback.

Found/fixed: an overly broad foreground mask included blue floor and biased estimates. Tight tolerance and a blue-background regression fixture fix this; final error evidence is in TEST_REPORT.md. Initial camera partly hid cylinder; a clearer higher oblique pose improved grounding. Greedy generation clears incompatible sampling flags. Natural-language goal/entity mismatch needed explicit resolved labels for task/diagnostic accounting. A mock DDM test fixture needed the existing required objective argument; no DDM refactor. Load and inference times are recorded separately, including failure diagnostics.

Real validation: only Qwen/Qwen3-VL-4B-Instruct downloaded to normal external Hugging Face cache; snapshot ebb281ec70b05090aa6165b016eac8ec08e71b17. CUDA/bfloat16 standalone inference succeeded before control integration. Final same-components cylinder/cube/sphere episodes all succeeded with one model load (7.941 s), three semantic inferences (3.393/2.692/2.695 s), 7/7/8 actions and fresh observation counts 8/8/9. Maximum localization errors 2.534/6.426/5.941 mm; true final center errors 6.782/8.887/5.999 mm. Actual model raw responses are retained in unique run artifacts. Original Phase 2 and both V2 color goals still succeed.

Viewer: real natural-language cylinder and cube runs reached, held final scene and closed cleanly on controlled SIGINT (expected 130). Fresh frames never contain viewer-only selected-target markers. Only selected actions are displayed; speculative MCTS branches remain invisible. No orphan GUI/process remains. Automated smoke validates operation; human assessment of visual appearance remains recommended.

Tests: final complete 105 passed (41 V1 + 42 V2 + 22 new V3), no skips with Panda/EGL. Independent regressions 41/42 pass. Core-only optional-dependency validation 62 passed/5 optional module skips; no core Torch/Transformers requirement. New tests cover schema/boxes, lazy loading, image/goal conversion, dependency/OOM errors, official processor plumbing with fakes, semantic cache/geometry/poisoned truth, uniform/mock-DDM physical search, logs/debugging and CLI. Real model checks are explicitly invoked utilities, not part of normal pytest. Ruff/format/whitespace pass.

Files added: `ddm_mcts/robotics/{vlm_backend,vlm_perception,semantic_scene}.py`; `examples/robotics/{vlm_sanity,validate_vlm}.py`; `tests/test_vlm_semantics.py`; `tests/robotics/test_vlm.py`; `docs/robotics/LOCAL_VLM.md`. Files modified: `pyproject.toml`, `README.md`, `ddm_mcts/robotics/{cli,physical_agent,representation,run}.py`, `docs/ARCHITECTURE_AND_CONCEPTS.md`, `docs/robotics/{README,PHYSICAL_AI,ROBOTICS_CHANGELOG,TEST_REPORT}.md`.

Integrity: 235 pre-existing runtime/result files and 54 protected historical source/test/script/result files unchanged. No weights/model assets, caches or run/image outputs tracked. Current branch remains v3-phase1-vlm; NO COMMIT, NO PUSH, NO MERGE. Stable main untouched.

Known limits: static generated shapes/common material, known center-height plane and calibrated perspective camera, finite silhouette/occlusion errors, eight-frame bounded tracking, model semantic errors despite valid JSON/confidence, explicit simulator proprioception/dynamics state and no hardware/collision-safety guarantee. GPU acceptance validated; CPU performance not. No V3 Phase 2, direct Qwen action scoring, VLDM training, RL, grasping, hardware/ROS, additional models, cloud APIs or dashboards.

Next step: user manually runs standalone and cylinder/cube/sphere viewer commands in LOCAL_VLM.md before authorizing any checkpoint. Logs remain unique under ignored robotics_runs/; model cache stays under ~/.cache/huggingface/hub/.

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


## 2026-10-03 — V3 Phase 3: unified multi-step physical-AI agent

COMPLETE, intentionally uncommitted on `v3-phase3-agent`; main remains the 2d13da0 checkpoint. No commit/push/merge or V4 work.

Added PhysicalTask/ApproachGoal, a bounded deterministic Approach/Visit/Go-to shape-list parser, OrderedTaskRunner/TaskResult/ApproachVerifier, and PhysicalAgent.run_task. The runner reuses the original closed-loop agent/MCTS/controller, tracks ordered status, stops on failed verification, and writes compact task traces linked to original subgoal logs. Physical robot/scene/camera state persists; full snapshots and state continuity are checked. Subgoal action budgets extend cumulative step bounds instead of resetting counters. Existing search already creates a new tree per decision; no search rewrite was necessary.

Verification checks actual TCP at the final outside-object waypoint, semantic identity, optional true requested-object waypoint and all-object signed gripper clearance. True geometry/entities are evaluation only and never repair perception or leak into Qwen. The final-waypoint check prevents the intermediate lift or a false episode-success flag from advancing the task; the true-object check rejects a mislabeled/mislocalized estimate. Clearance checks scope to selected physics execution, including transitions. Nested backend callbacks now compose clearance monitoring and viewer synchronization; no physics/controller behavior changed.

Goal-dependent grounding/tracking and visual caches refresh, while one lazy Qwen backend remains loaded and shared. Added cumulative logical perception counts and separate physical semantic inference/cache timing; per-task/per-goal differences remain distinct from shared model totals. Multi-step CLI uses --task multi-semantic-reach --instruction, preserving single-target commands. One viewer spans the entire sequence and holds the final scene only once. A small explicit real acceptance example demonstrates programmatic composition and bounded viewer cleanup.

Baseline 127 passed in 72.05 s; final 154 passed in 126.50 s: 41 V1 + 42 V2 + 26 Phase 1 + 18 Phase 2 + 27 Phase 3. Real Phase 1 cylinder/sphere/cube: 17/9/6 actions, TCP errors 5.12/4.71/6.09 mm, selected-object clearances 29.35/40.49/40.41 mm, one model load and three semantic calls. Real Phase 2 same order: 32 visual calls + 3 semantic calls, one model load, same 17/9/6 actions, 22 MCTS overrides. Qwen top-1 remains upward on all 32 steps. Alpha=0.5 and visual c_puct=0.05 remain; Phase 2 c_puct=1.4 failed results were not modified. Second real order cube/cylinder succeeds without reset. Both real continuous-viewer smoke paths completed and closed cleanly. Final public single-goal coordinate/red/blue/VLM/visual commands pass. Detailed timing, errors, clearance, raw-output references and acceptance checklist are in TEST_REPORT.md.

Ruff lint, nine changed/new Python files' formatting and whitespace pass. All 802 protected pre-existing outputs remain unchanged. Controller/TCP/approach/scene/MCTS code and all old tests remain intact. Weights are still external; runtime/cache/image/virtualenv artifacts remain ignored and unstaged. No dependencies were added.

Files added:

- `ddm_mcts/robotics/ordered_task.py`
- `docs/robotics/MULTI_STEP_AGENT.md`
- `examples/robotics/validate_multi_step.py`
- `tests/robotics/test_ordered_agent.py`
- `tests/test_ordered_tasks.py`

Files modified:

- `README.md`
- `ddm_mcts/robotics/__init__.py`
- `ddm_mcts/robotics/cli.py`
- `ddm_mcts/robotics/mujoco_backend.py`
- `ddm_mcts/robotics/physical_agent.py`
- `ddm_mcts/robotics/vlm_perception.py`
- `docs/ARCHITECTURE_AND_CONCEPTS.md`
- `docs/robotics/LOCAL_VLM.md`
- `docs/robotics/README.md`
- `docs/robotics/ROBOTICS_CHANGELOG.md`
- `docs/robotics/TEST_REPORT.md`
- `docs/robotics/VISUAL_DECISION.md`

Known limits retained: simple shape scene/object set and bounded grammar; calibrated-plane localization and configured extent; position-only safe approach, not grasping; no general collision planner or hardware safety claim; persistent upward visual bias; successful recovery does not prove Qwen improves search. Deferred: manipulation/grasping, training, another model, cloud/ROS/hardware, unrestricted language planning, RL and V4. No speculative extensions were implemented.

Manual commands and API: MULTI_STEP_AGENT.md. Recommended next action: inspect the final perception and visual-policy multi-step viewer on this branch before checkpointing.

### Final Phase 3 acceptance checkpoint

The user manually accepted both continuous cylinder → sphere → cube viewers: ordered completion, persistent Panda state, continuous transitions, safe approach and TCP waypoint marker. Fresh finalization validation: **154 passed in 163.17 s** (41 V1, 42 V2, 26 Phase 1/approach, 18 Phase 2, 27 Phase 3); Ruff, changed-code formatting and whitespace checks passed. All 802 inventoried historical files remained unchanged; generated artifacts and external Qwen weights are excluded from the checkpoint.

Recorded model findings are preserved exactly: all 32 visual top choices were MOVE_Z_POS; MCTS overrode 22/32; alpha=0.5 and visual c_puct=0.05. The c_puct=1.4 failure remains documented. This demonstrates search recovery rather than superior model action reasoning. No approach/TCP behavior, previous results, dependencies or scope changed during finalization. The user authorized commit `Complete V3 multi-step physical AI agent` and a normal push of `v3-phase3-agent` only; no merge to main or V4 work.

## 2026-10-03 — V4 Phase 1: six-DoF physical pickup

Baseline: clean `v4-phase1-grasping` from main `a8f0717`; **154 passed in 135.84 s** before edits. Existing V3 modules/results/tests are preserved; only a distinct pickup dispatch is added to the existing CLI.

Added:

- `pose_control.py`: validated wxyz pose targets, world-frame SO(3) error, both TCP Jacobians, weighted bounded DLS, quaternion trajectories, real actuator execution. Position-only controller unchanged.
- `gripper.py`: actual Panda tendon actuation, bounded width commands and measured finger state.
- `contacts.py`: geom/body/force/distance records and phase-aware contact rules.
- `grasp.py`: distinct object/pre-grasp/grasp/lift representations, upright box-face and cylinder-radial grasps, independent bilateral-contact and lift/hold verification.
- `manipulation_scene.py`: separate gravity-driven free-joint objects, support table and explicit pickup-only compliance/servo configuration.
- `manipulation.py`: ordered physical pickup stages, safe transit, live substep monitoring, failure stop, unique structured diagnostics and object/TCP transforms.
- `manipulation_visual.py`: existing display-copy viewer publication/cleanup reused for actual pickup motion and final inspection.
- `examples/robotics/validate_pickup.py`, `tests/robotics/test_manipulation.py`, and `docs/robotics/MANIPULATION.md`.

Found/fixed: free-body home keyframes initially lacked object poses; initialization now includes them before physical settling. Default cylinder contact/servo compliance allowed slip/drop; task-only pad/object solref is 5 ms, finger stiffness 400 N/m and damping 20 N s/m, retaining force limits and friction. Trial changes and longer-hold sensitivity are documented rather than hidden. External Menagerie files, V3 controller/TCP/approach, MCTS and Qwen remain unchanged. No object weld, attachment, live object teleport, forced velocity or gravity removal is used.

Real acceptance: cube **3/3**, cylinder **3/3** with bilateral finger force, 100 mm lift and one-second checked hold. Cube actual elevation 99.08 mm, relative drift 1.30 mm; cylinder elevation 97.52 mm, drift 2.02 mm. Bounded WSL viewer runs for both completed and closed cleanly. Invalid offset grasp and dropped/unsupported-object tests fail correctly. New tests cover SO(3)/pose control, finite bounds, actual finger motion, contacts/rules, generation, state ordering, physical verification, no reset/teleport, logs, snapshot/restore and viewer lifecycle.

Limitations: known upright cube/cylinder only; deterministic simulator geometry; bounded contact/hold verification; no whole-arm collision planner or real-hardware safety claim; cylinder retention remains contact-sensitive. No placement, learned grasping, new VLM, training, ROS, hardware, RL or V4 Phase 2/3. Qwen's historical upward-bias/MCTS-recovery results are unchanged. Work remains uncommitted/unpushed for manual pickup viewer inspection. Final test counts and runtime manifest locations are recorded in TEST_REPORT.md.

### Phase 1 checkpoint finalization

Authorized checkpoint: `Complete V4 Phase 1 physical grasping`, branch `v4-phase1-grasping` only. Fresh final suite: **200 passed in 134.29 s** (41 V1 / 42 V2 / 71 V3 / 46 V4 Phase 1); Ruff, changed-code formatting and whitespace checks pass. The 15-file manifest contains only Phase 1 source, tests, example and documentation. No code/physics changes were needed during finalization. Recorded real results, servo/contact configuration, cylinder rotation and bounded-retention limitations remain intact. All 960 protected historical files are unchanged; runtime/model artifacts are excluded. Main remains `a8f0717`; no merge or Phase 2 implementation is included.

## 2026-10-03 — V4 Phase 2: physical pick-and-place and rearrangement

Completed the explicitly authorized pending Phase 1 merge on main (`da9884ac0e2d2fbd21702e88ededdbf912215703`, Phase 1 `c05339d` ancestor) and pushed main normally through its existing SSH remote. Created `v4-phase2-pick-place` at that merge; baseline **200 passed in 132.79 s**. Phase 2 changes remain uncommitted and unpushed for manual acceptance.

Added placement/relative-destination representations, rigid transform composition/inversion, measured grasp-relative TCP placement, deterministic transport, substep retention/contact monitoring, verified pre-place, descent to actual support, physical release/retreat, natural settling, independent placement/relation verification, persistent plans and bounded one-retry recovery. Existing pickup executor, grasp strategies, dynamic scene, physics/friction/servo tuning, controller, V3 pipelines and historical tests remain unchanged. CLI adds pick-place and rearrange-demo; API and commands are in PICK_AND_PLACE.md.

Real physics acceptance: cube absolute **3/3**, cylinder next_to **3/3**, continuous two-object rearrangement **2/2**; no forbidden contacts observed. Cube position error 0.267 mm, cylinder 0.132 mm, sequential cube 0.090 mm; cylinder/reference gap 39.878 mm, no overlap. Full integration state persists with zero resets. Real viewer smokes completed both single operations and a continuous plan in one viewer, retaining final-scene hold in normal CLI. A deliberately missed first grasp recovered physically with one retry; invalid destination stopped before physics and skipped the next operation.

Honest diagnostics retained: initial 4 mm transport increments unloaded a finger; 1 mm held motion is used. Cylinder full yaw drift reached 0.749 rad while axis tilt remained 0.0165 rad; positional drift reached 13.780 mm under the existing 15 mm retention bound. Cylinder yaw is physically symmetric and separately logged; it is not rigid attachment. Force unloading is filtered for at most 20 ms only with bilateral pad geometry and valid width/drift, tested against missing/prolonged unloaded contacts. Actual gaps were ≤4 ms. A too-near Y-side placement blocked the second open-jaw grasp; destination candidates reserve that sweep. Failed diagnostics remain in their original unique runtime directories. No object state teleportation, force-following, gravity removal, attachment or artificial settling was introduced.

Known limits: known simple upright objects/table, deterministic geometry and waypoint transport, contact-sensitive finite retention, bounded pre-release retry, no general motion/clutter planner or hardware safety claim. No Qwen manipulation, language integration or V4 Phase 3. Existing V3 upward bias and MCTS-recovery findings are preserved.

Final full regression after the trace audit: **271 passed in 198.74 s** (41 V1, 42 V2, 71 V3, 46 Phase 1, 71 Phase 2). Repository Ruff, seven changed-code format checks and whitespace/artifact checks pass; all 1,032 protected historical runtime/result files remain identical. Phase 2 staging is empty.

### Phase 2 manual acceptance and checkpoint

The user manually validated all three viewer demonstrations: cube placement, upright cylinder next_to cube, and continuous cylinder-then-cube rearrangement with physical release, stable objects and no reset/teleportation/fake attachment observed. Authorized checkpoint: `Complete V4 Phase 2 physical pick and place`, normal SSH push of `v4-phase2-pick-place` only. The measured results and cylinder drift limitations are unchanged; nine earlier development failures remain documented. Main remains the Phase 1 merge `da9884ac0e2d2fbd21702e88ededdbf912215703`; no merge or Phase 3 work. Earlier uncommitted status records the preceding implementation checkpoint.

Fresh checkpoint suite: **271 passed in 185.56 s**, with all five regression groups intact. Ruff, changed-code formatting and whitespace checks pass. The 13 reviewed checkpoint files exclude runtime/model artifacts; all 1,242 inventoried historical/runtime files remain unchanged. No physics or source/test changes were needed during finalization.

## 2026-10-03 — V4 Phase 3: closed-loop language manipulation

Verified clean `v4-phase3-physical-ai-agent` at merged main `f7d6d801eeec20bb3357f20f0cf829b5828b6d4d`, Phase 2 checkpoint `2fbfe00493e9cf57f0a03cc457ef2440226ad448` ancestor. Baseline 271 tests passed. Added bounded ordered manipulation instructions, a composed agent API, fresh known-scene/RGB observations, existing Qwen semantic grounding and calibrated metric binding, real MuJoCo high-level skill futures through existing MCTS, authoritative frozen skill verification, read-only lift/placement checkpoints, cache/tree refresh, state continuity, bounded semantic re-observation and structured task traces.

Frozen pickup/placement implementations, controller, scene contact/friction/servo settings, recovery criteria, all previous tests and historical results remain unchanged. CLI adds manipulate; existing task modes retain their paths. Default high-level priors are uniform; Cartesian Qwen visual priors are not reinterpreted as manipulation decisions. Fixed-position/pickup tasks are honestly forced. New real MCTS proof overcomes a 0.99 inferior prior and saves 34.996 mm of verified carry travel.

Deterministic single 1/1 and multi 2/2; cached real-Qwen single 1/1 and multi 2/2, one model load and 18 semantic calls across those trials. Wrong post-cube grounding recovered through one alternate camera view; failed earlier attempts remain recorded, along with hallucinated shadow descriptions and contact-sensitive cylinder drift. GUI smoke completed single and continuous multi-step tasks. First complete suite 317 passed (46 new tests). No generality or hardware safety claim. No Phase 3 commit/push/merge and no V5 work; manual viewer acceptance remains the next checkpoint.

Final audit: **320 passed in 214.54 s**, including 49 new Phase 3 cases and all 271 frozen regressions. Real-Qwen GUI single/continuous tasks also passed (one shared load, 4/7 task calls); three malformed-response protocol cases fail safely without motion. Ruff, changed-code formatting and whitespace checks pass. No historical source/test/runtime file changed, no artifacts staged, and no Phase 3 commit/push/merge was made.

### Phase 3 manual acceptance and checkpoint

The user manually accepted all four deterministic/real-Qwen single/multi-step manipulation viewer modes, confirming physical execution and continuous state without visible teleportation, attachment or resets. Semantic/metric separation, bounded alternate-view recovery, actual MuJoCo MCTS override and all historical limitations/results are preserved. Authorized checkpoint `Complete V4 Phase 3 physical AI agent`, SSH push of the Phase 3 branch only; main remains the Phase 2 merge and no V5 work is included.

Fresh checkpoint validation: **320 passed in 272.49 s**, all six groups intact. Ruff, eight changed-code format checks, whitespace and artifact audits pass. Cylinder relative-placement regression passed again; existing physical acceptance records and all protected historical files remain unchanged. The reviewed 16-file checkpoint excludes runtime/model artifacts and preserves all measured semantic, physical and MCTS findings.

## V5 — compositional physical reasoning (working tree)

Added ordered predicate goals, bounded goal parsing, observed relationship graphs, support surfaces/affordances, reusable interaction execution, physical waits and actual MuJoCo push-future MCTS with snapshot restoration. Real acceptance: stack 3/3, cylinder toppling 3/3, deterministic headline 3/3, local-Qwen headline 1/1. Added controlled physical failure and misleading-prior evidence. No frozen V4 controller/skill source was changed; only the existing CLI gains additive dispatch. Evidence and limitations: [V5 guide](COMPOSITIONAL_PHYSICAL_REASONING.md).

### V5 final checkpoint audit — 2026-10-04

Recorded user manual viewer acceptance of stacking, pushing/toppling and both
headline modes. Audited reusable goal composition, physical predicates, live
physics integrity, contact compliance and isolated MuJoCo MCTS futures. No new
features or changes to frozen V1–V4 behavior were introduced during finalization.
