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
