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
