# V3 Phase 2: VLM-backed visual decision policy

V3 Phase 3 also supports [ordered physical tasks](MULTI_STEP_AGENT.md) through this same root-only policy, with one shared Qwen backend. It retains alpha=0.5, c_puct=0.05, the upward-bias limitation and the historical failed c_puct=1.4 results documented below. Fresh observations and goal context are rebound at each subgoal and physical decision.

This is a prompted Qwen3-VL adapter, not a robotics VLDM trained by this project. Qwen supplies action priors; the original MCTS searches MuJoCo futures and selects the executed action. Existing perception, controller, TCP and safe approach remain available.

## Run

Use the existing environment and cached model; no new dependencies or model downloads are needed. Set:

```bash
export PANDA_MODEL=/home/ishaan/robot-arm-playground/mujoco_menagerie/franka_emika_panda/scene.xml
```

Standalone real RGB/action scoring (no physical execution):

```bash
HF_HUB_OFFLINE=1 MUJOCO_GL=egl .venv/bin/python -m examples.robotics.visual_decision_sanity --model "$PANDA_MODEL" --vlm-device cuda:0
```

Cylinder visual-policy viewer:

```bash
HF_HUB_OFFLINE=1 .venv/bin/python -m ddm_mcts.robotics.cli \
  --model "$PANDA_MODEL" --task semantic-reach \
  --goal 'Reach the cylindrical object.' --decision-policy visual --viewer --diagnostics
```

Change only the goal to `Reach the cube.` or `Reach the spherical object.` for the other objects. Headless:

```bash
HF_HUB_OFFLINE=1 MUJOCO_GL=egl .venv/bin/python -m ddm_mcts.robotics.cli \
  --model "$PANDA_MODEL" --task semantic-reach --goal 'Reach the cylindrical object.' \
  --decision-policy visual --diagnostics
```

Original Phase 1 viewer, without visual action scoring:

```bash
.venv/bin/python -m ddm_mcts.robotics.cli --model "$PANDA_MODEL" --task semantic-reach \
  --goal 'Reach the cylindrical object.' --perception vlm --viewer --diagnostics
```

Original V2 color viewer:

```bash
.venv/bin/python -m ddm_mcts.robotics.cli --model "$PANDA_MODEL" --task visual-reach \
  --goal red --observation camera --perception color --viewer --diagnostics
```

## Components and information boundaries

`QwenVisualPolicy` implements the existing `VisualDecisionPolicy` root boundary. `QwenVisualDecisionModel.score(rgb, goal, actions)` invokes the same lazy `Qwen3VLBackend.generate` used by semantic perception's `infer` method. CLI creates one backend shared by both adapters. The current observation is bound before every real decision. Deeper MCTS nodes use the configured `--policy` (uniform by default; existing Laya/Mica adapters remain supported), without speculative rendering or Qwen visual requests.

The semantic perception path still resolves the target for the task/evaluator using real RGB and deterministic calibrated-plane localization. It does not put that target's XYZ, simulator name or selected waypoint into the visual-policy prompt. MuJoCo still initializes physics from live integration snapshots and evaluates simulated futures against the resolved structured goal. Images are not stepped as world-model state.

The policy image is a copy of the real camera frame with a small magenta/white dot projected at the current TCP using known calibration and robot proprioception. This reveals only the current gripper TCP, not the target, goal waypoint or correct action. The perception input is unchanged. The red marker in the interactive viewer remains the current TCP approach waypoint and is not included in camera observations. The robot intentionally stops above the object: bounding radius 40 mm plus standoff 50 mm, +Z approach, lift before translation, hand-local TCP offset 103.4 mm. No grasping, closure or contact manipulation is implemented.

Actions retain `MOVE_X_POS`, `MOVE_X_NEG`, `MOVE_Y_POS`, `MOVE_Y_NEG`, `MOVE_Z_POS`, `MOVE_Z_NEG` identity. World +Z is up; X/Y are horizontal. Each prompt gives the actual displacement in meters and projected pixel delta calculated as `project(tcp + displacement) - project(tcp)`. Pixel origin is top-left, X right, Y down. This mapping is local and perspective-dependent, recomputed per observation. It is not a fixed invented image-axis mapping and uses no target geometry.

## Trust, order and parsing

Canonical presentation is lexicographic stable action ID order. Optional `--visual-permutations 2` or `3` reuses V1 PermutationAveragedPolicy, maps probabilities by action identity and averages. Default is 1 because each order costs a real inference. The cache is cleared at each new live image, so future images never reuse old-image priors. Repeated requests within one bound decision can hit the cache.

`--visual-alpha` defaults to 0.5, reusing V1 MixedPolicy: `(1-alpha)/n + alpha*P_visual`. Alpha 0 skips visual scoring (semantic perception still runs); alpha 1 trusts visual priors fully. PUCT strength is separate: `--c-puct` defaults to 0.05 in visual mode because reaching values are negative distance in meters. Other CLI modes retain 1.4. These coefficients are configurable and are not safety guarantees.

The parser accepts JSON, code fences and harmless surrounding prose. It rejects duplicates, missing/unknown action IDs, negative/nonfinite/nonnumeric values, and zero total mass. It normalizes valid nonnegative scores; no missing scores or ground-truth answers are manufactured. Prompt numeric preferences are heuristic, not calibrated action probabilities. Errors stop the run with backend/raw-response diagnostics; there is no hidden repair/fallback.

## Logs and performance

Unique ignored robotics_runs directories contain configuration, observations, steps and summaries. Steps record candidate descriptions/order, raw response, parsed scores, normalized visual probabilities, alpha, actual root priors, Qwen top-1, selected MCTS action, agreement, root visits/Q, physical inference time, visual-policy wall time, MCTS search time and execution results. Configuration records actual MCTS settings and controller parameters. Summary diagnostics separate model load time/count, total shared backend calls, visual logical requests/cache hits/misses, physical visual calls/time/mean, planning/search/execution time. End-to-end decision accounting sums acquisition, perception, planning and execution components; disk/logging overhead is not a profiling claim.

Root statistics independently show the probabilities MCTS consumed. Qwen top-1 can differ from selected action; a deterministic test biases the bad action 100:1 and proves real MCTS chooses the better action. Only selected actions publish viewer frames; the robot remains stationary while Qwen/MCTS plan. CLI keeps the final scene open; close normally or Ctrl+C.

Real-model scoring is optional validation, never a normal test-suite download. `examples.robotics.validate_visual_policy` runs the three physical goals with one model session and writes a small acceptance summary. Use `--viewer` for an automated smoke (it closes after two seconds); normal CLI remains the manual inspection path. Priors can be poor or misleading, particularly the model's tendency to keep lifting. Search correctness still depends on simulator, objective, budget and trust; this is not general collision planning or real-robot safety.

## Final checkpoint acceptance

The user manually validated the cylinder visual-policy viewer and the safe above-object TCP endpoint. VLM perception identifies/localizes the object; the VLM-backed visual decision policy supplies root priors; a structured DDM can supply deeper-node priors; MCTS searches and selects; MuJoCo predicts physical transitions. These are distinct roles, and Qwen is not a robotics model trained by this project.

The upward-motion bias remains documented, without revised results or capability claims. Initial c_puct=1.4 runs failed by over-trusting that prior at the tested budget; visual-mode c_puct=0.05 was needed for search recovery with meter-valued rewards. Alpha remains 0.5. The deterministic MCTS override regression and safe TCP/approach clearance tests are preserved. This checkpoint belongs on v3-phase2-vldm only; finalization does not merge main or start another phase.
