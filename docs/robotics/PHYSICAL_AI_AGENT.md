# V4 Phase 3: language-conditioned manipulation agent

`PhysicalAIManipulationAgent` composes the existing observation/perception contracts, structured world state, generic policy/MCTS, MuJoCo snapshots and frozen V4 pickup/placement executors. It adds a bounded task interpreter and physical-verification-gated closed loop. It does not replace V3's approach agent or V4's deterministic skill APIs.

## Supported language and API

The parser accepts one to four ordered operations separated by `then` (optionally preceded by a comma), case-insensitive, with optional `the` and a final period:

- `Pick up the cube.` / `Pick up the cylinder.` (single pickup only)
- `Place the cylinder next to the cube.`
- `Pick up the cylinder and place it next to the cube.`
- `Move the cylinder next to the cube.`
- `Move the cube to the target location.` / `... to the other target location.`
- `Move the cylinder next to the cube, then move the cube to the target location.`

`Place` means a complete pickup-and-placement operation from the table, not release of a previously held object. Multi-operation instructions require complete pick-and-place operations. Sphere manipulation, self-reference, other relations, arbitrary coordinates in language and unrestricted prose are rejected. `target location` names the calibrated object-center destination `(0.46, 0.10, 0.37)` m; it is not inferred by Qwen. The parser API accepts a caller-supplied target position. Programmatic structured tasks accept validated `PlacementTarget`/`NextTo` destinations.

```python
from ddm_mcts.robotics.manipulation_agent import PhysicalAIManipulationAgent
from ddm_mcts.robotics.manipulation_task import parse_manipulation_task
from ddm_mcts.robotics.manipulation_scene import panda_pickup_scene

world = panda_pickup_scene(model_path)
agent = PhysicalAIManipulationAgent(world)
task = parse_manipulation_task(
    "Move the cylinder next to the cube, then move the cube to the target location."
)
try:
    result = agent.run_task(task)
    # Equivalent: agent.run_instruction(task.instruction)
    assert result.success, result.trace
finally:
    agent.close()
```

Observation, perception, policy, search configuration and skill executor are injectable components. The programmatic policy uses the existing `Policy.probabilities(state, actions)` interface, allowing an appropriately adapted DDM to supply high-level priors. Default CLI priors are uniform. Existing Cartesian visual/DDM adapters are not silently interpreted as manipulation policies.

## Responsibilities and information boundaries

- **Language interpreter:** explicit bounded grammar -> ordered `ManipulationOperation` objects.
- **Observation:** fresh raw RGB or explicit known-scene ground truth, plus simulator robot telemetry. The free camera renders copied MuJoCo data and adds no physical scene geometry.
- **Qwen/VLM:** image + semantic goal -> validated semantic label/description/box. No body IDs, target XYZ, grasp transforms, destinations or correct actions enter the prompt.
- **Metric binding:** reuse V3 calibrated-plane foreground localization, then bind the visual region to a known scene object. The nearest object must match the requested semantic object and be within 35 mm. A wrong region fails; there is no silent ground-truth fallback.
- **Structured world:** simulator-native metric object poses/dimensions are an **explicit privileged known-scene geometry source**, after semantic grounding gates the binding. This is not a vision-only metric manipulation estimator. The calibrated object-height plane is known geometry, not a VLM prediction.
- **Policy/DDM:** action priors only, where configured.
- **MCTS:** choose between geometry-valid placement alternatives using actual physical skill futures.
- **MuJoCo:** contact/friction/gravity transitions; snapshot/restore hypothetical branches.
- **V4 skills:** frozen deterministic grasp geometry, six-DoF control, physical finger actuation, transport, support-triggered release, retreat and natural settling.
- **Physical verifiers:** authoritative grasp/lift/placement/relation success. Model confidence or a command return cannot certify success.

The default camera looks from the table front at 45 degrees. Semantic grounding failure permits **one fresh alternate 25-degree view**, then stops. This changes rendering only, not the robot/object state. It improves visibility and distinguishes dark/shadowed geometry; failures and raw model replies remain logged. There is no unconditional retry or metric fallback.

## Closed loop and state persistence

Before each operation, invalidate goal-dependent perception, acquire a fresh observation, bind target/reference and reconstruct world state. After verified pickup/lift, acquire a read-only checkpoint with robot/scene/contact-based physical skill diagnostics; do not ask Qwen to ground a possibly occluded held object. After supported release, retreat and settling, acquire fresh RGB/grounding again, check consistency with the actual skill end pose and independently verify any requested relation from current physical poses.

The task tracks pending, active, physically complete and verified/failed progress. Failed parsing/perception/planning/execution/verification stops later operations. Successful operations advance from the exact previous physical end state. No Panda, object or scene reset occurs between operations. Backend, camera, viewer and Qwen session persist. Observations, world representations, semantic caches and MCTS trees refresh; physical state remains continuous.

The after-lift checkpoint is a small `PickPlaceAgent` subclass hook that delegates to the unchanged Phase 2 pickup/recovery method before observing. Camera rendering reads copied data and performs no physics step, so it can run between stages while the outer accounting scope remains installed. Speculative skills have no checkpoint hook, model calls or viewer callbacks.

Recovery reuses Phase 2's one-retry pre-release budget. Collisions, drops, invalid destinations, exhausted retries and post-release failures stop safely. Agent-level semantic recovery is separately bounded to one new view per failed checkpoint. There is no automatic post-release regrasp.

## Where MCTS genuinely contributes

A fixed-position operation or pickup has one valid action; execute it directly and log `forced=true`, `simulations=0`. For `next_to`, generate feasible −X/+X alternatives from actual reference pose, directional object half-extents, requested gap and workspace constraints. Preserve Phase 2's narrow-gap jaw-sweep constraint. If only one candidate fits, it is forced.

With two alternatives, the existing `RoboticsPlanner`/`SimulatorAdapter`/PUCT MCTS expands actual frozen pick/place skills in MuJoCo futures. Both candidate transitions physically grasp, carry, release, settle and verify, including independently measured relation. Root children retain those physical outcomes; subsequent simulations reuse the children rather than executing the whole skill per simulation. Default: 60 simulations, horizon **one full skill**, `c_puct=0.05`, uniform priors. The CLI's legacy reach `--horizon` does not set manipulation servo or skill depth.

Reward is `-min(carry_path_m, 1)` for verified success and −2 for failure; verified success dominates, and successful alternatives are compared by measured transport distance. This is a limited physical cost objective, not language-generated physics or a claim of optimal general manipulation. Trees are rebuilt for every operation. Full integration state is restored after hypothetical transitions, checked exactly before live execution. Only the selected action is executed/animated in the real episode; speculative runtime logs are separate.

A real MuJoCo regression physically relocates the cube to `(0.54,-0.08,0.37)` first. A deliberately inferior prior assigns 0.99 to +X placement. With 60 simulations and `c_puct=0.005`, MCTS chooses the shorter successful −X carry (over 20 mm less travel). Both branches use real grasp/placement physics; no invented symbolic transition or illegal decoy action establishes the override.

## Qwen lifecycle and visual policy

Use the cached `Qwen/Qwen3-VL-4B-Instruct` backend, one shared instance, CUDA/bfloat16. Semantic calls occur at observation checkpoints only, never per servo step or speculative simulation. Logs separate logical semantic requests from backend physical inference count, load time/count and inference latencies. The acceptance helper shares one backend across separate trials as well as operations.

The existing V3 Cartesian visual-policy pathway remains unchanged. It is **not extended to the new manipulation action space** in this phase; `--decision-policy visual` is rejected for `manipulate` rather than mislabeling Cartesian scores as skill priors. No visual-prior improvement experiment is claimed. Historical persistent MOVE_Z_POS bias, alpha=0.5, visual c_puct=0.05 and MCTS-recovery findings remain intact.

## CLI and traces

```bash
export PANDA_MODEL=/home/ishaan/robot-arm-playground/mujoco_menagerie/franka_emika_panda/scene.xml
.venv/bin/python -m ddm_mcts.robotics.cli --model "$PANDA_MODEL" --task manipulate \
  --instruction 'Pick up the cylinder and place it next to the cube.' --viewer --diagnostics
HF_HUB_OFFLINE=1 .venv/bin/python -m ddm_mcts.robotics.cli --model "$PANDA_MODEL" --task manipulate \
  --instruction 'Pick up the cylinder and place it next to the cube.' --perception vlm --viewer --diagnostics
.venv/bin/python -m ddm_mcts.robotics.cli --model "$PANDA_MODEL" --task manipulate \
  --instruction 'Move the cylinder next to the cube, then move the cube to the target location.' --viewer --diagnostics
HF_HUB_OFFLINE=1 .venv/bin/python -m ddm_mcts.robotics.cli --model "$PANDA_MODEL" --task manipulate \
  --instruction 'Move the cylinder next to the cube, then move the cube to the target location.' --perception vlm --viewer --diagnostics
# Existing Phase 2 continuous regression:
.venv/bin/python -m ddm_mcts.robotics.cli --model "$PANDA_MODEL" --task rearrange-demo --viewer --diagnostics
# Headless acceptance; --vlm reuses one backend for all three tasks:
HF_HUB_OFFLINE=1 MUJOCO_GL=egl .venv/bin/python -m examples.robotics.validate_manipulation_agent \
  --model "$PANDA_MODEL" --trials 2 --vlm
```

Viewer mode keeps one viewer through the entire task and holds the final scene open. Terminal output shows instruction/operation/observation, frozen physical stages and final task status. Headless mode requires no viewer.

Unique ignored `robotics_runs/*_manipulation_agent_*/task_trace.json` files contain original instruction, parsed operations/progress, observation sequence and robot/scene records, semantic replies, world states, search alternatives/priors/visits/values, selected action, physical skill results, relation checks, recovery, continuity and final scene. `search_futures/` holds speculative skill logs; `executed/` holds only real execution. No raw image arrays or model weights are duplicated into JSON. Validation manifests retain every trial.

## Limits

Known simple scene and upright cube/cylinder; bounded language, explicit simulator-native calibrated metric geometry, deterministic grasp/placement geometry, waypoint transport, finite contact retention and bounded recovery. No general collision/clutter planner or hardware safety claim. Simulation is not real-world manipulation validation. Qwen can confidently ground the wrong object; failed checks stop execution/completion rather than trusting confidence. Cylinder retention remains contact-sensitive: the frozen Phase 2 13.780 mm positional drift against a 15 mm bound and 0.749 rad full rotation versus 0.0165 rad axis tilt remain historical findings. Successful manipulation does not prove Qwen's action reasoning improves search. No V5 work is included.

## Manual viewer acceptance and checkpoint

The user manually accepted deterministic and real-Qwen single-step and multi-step viewer demonstrations. They observed genuine physical grasp, lift, transport, placement, release, retreat and continuous execution without visible teleportation, attachment, scene reset or object reset. Real-Qwen runs performed semantic grounding while deterministic geometry remained responsible for metric manipulation. This acceptance leaves the documented semantic failures, alternate-view recovery, explicit simulator metric geometry, historical visual-action bias and cylinder retention limitations unchanged.

The authorized checkpoint is `Complete V4 Phase 3 physical AI agent`, pushed only on `v4-phase3-physical-ai-agent`. Main is not merged or changed; no V5 work is included. Earlier pending-acceptance statements in the test report/changelog describe the preceding implementation checkpoint.
