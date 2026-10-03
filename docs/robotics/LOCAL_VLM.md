# Local VLM perception — V3 Phase 1

V3 Phase 3 composes this same backend/perception into [ordered multi-step tasks](MULTI_STEP_AGENT.md). Goal changes clear grounding/tracking while preserving the model and cumulative inference metrics. Existing single-goal commands and Phase 1 results below remain unchanged.

This adds a third perception option to the existing toolkit: real local Qwen3-VL semantics, deterministic metric localization, then the original policy/MCTS/physics/IK pipeline. Ground-truth and color modes retain their existing commands. V3 Phase 1 has been manually validated in the real Qwen viewer, including the outside-object approach. The implementation is checkpointed on `v3-phase1-vlm`; main is not merged as part of finalization.

## Install

Python 3.11+ and the external Menagerie Panda model remain required. The optional extra leaves `pip install -e .` dependency-free:

```bash
python -m pip install -e '.[dev,robotics,vlm]'
```

Choose a CUDA PyTorch wheel appropriate to your driver before installing the extra. The validated WSL stack uses PyTorch 2.7.1+cu128, torchvision 0.22.1+cu128, Transformers 4.57.6, Accelerate 1.15.0 and Pillow 12.3.0. For that stack:

```bash
python -m pip install torch==2.7.1 torchvision==0.22.1 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e '.[dev,robotics,vlm]'
```

`uv pip install --python .venv/bin/python ...` is equivalent when this environment has uv but no pip. The extra constrains Transformers >=4.57,<5 and PyTorch >=2.6,<3. It does not upgrade dependencies for users installing only the core/robotics extras. See [official Qwen instructions](https://github.com/QwenLM/Qwen3-VL) and the [model card](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct).

The default model is `Qwen/Qwen3-VL-4B-Instruct`, approximately 8 GB of weights. First use downloads only that model through Hugging Face. The normal cache is `~/.cache/huggingface/hub/`; weights never belong in the repository. The tested snapshot is `ebb281ec70b05090aa6165b016eac8ec08e71b17`. Subsequent launches reuse cached files; `HF_HUB_OFFLINE=1` avoids network checks after download. No paid/cloud inference is used.

The backend selects CUDA if available, bfloat16 on compatible GPUs, otherwise float16 on CUDA or float32 on CPU. Validation used an RTX 4090 Laptop GPU with 16 GB VRAM. Allow room for roughly 8 GB of weights plus image/generation activations; avoid running two loaded models simultaneously on this GPU. CPU mode is supported through `--vlm-device cpu`, but was not used for real acceptance and will be much slower.

## Exact commands

From the repository root:

```bash
export PANDA_MODEL=/home/ishaan/robot-arm-playground/mujoco_menagerie/franka_emika_panda/scene.xml
```

1. Standalone rendered-image sanity test, no planner/robot action:

```bash
MUJOCO_GL=egl .venv/bin/python -m examples.robotics.vlm_sanity \
  --model "$PANDA_MODEL" --goal 'Reach the cylindrical object.' --vlm-debug
```

2. Cylinder viewer:

```bash
.venv/bin/python -m ddm_mcts.robotics.cli \
  --model "$PANDA_MODEL" --task semantic-reach \
  --goal 'Reach the cylindrical object.' --observation camera --perception vlm \
  --vlm-model Qwen/Qwen3-VL-4B-Instruct --viewer --diagnostics
```

3. Change only the semantic goal to cube:

```bash
.venv/bin/python -m ddm_mcts.robotics.cli \
  --model "$PANDA_MODEL" --task semantic-reach \
  --goal 'Reach the cube.' --observation camera --perception vlm --viewer --diagnostics
```

4. Third goal:

```bash
.venv/bin/python -m ddm_mcts.robotics.cli \
  --model "$PANDA_MODEL" --task semantic-reach \
  --goal 'Reach the spherical object.' --observation camera --perception vlm --viewer --diagnostics
```

5. Headless VLM run:

```bash
MUJOCO_GL=egl .venv/bin/python -m ddm_mcts.robotics.cli \
  --model "$PANDA_MODEL" --task semantic-reach --goal cylinder \
  --observation camera --perception vlm --diagnostics
```

6. Original Phase 2 reach:

```bash
.venv/bin/python -m ddm_mcts.robotics.cli \
  --model "$PANDA_MODEL" --target 0.5945 0.02 0.6245 --viewer
```

7. Original V2 color perception:

```bash
.venv/bin/python -m ddm_mcts.robotics.cli \
  --model "$PANDA_MODEL" --task visual-reach --goal red \
  --observation camera --perception color --viewer --diagnostics
```

`semantic-reach` defaults to camera/VLM and a cylinder language goal. It rejects color/ground-truth CLI pairs to keep the actual demonstration explicit; baseline providers remain available through the Python API. V2 `visual-reach` still accepts only red/blue. The original coordinate task is unchanged.

The observation camera is oblique to reveal shape. VLM mode uses at least 640x480; larger `--width`/`--height` work. The viewer uses its own freely movable camera. Only selected actions are animated; MCTS branches and VLM input acquisition stay invisible. The robot is stationary during inference/search, and the initial model load/goal resolution completes before the viewer opens. After success the frozen final pose stays open until normal closure or terminal Ctrl+C.

## How semantics and localization are separated

The scene contains a cube, sphere and upright cylinder, all with the **same teal material**. They are visual non-colliding objects at a shared center-height plane z=0.6245 m. Scene geometry/positions configure rendering and optional diagnostics. They are not passed to Qwen or the visual localizer. No labels are rendered into the image. The VLM sees real RGB plus the language goal and identifies the requested object from visual shape.

Qwen returns strict JSON fields `target_label`, `target_description`, `bbox_2d` and optional `confidence`. Following the [official grounding cookbook](https://github.com/QwenLM/Qwen3-VL/blob/main/cookbooks/2d_grounding.ipynb), XYXY uses relative 0..1000 coordinates: top-left origin, X right, Y down. The parser validates order/range/finite numbers, normalizes to 0..1, and rejects malformed/missing/unknown outputs. Fenced JSON is accepted; arbitrary surrounding prose is rejected. Reported confidence is model self-report, not calibrated probability of correctness.

The localizer converts the box to original image coordinates, adds 20% padding, and uses a common foreground-material mask to refine its centroid. All three shapes share this mask: it cannot select cylinder versus cube or sphere. Only the VLM box supplies semantic selection. Comparable multiple components are rejected. Mask chromaticity tolerance is 0.12; partial components below 65% of historical full area use bounded static tracking for at most eight frames. These constants are logged in configuration.

The centroid uses existing `CameraCalibration.intersect_plane` to obtain world XYZ. Calibration comes from the actual named camera's fovy/pose. This is RGB plus an explicit known plane, not guessed 3D coordinates or target truth. Finite 3D shape silhouettes and occlusion produce measurable centroid error; this is not exact object-pose reconstruction. No depth rendering is used in this phase. The estimated object label is stored separately from the free-form language goal through `WorldState.resolved_target_label`, preserving the original representation and planner API.

The resolved estimated object center is converted by ApproachReachTask into an outside-object waypoint, which drives the same ReachTask, Policy, RoboticsPlanner and MCTS. MuJoCo still initializes dynamics from privileged live robot snapshots, and robot telemetry is simulator proprioception. Perceived targets remain fixed within a search. The VLM never emits actions, priors or speculative futures, and the controller remains the existing DLS IK.

## Closed loop and reuse

Every selected action is followed by a fresh camera observation and deterministic localization before another decision. `--vlm-refresh 0` caches static-scene semantic grounding until reset, goal change or calibration change; this is the default. `--vlm-refresh 1` reinfers every live observation; N reinfers every N observations. A cached box is not a future image or an open-loop action trajectory. Visible/stale flags and semantic age are logged. Expired tracking or absent/ambiguous targets stop execution; there is no simulator-truth fallback. Moving targets, changing viewpoints or broad scene changes require appropriate refresh/tracking configuration and are not the acceptance scope.

`Qwen3VLBackend` lazily loads one model/processor and can be shared across agent episodes. Perception reset clears grounding/tracking, not the loaded model. The acceptance utility demonstrates three goals with one load:

```bash
MUJOCO_GL=egl .venv/bin/python -m examples.robotics.validate_vlm --model "$PANDA_MODEL"
```

The Python components are `panda_semantic_reach`, `MujocoCameraObservationProvider`, `Qwen3VLBackend(VLMConfig(...))`, `VLMSemanticPerception(backend, SEMANTIC_PLANE_Z, OBJECT_RGB)`, `ApproachReachTask`, the existing `RoboticsPlanner`, and `PhysicalAgent`. An existing DDM can replace UniformPolicy without changing any perception component. The injected runtime factory is intended for tests; real CLI inference uses official Qwen/Transformers APIs.

## Logs, tests and failure handling

Unique runs stay under ignored `robotics_runs/`. Agent configuration records model/device/generation/refresh/geometry options. Observation records include parsed semantic result, raw response, normalized grounding, localized entity, cache age, acquisition/perception time and optional true-entity errors. Summary diagnostics retain model load time/count separately from inference count/times/total. Steps retain MCTS settings, root statistics, selected actions, planning/execution time and final success/error.

`--vlm-debug` saves input PNG, annotated grounding/centroid PNG and response/metrics JSON per live observation under the run's `vlm_debug/`. Nothing is saved for speculative branches; images are not saved by default. This can include retained stale groundings, explicitly labeled in JSON. Malformed output/CUDA OOM/model-unavailable errors propagate with an error summary; no fallback is implicit.

Normal tests use fake runtimes and never download/load Qwen. Real model acceptance is the explicitly invoked sanity/validation utilities. Run:

```bash
MUJOCO_GL=egl PANDA_MODEL="$PANDA_MODEL" PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  .venv/bin/python -m pytest -ra
```

Troubleshooting: verify `torch.cuda.is_available()` and compatible CUDA wheels; install the VLM extra if imports fail; close other model processes for OOM; use the Hugging Face cache/offline flag after initial download; select EGL for headless rendering before MuJoCo import. Invalid JSON, wrong semantics and grounding error remain possible even at high self-reported confidence. Inspect opt-in frames and raw responses. There is no automatic download of another model, cloud fallback, quantization, FlashAttention, training, direct VLDM action scoring, grasping, hardware, ROS or next-phase work.

## Physically valid semantic reach (2026-10-01)

Semantic reach stops above the object rather than at its centroid. The final target is `object_center + (0,0,1) * (object_radius + standoff)`. Defaults are a conservative 40 mm bounding-radius prior and 50 mm additional clearance; configure them with `--object-radius` and `--standoff` (meters). The radius is explicitly configured, not an extent inferred by Qwen or secretly read from MuJoCo. It bounds all three demonstration objects. For larger objects increase it. The robot first lifts vertically to the clearance height, then translates above the perceived target; the existing closed-loop planner resolves these waypoints.

The semantic scene uses an invisible `panda_gripper_tcp` site at hand-local `(0,0,0.1034)` meters, near the finger-pad center. Both Cartesian feedback and the IK Jacobian use this site. The installed Panda model has no original TCP site. Original Phase 2 `panda_reach` retains its hand-body frame; `panda_semantic_reach` defaults to the gripper TCP and supports `use_tcp=False` for legacy API use.

With `--diagnostics`, logs include object center, configured radius/diameter, approach vector, standoff, final pre-contact target, current waypoint, requested per-action Cartesian endpoint, controlled frame, final TCP pose/error and signed minimum distance between gripper collision geometries and the selected object. Actual target geom sizes are diagnostic truth only. Viewer terminal output distinguishes the current waypoint from the final pre-contact target. Positive signed distance indicates separation. Success is proximity to the outside-object TCP target, not proximity to the object center.

The original viewer command remains valid:

```bash
.venv/bin/python -m ddm_mcts.robotics.cli \
  --model "$PANDA_MODEL" --task semantic-reach \
  --goal 'Reach the cylindrical object.' --perception vlm --viewer --diagnostics
```

Replace the goal with `Reach the cube.` or `Reach the spherical object.` for the other objects. This remains position-only reaching with unchanged physics, localization and MCTS. Lift-first staging is not general collision avoidance; orientation is unconstrained and there is no grasping/contact manipulation. Signed diagnostics use collision geometry, which can differ from visual meshes. Human inspection remains useful.

V3 Phase 2 optionally adds a [VLM-backed visual decision policy](VISUAL_DECISION.md) via `--decision-policy visual`. Perception remains separate; fresh-image priors guide only the MCTS root, with the existing configured policy at deeper nodes. The safe approach waypoint remains above the object.
