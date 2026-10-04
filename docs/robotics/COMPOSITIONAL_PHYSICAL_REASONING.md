# V5: compositional physical reasoning in a bounded MuJoCo domain

V4 rejects stacking, physical waits and toppling. V5 adds a separate `physical-reason` mode: language describes **desired physical predicates**, rather than selecting a sentence-specific tower handler. Frozen `manipulate`, pickup, placement, reach and visual-policy modes retain their original paths.

```
language -> ordered goals / conjunctions -> fresh physical observation
  -> relationship graph + affordances -> unsatisfied goal
  -> geometry-derived interactions -> MCTS <-> MuJoCo (when alternatives exist)
  -> Panda actuation -> observe -> physical predicate verification
  -> next goal or bounded recovery / failure
```

## Commands

```bash
export PANDA_MODEL=/home/ishaan/robot-arm-playground/mujoco_menagerie/franka_emika_panda/scene.xml

MUJOCO_GL=glfw .venv/bin/python -m ddm_mcts.robotics.cli --model "$PANDA_MODEL" \
  --task physical-reason --instruction 'Put the cube on the cylinder.' --viewer --diagnostics

MUJOCO_GL=glfw .venv/bin/python -m ddm_mcts.robotics.cli --model "$PANDA_MODEL" \
  --task physical-reason --instruction 'Wait for 2 seconds.' --viewer --diagnostics

MUJOCO_GL=glfw .venv/bin/python -m ddm_mcts.robotics.cli --model "$PANDA_MODEL" \
  --task physical-reason --instruction 'Push the cylinder over.' --viewer --diagnostics

MUJOCO_GL=glfw .venv/bin/python -m ddm_mcts.robotics.cli --model "$PANDA_MODEL" \
  --task physical-reason \
  --instruction 'Pick up the cube and place it on the cylinder. Once that is done, wait for 2 seconds and topple the tower.' \
  --viewer --diagnostics

HF_HUB_OFFLINE=1 MUJOCO_GL=glfw .venv/bin/python -m ddm_mcts.robotics.cli --model "$PANDA_MODEL" \
  --task physical-reason --perception vlm \
  --instruction 'Pick up the cube and place it on the cylinder. Once that is done, wait for 2 seconds and topple the tower.' \
  --viewer --diagnostics
```

Use `MUJOCO_GL=egl` and omit `--viewer` for headless runs. The CLI keeps the final display open. The GUI smoke test used one passive viewer at 4x pacing, closed automatically after successful final verification; manual viewer acceptance remains for the user.

A standalone wait on a fresh CLI scene does not construct a tower. To wait on a stack, use a combined instruction or keep the same programmatic agent/world. Similarly, `Topple the tower` requires a previously physically verified tower group; it never invents a rigid tower.

## Programmatic composition

```python
from ddm_mcts.robotics.reasoning_scene import panda_reasoning_scene
from ddm_mcts.robotics.physical_reasoning import CompositionalPhysicalAgent

world = panda_reasoning_scene(model_path)
agent = CompositionalPhysicalAgent(world)
stack = agent.run_goal_instruction("Put the cube on the cylinder.")
if stack.success:
    wait = agent.run_goal_instruction("Wait for 2 seconds.")
    topple = agent.run_goal_instruction("Topple the tower.")
```

These calls share the live world, robot, independently simulated objects and logical group. The same placement, wait, push, search and predicate components implement the combined headline instruction. There is no comparison against the complete acceptance sentence.

The interpreter accepts one to five clauses: pick up cube/cylinder; put/place/move one on or on top of the other; next to; move to the named target location; wait for a positive duration up to ten seconds; topple cube/cylinder/tower; push cube/cylinder over. `then`, sentence punctuation and `once that is done` impose ordering. `pick up X and place/put it ...` resolves the pronoun within that bounded clause. Unknown objects, unsupported relations and malformed clauses raise a clear rejection before actuation. This is not unrestricted language understanding.

## Goals, physical state and predicates

`PhysicalGoal`, `GoalConjunction`, `GoalSequence` and `TemporalGoal` represent intent. Stack language produces ON_TOP_OF AND SUPPORTED_BY AND STABLE. Every conjunct must verify before the wait or toppling goal begins.

`PhysicalState.observe()` reads actual object poses/orientations and body velocities, contacts, gripper state and time-indexed settling evidence. It reconstructs support/contact edges, satisfied unary predicates, affordances and logical groups. Requested goals do not enter the truth evaluator.

- HELD: bilateral finger force, physical width in (5, 75) mm, measured TCP-relative displacement below 15 mm, and no table support. The grasp-relative reference comes from a verified physical pickup.
- CONTACTING: actual force-bearing contact (>0.01 N), at most 0.5 mm separation.
- SUPPORTED_BY / ON_TOP_OF: support contact, upward support surface, vertical ordering, COM projection inside the usable support region and bottom/support height difference below 3 mm.
- NEXT_TO: the frozen independent final-pose relation verifier, including surface gap and support geometry.
- UPRIGHT: local +Z tilt below 0.10 rad.
- STABLE: at least 0.45 s of evidence in a trailing 0.5 s window, positional drift below 2 mm, maximum linear speed below 0.01 m/s and angular speed below 0.10 rad/s.
- RELEASED: no force-bearing finger contact.
- AT_LOCATION: actual center within 10 mm, table contact and release.
- TOPPLED(object): tilt above pi/4, actual table contact and no finger contact.
- TOPPLED(tower): a previously verified stack relationship has disappeared and at least one constituent has tilt above pi/4.

A cube that falls from a poor stack and subsequently settles on the table can correctly be STABLE. It still fails the stack conjunction because ON_TOP_OF and SUPPORTED_BY are false. Stability alone cannot certify the requested support relationship.

The tower is a retained logical pair `(upper, lower)`, formed only after a verified stack conjunction. It has no MuJoCo body, joint, weld or special mass. After toppling, the current relationship graph loses the stack support edges while retaining the entity's identity.

## Affordances and support surfaces

Affordances expose graspability, pushability, contactability, support availability and binary placement availability. `placeable_on` lists eligible current support owners; an upright object held away from the table is not an available stack base. A verified HELD precondition suppresses redundant grasping when composing a pick goal with a subsequent placement goal. Uprightness and Panda jaw range constrain grasp/placement; support-surface validation and COM containment constrain support goals. Candidate push geometry and workspace limits constrain interaction alternatives.

The table surface is read from its current collision geometry. An object's top surface is transformed from its current pose and full dimensions. Cylinder tops use a circular usable region of radius minus a 5 mm COM margin; box tops use inset rectangular bounds. A tilted owner is rejected as an upright support surface. COM containment is not a claim that every cube corner lies over the circular cylinder top.

After verified lift, the support pose is refreshed and the desired object center is its current top plus half the held object's height. Placement uses the **measured** TCP/object rigid transform and the frozen transform composition/inversion functions. Transport lifts before translation, descends in 1 mm commands, stops at actual support contact (with a bounded 2 mm compliance allowance), opens physically, retreats 100 mm and advances one second of natural settling.

## Reusable interaction boundary

`Primitive` and `InteractionExecutor.execute()` expose MOVE_TCP, OPEN_GRIPPER, CLOSE_GRIPPER, GRASP, RELEASE, MOVE_HELD_OBJECT, CONTACT, PUSH, WAIT, OBSERVE and VERIFY. They are execution interfaces, not language-level metric commands. GRASP delegates to the frozen Phase 2 pickup/recovery component, which in turn executes frozen Phase 1 pickup. No duplicate grasp controller is introduced.

MOVE_HELD_OBJECT works with a desired object pose and a support surface, not a fixed tower coordinate. It monitors contact-based retention, 15 mm positional drift and 0.35 rad rotational/axis drift using the frozen retention verifier. Bilateral geometric contact permits the existing maximum 20 ms force-unloading allowance. Support descent can unload finger forces without mistaking the supported object for a drop. Pose commands retain 2 mm / 0.025 rad tolerances and at most one convergence retry.

WAIT calls MuJoCo stepping, never Python sleep with frozen physics. It advances the requested duration to within one 2 ms integration step, temporarily shortening the last execution block and restoring the block size even on failure. Observations/predicates refresh while bodies continue to evolve. Viewer pacing is separate from simulation advancement.

PUSH closes the physical gripper, executes lift-before-translate to pre-contact, approaches, follows a bounded contact trajectory, withdraws upward and allows settling. Object contact points are derived from current bounds; conversion to TCP height uses the distal extent of the actual Panda finger collision meshes/boxes. Requested speed determines the command increment; it is not a direct object velocity or a certified hardware velocity limit. The trajectory stops early after 45 degrees of target tilt. CONTACT uses the same bounded approach with a minimal displacement. No forces or impulses are applied directly to objects.

## MCTS and world-model responsibility

Four bounded alternatives combine world-X directions with low/short (30% height, 15 mm) and rim/long (95% height, 50 mm) pushes at 0.01 m/s nominal command speed. They are generated from current geometry; the rim TCP height includes the measured finger-tip offset. The short/low interactions can slide the cylinder without toppling it. On a tower, some low approaches predict forbidden hand/upper-object contact and are rejected by search.

The existing MCTS/RoboticsPlanner/SimulatorAdapter searches these actual MuJoCo transitions. Default: 60 search iterations, one interaction per branch, uniform priors, c_puct=0.05. Tree expansion evaluates the four physical futures once and reuses their terminal outcomes in subsequent visits; 60 iterations does not mean 60 complete physical pushes. Verified toppling dominates failure; nominal push distance breaks cost ties. Forbidden-contact outcomes are unsuccessful. A selected future with a forbidden-contact failure is rejected before live execution.

No MCTS is advertised for the forced grasp/support-placement/wait components. A fresh push tree is created after each unsuccessful live push. The world model restores full MuJoCo integration state, velocities, controls, warm-start state, world progress and copied predicate history/group/grasp-reference state. Live viewer/camera/VLM callbacks are absent from speculative execution. Live and search counters remain separate.

The deliberately misleading-prior experiment gives the ineffective +X low push 0.99 prior. Each other action has 0.003333. MCTS gives each successful rim direction 29/60 visits, each ineffective low direction 1/60, and selects the +X rim alternative. The selected live push verifies ~pi/2 tilt with no forbidden contact. This demonstrates a physical transition/search effect, not a cosmetic prior override.

## Semantic grounding and recovery

Qwen remains the existing local `Qwen/Qwen3-VL-4B-Instruct` semantic backend. The image/goal prompt includes no body IDs, XYZ, support transforms, contact points, push vectors or privileged answer. Calibrated simulator geometry supplies metrics explicitly.

Independent target/support identities are grounded before placement. After verified lift and release, fresh physical state determines predicates; a fresh camera frame is captured after release in camera mode. The occluded/coincident tower members keep their previously grounded, physically verified identity. The model is not asked to recover millimeter geometry from an occluded stack. One backend stays loaded; scene changes invalidate semantic caches, including verified lift, release and push. It does not run after every physics step or during search.

Semantic grounding permits one alternate-view retry. Grasp recovery reuses Phase 2's maximum one pre-release retry. Pose convergence permits one repeat. Toppling permits at most two live pushes, with fresh observation/search between them. Unstable/incorrect released support, dropped objects, forbidden contacts, unsafe search selections or exhausted retries stop later goals. No complex post-release regrasp recovery is added.

## Contact tuning and failed development evidence

V4 assets and scenes remain unchanged. The new V5 scene changes cube/cylinder geometry time constants from 5 ms to 4 ms and the table geometry from its 20 ms default to 4 ms, MuJoCo's safe minimum of two 2 ms timesteps. Friction, mass, gravity, joint constraints, actuator limits and finger servos remain unchanged. Measured object/table contact solref changes from 12.5 ms to 4 ms; original finger-pad parameters remain unchanged. This reduces transient impact penetration when the cube falls about 50 mm during legitimate toppling; it does not provide adhesion or artificial stabilization. The penetration rejection bound remains 3 mm.

Development evidence is retained: lower side/rim approaches slid the cylinder; a distal-tip rim approach toppled it. Two early full tasks physically toppled the tower but were rejected for impact penetration (about 3.172 mm and 3.013 mm; one speculative direction also dropped the cube off the table). Slower, shorter pushes, early withdrawal and the documented compliance change resolved this. An early diagnostic invocation omitted a required pose quaternion; a failure-trace import error was fixed and failure tests rerun. No failed experiment was counted as successful acceptance.

## Designated acceptance

Runtime manifests (ignored, not committed):

- Stacking: `robotics_runs/20261003T233350_v5_validation_8f1050896341/validation.json`: **3/3**. Horizontal error 0.886099 mm; cube orientation error 0.000806839 rad; support height error -0.007925 mm; real cube/cylinder support, one-second settling, trailing-window maximum speeds 0.000360 m/s and 0.005045 rad/s. Stable, no forbidden contacts.
- Cylinder push: `robotics_runs/20261003T233402_v5_validation_1d762d12a644/validation.json`: **3/3**, final tilt ~1.570796 rad, actual finger contact, released/table-supported, no forbidden contacts.
- Deterministic headline: `robotics_runs/20261003T233402_v5_validation_039f98592853/validation.json`: **3/3**, all stack predicates verified, physical wait 2.000000 s, tower toppled, no reset, no forbidden contacts.
- Real-Qwen headline: `robotics_runs/20261003T233402_v5_validation_e171fbdad3c0/validation.json`: **1/1**, CUDA/bfloat16, one shared load (15.491219 s during concurrent validation), two semantic calls (4.352975 / 2.846543 s), no alternate-view recovery in this trial.
- Misleading prior: `robotics_runs/20261003T234234_physical_reason_ae17d2f25e4c/task_trace.json`; actual MCTS alternatives and live verified outcome.
- Controlled failures: `robotics_runs/20261003T234327_v5_failures_991c6d190acf/validation.json`: poor COM alignment leads to a failed stack relationship; two ineffective physical pushes fail to topple; unsupported input rejected. No later wait executes.
- Continuous GUI: `robotics_runs/20261003T233847_physical_reason_09271c972d2c/task_trace.json`: successful full task, exact qpos/qvel/control/time continuity at goal boundaries, one viewer.

Across the four designated manifests: **254,000 live physics substeps**, **354,494 speculative substeps**, **4,677,105 intended contact observations**, zero live forbidden contacts, maximum live penetration **1.568301 mm**. Search-future forbidden contacts are recorded separately and excluded from live accounting. These totals exclude development, controlled failure, override, GUI and regression episodes.

Validation scripts:

```bash
MUJOCO_GL=egl .venv/bin/python examples/robotics/validate_physical_reasoning.py \
  --model "$PANDA_MODEL" --scenario headline --trials 3
HF_HUB_OFFLINE=1 MUJOCO_GL=egl .venv/bin/python examples/robotics/validate_physical_reasoning.py \
  --model "$PANDA_MODEL" --scenario headline --trials 1 --vlm
MUJOCO_GL=egl .venv/bin/python examples/robotics/validate_physical_failures.py --model "$PANDA_MODEL"
```

The earlier designated manifests label their TCP-height parameter `contact_point`; the final interface separates the genuine object-surface point from its geometry-derived TCP height, retaining the same executed poses. Those historical files were not rewritten.

Every operation writes a unique ignored `task_trace.json`, including goals, observations, graph changes, affordances, search outcomes, selected interactions, contact accounting, verification and failures. Historical runtime results are not overwritten.

## Limits

This is compositional physical reasoning in a **bounded, calibrated MuJoCo manipulation domain**. It supports known cube/cylinder geometry, inset upright supports, a bounded grammar, deterministic grasp/placement geometry and bounded waypoint pushes. It has no general collision planner, arbitrary clutter/mesh manipulation, learned physics, unrestricted language or real-hardware safety guarantee. Qwen can confidently misidentify shadowed objects; alternate views may be necessary. Historical Qwen visual-action upward bias and Phase 2 cylinder drift limits (13.780 mm against 15 mm; full rotation 0.749 rad with axis tilt 0.0165 rad) remain unchanged. Simulation success is not real-robot safety or reliability validation.

The final viewer hold is a display for inspection, not an unlimited physical stability test. Stability claims apply only to the executed settling/wait intervals.

Final-code real-Qwen revalidation: `robotics_runs/20261004T001847_v5_validation_07a1263d996d/validation.json`, 1/1 success with one model load (14.388791 s), two inference calls (3.791633 / 3.164361 s), no semantic recovery. This additional audit is excluded from the designated totals above. Separate stack / wait / topple API calls also succeed in one scene: `robotics_runs/20261004T000034_v5_composition_640b5f999745/validation.json`.

Live frozen-workflow regressions passed: cube pickup, cube table placement, continuous Phase 2 rearrangement, V4 language manipulation and real-Qwen V3 cylinder approach. A requested cube destination (0.46, 0.10, 0.37) on the initial scene was correctly rejected for reference-object clearance; the valid regression destination (0.46, -0.12, 0.37) passed. That rejected trial is retained at `robotics_runs/20261003T235019_pick_place_164d8f2ffc64`.

Early rejected full tasks remain at `robotics_runs/20261003T232804_physical_reason_a76c089f52a8` and `robotics_runs/20261003T232908_physical_reason_1a59ceb29e23`. No historical acceptance file was rewritten.

Final automated validation: **372 passed** (all 320 frozen tests plus 52 new V5 tests). Tests include a real already-satisfied placement goal with zero actuator/physics execution, independent PICK → support placement without redundant regrasp, exact state continuity, actual MuJoCo misleading-prior override, WAIT failure gating, invalid interaction parameters and CLI zero-exploration forwarding. Ruff, changed-code formatting and whitespace checks pass.

## Final checkpoint audit and manual acceptance — 2026-10-04

The user manually ran and visually accepted stacking, physical pushing/cylinder
 toppling, the deterministic full headline instruction, and its real-Qwen version.
This is manual visual acceptance, separate from the automated measured acceptance
above; it adds no measurements or reliability claims.

### Source integrity and composition

`physical_goals.py:70` parses reusable bounded clauses into goals. The exact
headline sentence occurs in validation/examples/documentation, not as a production
execution dispatch. The named V4-compatible target location is a configured fixed
coordinate; it is not used to generate stack or push geometry.
`physical_reasoning.py:141` derives placement from the current support surface and
object dimensions, refreshes it after pickup, and executes through the measured
TCP/object transform (`interactions.py:249`). `interactions.py:86` derives four push
candidates from current object dimensions/pose. `physical_reasoning.py:201` executes
exactly the candidate returned by `search_push`, with at most two attempts.
There is no fixed stack/wait/topple sequence in the executor: ordered parsed goals
and authoritative physical predicates gate each transition. Production contains
no test monkeypatch hooks, forced predicate truth or forced success; fault injection
is confined to tests and the controlled-failure validation example.

### Predicate audit

All predicates are recalculated by `PhysicalState.evaluate`; each can become false
again when the physical state changes. Recorded historical goal satisfaction does
not force current truth. Tests are in `test_physical_reasoning.py` (support/contact
true/false at line 98, unary at 111, stability at 124, held at 130, tower at 141,
toppled at 151, and relation/location at 161).

| Predicate | Source line | Signals and important bounds |
|---|---:|---|
| HELD | physical_predicates.py:169 | Bilateral force-bearing contact, width 5–75 mm, measured relative positional drift <15 mm, no table contact |
| SUPPORTED_BY | physical_predicates.py:145 | Actual support contact, usable COM region (5 mm inset), bottom/support gap <3 mm, vertical ordering |
| ON_TOP_OF | physical_predicates.py:145 | Same conservative contact/COM/vertical support criteria as SUPPORTED_BY |
| NEXT_TO | physical_predicates.py:189 | Final physical poses; frozen verifier requires upright/table-supported geometry and surface gap 30–50 mm |
| CONTACTING | physical_predicates.py:142 | Actual contact normal force >0.01 N, separation <=0.5 mm |
| UPRIGHT | physical_predicates.py:163 | World-up axis tilt <0.10 rad |
| STABLE | physical_predicates.py:178 | Trailing 0.5 s history with >=0.45 s evidence; drift <2 mm, linear speed <0.01 m/s, angular speed <0.10 rad/s |
| TOPPLED | physical_predicates.py:165 | Object tilt >pi/4, table contact, no finger contact; tower additionally uses a previously verified group and loss of its stack relation (line 121) |
| RELEASED | physical_predicates.py:167 | No force-bearing finger contacts |
| AT_LOCATION | physical_predicates.py:194 | Actual position error <10 mm, table contact, no finger contact |

STABLE can correctly become true on the table after a failed stack falls; ON_TOP_OF
and SUPPORTED_BY still fail and prevent subsequent goals. ON_TOP_OF and SUPPORTED_BY
share an intentionally conservative implementation in this upright support domain,
not separate general mechanical-support estimators.

### Physics and compliance integrity

The live V5 execution path contains no object qpos/velocity assignments, welds,
parenting, gravity disabling, freezing, direct impulses, forced orientations,
contacts or stabilization. State reads and speculative snapshot/restore are
legitimate; test initialization is separate from live execution. WAIT advances
MuJoCo via `interactions.py:207`, not sleeping with frozen physics.

`reasoning_scene.py:6` changes only cube/cylinder/table geom solref time constants:
objects 5 ms ->4 ms; table 20 ms ->4 ms. This affects contacts involving those
geometries, not only intended contacts. It is V5-scene-wide contact compliance;
friction, mass, gravity, damping ratio, finger servo parameters and frozen V4 scenes
remain unchanged. The measured combined original object/table response was 12.5 ms.
The 4 ms response is two 2 ms timesteps, reducing impact penetration during physical
falls while keeping the 3 mm rejection bound unchanged. The observed 1.568301 mm
maximum live penetration is finite compliant contact under impact, below that bound;
it is not a zero-penetration or real-hardware claim. This tuning changes physical
outcomes as contact parameters normally do, but introduces no adhesion, stabilizing
constraint or imposed successful state. Earlier rejected impact trials remain
preserved as development evidence.

### Search and failure integrity

`interaction_search.py:48` simulates each candidate using the real executor and
physical TOPPLED predicate. Success value is minus nominal distance; failure value
is -2 plus bounded tilt progress. Snapshot/restore includes simulator state and
reasoning history/group/grasp state (`:39`); live snapshot equality is checked
following search (`:77`). Four alternatives are expanded physically; terminal
values are reused across 60 visits, rather than pretending 60 full pushes occurred.
The misleading-prior MuJoCo test at `test_physical_reasoning.py:232` reproduces
prior override and executes the selected push live. Forced single-action stages
bypass search. Search and live counters remain separate.

Independent stack, push, wait, and combined tasks use these same paths. Controlled
poor alignment prevents later goals; ineffective pushes re-observe and search anew
with a two-attempt bound; unsupported language rejects before actuation. No infinite
recovery is available. Historical acceptance logs/results remain unchanged.

Final-checkpoint misleading-prior reproduction (current code): low -X/+X pushes
received one visit each, Q=-1.999979/-1.999986; rim -X/+X received 29 visits each,
Q=-0.05. The prior was 0.99 on low +X and 0.003333333 on each alternative.
Selected rim +X therefore differed from prior top-1 and physically achieved TOPPLED
(tilt 1.570796327 rad). Snapshot equality passed. This additional audit used 64,050
speculative and 16,525 live substeps, 0 forbidden live contacts; it is separate from
previous designated acceptance totals. Diagnostic audit output is outside Git.
