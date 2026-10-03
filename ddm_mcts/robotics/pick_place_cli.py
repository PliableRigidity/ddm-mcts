"""CLI assembly only; the persistent manipulation API has no argparse dependency."""

import json
from math import isfinite

from .manipulation_scene import panda_pickup_scene
from .pick_place import PickPlaceAgent, rearrangement_plan
from .placement import NextTo, PlacementTarget


def run_pick_place(args, parser):
    if not args.model or not isfinite(args.viewer_speed) or args.viewer_speed <= 0:
        parser.error("provide --model/PANDA_MODEL and positive finite viewer speed")
    if args.perception is not None or args.decision_policy != "structured" or args.instruction or args.target:
        parser.error("pick/place uses deterministic geometry; omit reach/perception/language options")
    if args.task == "pick-place":
        if args.goal not in ("cube", "cylinder") or bool(args.place_position) == bool(args.place_next_to):
            parser.error("pick-place requires --goal cube|cylinder and exactly one placement destination")
        try:
            destination = PlacementTarget(tuple(args.place_position)) if args.place_position else NextTo(args.place_next_to)
        except ValueError as exc:
            parser.error(str(exc))
    elif args.goal or args.place_position or args.place_next_to:
        parser.error("rearrange-demo has a predefined geometric operation list; omit goal/destination")
    world = panda_pickup_scene(args.model)
    agent = PickPlaceAgent(world)

    def run(visual=None):
        return (
            agent.run_manipulation_plan(rearrangement_plan(), args.output, visualization=visual)
            if args.task == "rearrange-demo"
            else agent.pick_and_place(args.goal, destination, args.output, visualization=visual)
        )

    if args.viewer:
        from .manipulation_visual import PickupInspection
        from .visual import ViewerClosed

        try:
            with PickupInspection(world, args.viewer_speed) as visual:
                result = run(visual)
                print(json.dumps(result.trace, indent=2) if args.diagnostics else f"Success: {result.success}; logs: {result.folder}")
                print("Final scene remains open. Close the viewer or press Ctrl+C.", flush=True)
                visual.wait_until_closed()
        except ViewerClosed:
            return 1
        except KeyboardInterrupt:
            return 130
    else:
        result = run()
        print(json.dumps(result.trace, indent=2) if args.diagnostics else f"Success: {result.success}; logs: {result.folder}")
    return 0 if result.success else 1
