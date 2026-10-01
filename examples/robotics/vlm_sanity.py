"""Real rendered-image Qwen sanity check, before any planning/control."""

import argparse
import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from ddm_mcts.robotics.camera import MujocoCameraObservationProvider
from ddm_mcts.robotics.semantic_scene import panda_semantic_reach
from ddm_mcts.robotics.vlm_backend import Qwen3VLBackend, VLMConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Panda scene.xml")
    parser.add_argument("--goal", default="Reach the cylindrical object.")
    parser.add_argument("--vlm-model", default=VLMConfig().model_id)
    parser.add_argument("--vlm-device", default="auto")
    parser.add_argument("--vlm-debug", action="store_true")
    parser.add_argument("--output", default="robotics_runs")
    args = parser.parse_args()
    folder = Path(args.output) / (datetime.now(UTC).strftime("%Y%m%dT%H%M%S") + "_vlm_sanity_" + uuid4().hex[:8])
    folder.mkdir(parents=True, exist_ok=False)
    world = panda_semantic_reach(args.model)
    backend = Qwen3VLBackend(VLMConfig(args.vlm_model, args.vlm_device))
    with MujocoCameraObservationProvider(world, width=640, height=480) as camera:
        observation = camera.observe()
    result = backend.infer(observation.rgb, args.goal)
    record = {"goal": args.goal, "semantic_result": asdict(result), "vlm": backend.diagnostics()}
    (folder / "vlm_sanity.json").write_text(json.dumps(record, indent=2) + "\n")
    if args.vlm_debug:
        from PIL import Image, ImageDraw

        image = Image.fromarray(observation.rgb)
        image.save(folder / "input.png")
        draw = ImageDraw.Draw(image)
        box = tuple(v * (image.width if i % 2 == 0 else image.height) for i, v in enumerate(result.bbox))
        draw.rectangle(box, outline="red", width=2)
        image.save(folder / "grounding.png")
    print(json.dumps({**record, "output": str(folder)}, indent=2))


if __name__ == "__main__":
    main()
