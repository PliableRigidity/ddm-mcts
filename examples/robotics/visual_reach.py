"""Real Phase 3 camera/perception example; delegates to the existing toolkit CLI.

python -m examples.robotics.visual_reach --model PATH --goal red --viewer
"""

import sys

from ddm_mcts.robotics.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["--task", "visual-reach", *sys.argv[1:]]))
