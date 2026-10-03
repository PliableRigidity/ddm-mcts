"""Observe/perceive coordination over the existing receding-horizon planner."""

from dataclasses import asdict
from math import dist
from time import perf_counter

from .run import run_episode


class PhysicalAgent:
    """Composable facade, not a second search or physics implementation.

    task.resolve(world_state) binds the task's perceived goal to the world model.
    The configured planner remains the original RoboticsPlanner. Robot dynamics
    initialization uses the live simulator snapshot explicitly.
    """

    def __init__(self, environment, observer, perception, task, planner, *, diagnostics=None, save_images=False, visual_policy=None):
        if planner.adapter.world is not environment:
            raise ValueError("agent and planner must share the same environment")
        self.environment, self.observer, self.perception = environment, observer, perception
        self.task, self.planner = task, planner
        self.search, self.adapter = planner.search, planner.adapter
        self.diagnostics_provider = diagnostics
        self.save_images, self.visual_policy = save_images, visual_policy
        if visual_policy is not None:
            self.search.root_policy = visual_policy
        self._prepared = False
        self.state = self.observation = None
        self._folder = None
        self._saved_sequences = set()
        self.decision_context = {}
        self.perception_requests = 0
        self.total_observation_seconds = self.total_perception_seconds = 0.0
        self.run_configuration = {
            "observation_provider": type(observer).__name__,
            "perception_provider": type(perception).__name__,
            "semantic_task": asdict(task),
            "world_model_initialization": "live simulator snapshot; privileged robot dynamics",
            "diagnostics_enabled": diagnostics is not None,
            "save_images": save_images,
            "camera": {
                "name": getattr(observer, "camera", None),
                "width": getattr(observer, "width", None),
                "height": getattr(observer, "height", None),
            },
            "perception_configuration": getattr(perception, "configuration", lambda: {})(),
            "direct_visual_policy": type(visual_policy).__name__ if visual_policy is not None else None,
        }

    def prepare(self):
        """Observe again after every action, including the final success check."""
        if not self._prepared:
            started = perf_counter()
            observation = self.observer.observe()
            observation_seconds = perf_counter() - started
            self.total_observation_seconds += observation_seconds
            started = perf_counter()
            self.perception_requests += 1
            try:
                state = self.perception.perceive(observation, self.task.goal)
            finally:
                self.total_perception_seconds += perf_counter() - started
            prepare_state = getattr(self.task, "prepare_state", None)
            if prepare_state is not None:
                state = prepare_state(state)
            perception_seconds = perf_counter() - started
            self.environment.task = self.task.resolve(state)
            if self.visual_policy is not None:
                self.visual_policy.bind(observation, self.task.goal)
            self.observation, self.state = observation, state
            context = {
                "observation": {
                    "source": observation.source,
                    "sequence": observation.sequence,
                    "timestamp": observation.timestamp,
                    "metadata": dict(observation.metadata),
                },
                "perceived_state": asdict(state),
                "observation_seconds": observation_seconds,
                "perception_seconds": perception_seconds,
                "goal": asdict(self.task.goal) if self.task.goal is not None else None,
                "warnings": [f"{e.label}: retained estimate for {e.missed_frames} missed frames" for e in state.entities if not e.observed],
            }
            if observation.camera is not None:
                context["camera"] = asdict(observation.camera)
            task_diagnostics = getattr(self.task, "diagnostics", None)
            if task_diagnostics is not None:
                context["approach"] = task_diagnostics(state)
                context["tcp"] = self.environment.controller.frame_diagnostics()
            perception_metrics = getattr(self.perception, "diagnostics", None)
            if perception_metrics is not None:
                context["perception_diagnostics"] = perception_metrics()
            if self.diagnostics_provider is not None:
                truth = {e.label: e for e in self.diagnostics_provider()}
                context["ground_truth_entities"] = [asdict(entity) for entity in truth.values()]
                context["perception_errors_m"] = {
                    e.label: dist(e.position, truth[e.label].position) for e in state.entities if e.label in truth
                }
                context["actual_simulator_state"] = asdict(self.environment.get_state())
            self.decision_context = context
            self._prepared = True
        self._log_observation()
        return self.state

    def plan(self):
        state = self.prepare()
        try:
            result = self.planner.plan(state, state_projection=state.predict_observation)
            if self.visual_policy is not None:
                self.decision_context["visual_decision"] = self.visual_policy.diagnostics()
            return result
        finally:
            self._prepared = False

    def start_run(self, folder):
        self._folder = folder
        self._saved_sequences.clear()

    def _log_observation(self):
        if self._folder is None or self.observation.sequence in self._saved_sequences:
            return
        import json

        with (self._folder / "observations.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(self.decision_context) + "\n")
        if self.save_images and self.observation.rgb is not None:
            # Portable binary RGB image; no imaging dependency or huge arrays in logs.
            images = self._folder / "images"
            images.mkdir(exist_ok=True)
            rgb = self.observation.rgb
            path = images / f"observation_{self.observation.sequence:04d}.ppm"
            path.write_bytes(f"P6\n{rgb.shape[1]} {rgb.shape[0]}\n255\n".encode() + rgb.tobytes())
        save_debug = getattr(self.perception, "save_debug", None)
        if save_debug is not None:
            save_debug(self._folder, self.observation)
        self._saved_sequences.add(self.observation.sequence)

    def logged_state(self, simulator_state):
        if self.diagnostics_provider is not None:
            return simulator_state
        # This is a predicted post-action observation, not another camera detection.
        return self.state.predict_observation(simulator_state)

    def set_task(self, task):
        """Change semantic goal without changing environment, perception, or search."""
        self.task = task
        self.run_configuration["semantic_task"] = asdict(task)
        self._prepared = False

    def reset(self, task=None):
        """Start another episode, clearing perception history and prepared state."""
        self.environment.reset()
        reset = getattr(self.perception, "reset", None)
        if reset is not None:
            reset()
        self._prepared = False
        self.state = self.observation = None
        if task is not None:
            self.set_task(task)

    def run(self, output_root="robotics_runs", *, task=None, visualization=None, configuration=None):
        if task is not None:
            self.set_task(task)
        try:
            return run_episode(self.environment, self, output_root, configuration=configuration, observer=visualization)
        finally:
            # Later observe/plan calls must never append to a completed run.
            self._folder = None

    def policy_diagnostics(self):
        return {
            "structured": getattr(self.search.policy, "diagnostics", lambda: {})(),
            "direct_visual": self.visual_policy.diagnostics() if self.visual_policy is not None else None,
        }

    def run_task(self, task, output_root="robotics_runs", *, visualization=None, configuration=None, verifier=None):
        """Execute an ordered PhysicalTask without resetting the physical world.

        All components, including a loaded model and optional viewer, persist.
        Goal-dependent caches refresh and every subgoal has a new action budget.
        """
        from .ordered_task import OrderedTaskRunner

        return OrderedTaskRunner(self, verifier).run(task, output_root, visualization=visualization, configuration=configuration)

    def execution_diagnostics(self):
        diagnostics = getattr(self.task, "diagnostics", None)
        if diagnostics is None or self.state is None:
            return {}
        result = diagnostics(self.state)
        result["final_tcp_pose"] = self.environment.controller.frame_diagnostics()
        result["final_tcp_error_m"] = dist(self.environment.controller.position(), self.environment.task.goal)
        geometry = getattr(self.environment, "geometry_diagnostics", None)
        if geometry is not None and self.diagnostics_provider is not None:
            result.update(geometry(self.state.target().label))
        return result

    def close(self):
        close = getattr(self.observer, "close", None)
        if close is not None:
            close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
