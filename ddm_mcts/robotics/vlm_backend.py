"""Lazy official Qwen3-VL inference; no model dependencies on module import."""

import json
import math
from dataclasses import dataclass
from time import perf_counter

DEFAULT_VLM = "Qwen/Qwen3-VL-4B-Instruct"


@dataclass(frozen=True)
class VLMConfig:
    model_id: str = DEFAULT_VLM
    device: str = "auto"
    max_new_tokens: int = 256

    def __post_init__(self):
        if not self.model_id or self.max_new_tokens < 32:
            raise ValueError("provide a model ID and at least 32 output tokens")


@dataclass(frozen=True)
class SemanticResult:
    target_label: str
    target_description: str
    bbox: tuple[float, float, float, float]  # normalized 0..1 XYXY
    confidence: float | None
    raw_response: str


def parse_semantics(raw: str) -> SemanticResult:
    """Qwen relative 0..1000 boxes -> validated normalized XYXY."""
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) < 3 or lines[-1].strip() != "```":
            raise ValueError("VLM returned an incomplete JSON fence")
        text = "\n".join(lines[1:-1])
    try:
        obj = json.loads(text)
        label, description, box = obj["target_label"], obj["target_description"], obj["bbox_2d"]
    except (ValueError, TypeError, KeyError) as exc:
        raise ValueError(f"VLM requires JSON target_label, target_description, bbox_2d; response: {raw[:800]}") from exc
    if not isinstance(label, str) or not label.strip() or label.lower() in {"unknown", "none", "null"}:
        raise ValueError("VLM could not identify a visible target")
    if not isinstance(description, str) or not description.strip():
        raise ValueError("VLM target_description must be nonempty")
    if (
        not isinstance(box, list)
        or len(box) != 4
        or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in box)
    ):
        raise ValueError("VLM bbox_2d must contain four finite numbers")
    x1, y1, x2, y2 = box
    if not (0 <= x1 < x2 <= 1000 and 0 <= y1 < y2 <= 1000):
        raise ValueError("VLM bbox_2d must be ordered XYXY inside 0..1000")
    confidence = obj.get("confidence")
    if confidence is not None and (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not math.isfinite(confidence)
        or not 0 <= confidence <= 1
    ):
        raise ValueError("VLM confidence must be null or a number in 0..1")
    return SemanticResult(label.strip(), description.strip(), tuple(v / 1000 for v in box), confidence, raw)


class _TransformersRuntime:
    def __init__(self, config):
        try:
            import torch
            from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        except ImportError as exc:
            raise RuntimeError("Local VLM requires the optional extra: pip install -e '.[robotics,vlm]'") from exc
        self.torch = torch
        self.device = config.device if config.device != "auto" else ("cuda:0" if torch.cuda.is_available() else "cpu")
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable; check the PyTorch CUDA build and GPU driver")
        dtype = (
            torch.bfloat16
            if self.device.startswith("cuda") and torch.cuda.is_bf16_supported()
            else (torch.float16 if self.device.startswith("cuda") else torch.float32)
        )
        self.dtype = str(dtype)
        self.processor = AutoProcessor.from_pretrained(config.model_id)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            config.model_id, dtype=dtype, device_map={"": self.device}, attn_implementation="sdpa"
        ).eval()
        self.sync()

    def sync(self):
        if self.device.startswith("cuda"):
            self.torch.cuda.synchronize(self.device)

    def generate(self, rgb, prompt, config):
        from PIL import Image

        messages = [{"role": "user", "content": [{"type": "image", "image": Image.fromarray(rgb)}, {"type": "text", "text": prompt}]}]
        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
        )
        inputs = inputs.to(self.model.device)
        with self.torch.inference_mode():
            output = self.model.generate(
                **inputs, max_new_tokens=config.max_new_tokens, do_sample=False, temperature=None, top_p=None, top_k=None
            )
        trimmed = output[:, inputs.input_ids.shape[1] :]
        response = self.processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        self.sync()
        return response


class Qwen3VLBackend:
    """One lazy model per backend, reusable across frames and agent episodes.

    An injected runtime_factory supports offline tests without torch/model weights.
    """

    def __init__(self, config=None, *, runtime_factory=None):
        self.config = config or VLMConfig()
        self._factory = runtime_factory or _TransformersRuntime
        self.runtime = None
        self.load_seconds = 0.0
        self.load_count = self.calls = 0
        self.inference_seconds = []
        self.last_response = None

    def generate(self, rgb, prompt: str) -> str:
        import numpy as np

        if not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError("VLM input must be uint8 RGB HxWx3")
        if not prompt.strip():
            raise ValueError("VLM requires a nonempty prompt")
        if self.runtime is None:
            started = perf_counter()
            try:
                self.runtime = self._factory(self.config)
            except Exception as exc:
                raise RuntimeError(f"Cannot load local VLM {self.config.model_id}: {exc}") from exc
            self.load_seconds = perf_counter() - started
            self.load_count += 1
        started = perf_counter()
        self.calls += 1
        try:
            raw = self.runtime.generate(rgb, prompt, self.config)
            self.last_response = raw
        except Exception as exc:
            raise RuntimeError(f"Local VLM inference failed (device {self.runtime.device}); check memory/model/input: {exc}") from exc
        finally:
            self.inference_seconds.append(perf_counter() - started)
        return raw

    def infer(self, rgb, goal: str) -> SemanticResult:
        if not goal.strip():
            raise ValueError("VLM requires a language goal")
        prompt = (
            f"Goal: {goal}. Identify the requested freestanding geometric object in this image, not robot parts. "
            "Return ONLY one JSON object with target_label (shape name), target_description, bbox_2d [x1,y1,x2,y2], "
            "and confidence (number 0..1 or null). The bounding box tightly encloses that entire object. "
            "Use relative image coordinates 0..1000, origin top-left, x right, y down. "
            "If absent, use target_label=unknown. Never output world coordinates or robot actions."
        )
        return parse_semantics(self.generate(rgb, prompt))

    def diagnostics(self):
        return {
            "model_id": self.config.model_id,
            "device": getattr(self.runtime, "device", self.config.device),
            "dtype": getattr(self.runtime, "dtype", None),
            "model_load_seconds": self.load_seconds,
            "model_load_count": self.load_count,
            "inference_count": self.calls,
            "inference_seconds": self.inference_seconds.copy(),
            "total_inference_seconds": sum(self.inference_seconds),
            "last_raw_response": self.last_response,
        }
