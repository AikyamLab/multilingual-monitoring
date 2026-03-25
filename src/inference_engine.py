from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import torch
import yaml
from transformers import AutoTokenizer

from model_loader import load_model


class InferenceEngineError(Exception):
    """Raised when an inference backend cannot be initialized or used."""


def _read_models_config(config_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path).expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Model config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise InferenceEngineError("Top-level models YAML must be a mapping")
    models = data.get("models")
    if not isinstance(models, dict) or not models:
        raise InferenceEngineError("Expected non-empty 'models' mapping in models_config.yaml")
    return data


def _get_model_cfg(model_key: str, config_path: str | Path) -> dict[str, Any]:
    config = _read_models_config(config_path)
    models = config["models"]
    if model_key not in models:
        raise InferenceEngineError(
            f"Unknown model '{model_key}'. Available: {sorted(models.keys())}"
        )
    cfg = models[model_key]
    if not isinstance(cfg, dict):
        raise InferenceEngineError(f"Model entry '{model_key}' must be a mapping")
    return cfg


class BaseInferenceEngine(ABC):
    def __init__(self, model_key: str, config_path: str | Path) -> None:
        self.model_key = model_key
        self.config_path = Path(config_path).expanduser().resolve()
        self.model_cfg = _get_model_cfg(model_key, self.config_path)

    @property
    @abstractmethod
    def tokenizer(self) -> Any:
        raise NotImplementedError

    @abstractmethod
    def generate(self, prompts: list[str]) -> list[str]:
        raise NotImplementedError


class HFInferenceEngine(BaseInferenceEngine):
    def __init__(self, model_key: str, config_path: str | Path) -> None:
        super().__init__(model_key, config_path)
        self.model, self._tokenizer, self.model_cfg = load_model(
            model_key, config_path=self.config_path
        )
        self.model.eval()
        self.generation_kwargs = {
            "max_new_tokens": int(self.model_cfg["max_new_tokens"]),
            **dict(self.model_cfg.get("generation", {})),
        }
        self._device = next(self.model.parameters()).device

    @property
    def tokenizer(self) -> Any:
        return self._tokenizer

    def generate(self, prompts: list[str]) -> list[str]:
        if not prompts:
            return []
        inputs = self._tokenizer(prompts, return_tensors="pt", padding=True).to(self._device)
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                pad_token_id=self._tokenizer.pad_token_id,
                **self.generation_kwargs,
            )
        prompt_width = inputs["input_ids"].shape[1]
        new_token_ids = output_ids[:, prompt_width:]
        return self._tokenizer.batch_decode(
            new_token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )


class VLLMInferenceEngine(BaseInferenceEngine):
    """
    VLLM for inference.
    """
    def __init__(self, model_key: str, config_path: str | Path) -> None:
        super().__init__(model_key, config_path)

        try:
            from vllm import LLM, SamplingParams
        except ImportError as exc:
            raise InferenceEngineError("vLLM is not installed") from exc

        self._LLM = LLM
        self._SamplingParams = SamplingParams

        family = str(self.model_cfg.get("family", "")).strip().lower()
        trust_remote_code = family in {"qwen", "deepseek"}

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_cfg["path"],
            trust_remote_code=trust_remote_code,
        )
        self._tokenizer.padding_side = "left"
        if self._tokenizer.pad_token_id is None and self._tokenizer.eos_token is not None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        llm_kwargs: dict[str, Any] = {
            "model": self.model_cfg["path"],
            "tensor_parallel_size": int(self.model_cfg.get("tensor_parallel_size", 1)),
            "dtype": self.model_cfg.get("dtype", "auto"),
            "gpu_memory_utilization": float(self.model_cfg.get("gpu_memory_utilization", 0.92)),
            "enforce_eager": bool(self.model_cfg.get("enforce_eager", False)),
            "enable_prefix_caching": bool(self.model_cfg.get("enable_prefix_caching", True)),
            "enable_chunked_prefill": bool(self.model_cfg.get("enable_chunked_prefill", True)),
            "max_num_batched_tokens": int(self.model_cfg.get("max_num_batched_tokens", 32768)),
            "max_num_seqs": int(self.model_cfg.get("max_num_seqs", 256)),
        }

        if "max_model_len" in self.model_cfg:
            llm_kwargs["max_model_len"] = int(self.model_cfg["max_model_len"])

        quant_cfg = dict(self.model_cfg.get("quantization", {}))
        if quant_cfg.get("load_in_4bit", False):
            llm_kwargs["quantization"] = "bitsandbytes"

        self.llm = self._LLM(**llm_kwargs)

        # Sampling (compatible with old + new vLLM)
        gen_cfg = dict(self.model_cfg.get("generation", {}))
        self.sampling_params = self._SamplingParams(
            max_tokens=int(self.model_cfg["max_new_tokens"]),
            temperature=float(gen_cfg.get("temperature", 0.0)),
            repetition_penalty=float(gen_cfg.get("repetition_penalty", 1.0)),
            top_p=1.0,
        )

    @property
    def tokenizer(self) -> Any:
        return self._tokenizer

    def generate(self, prompts: list[str]) -> list[str]:
        if not prompts:
            return []
        outputs = self.llm.generate(prompts, self.sampling_params)
        return [out.outputs[0].text if out.outputs else "" for out in outputs]


def build_inference_engine(
    model_key: str,
    config_path: str | Path,
) -> BaseInferenceEngine:
    cfg = _get_model_cfg(model_key, config_path)
    backend = str(cfg.get("backend", "hf")).strip().lower()

    if backend == "hf":
        return HFInferenceEngine(model_key, config_path)
    if backend == "vllm":
        return VLLMInferenceEngine(model_key, config_path)

    raise InferenceEngineError(
        f"Unsupported backend '{backend}' for model '{model_key}'. "
        f"Supported: ['hf', 'vllm']"
    )