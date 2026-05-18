from __future__ import annotations

import os
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import torch
import yaml
from tqdm import tqdm
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
        trust_remote_code = family in {"qwen", "deepseek", "gptoss"}

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


# ======================================================================
# Helpers shared by the API backends
# ======================================================================

def _parse_env_file_for_key(env_path: Path, key_name: str) -> str | None:
    """Minimal KEY=VALUE .env parser. Returns the value for key_name or None."""
    if not env_path.is_file():
        return None
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        name, sep, value = line.partition("=")
        if not sep or name.strip() != key_name:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        return value or None
    return None


class _PassthroughTokenizer:
    """Duck-typed HF tokenizer stand-in for the API backends."""
    eos_token = None
    pad_token = None
    pad_token_id = None
    eos_token_id = None
    padding_side = "left"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **_):
        if isinstance(messages, str):
            return messages
        if not isinstance(messages, list):
            return str(messages)
        parts = []
        for m in messages:
            if not isinstance(m, dict):
                parts.append(str(m))
                continue
            role = m.get("role", "user")
            content = m.get("content", "")
            if role == "system":
                parts.append(f"[System]\n{content}")
            elif role == "assistant":
                parts.append(f"[Assistant]\n{content}")
            else:
                parts.append(content)
        return "\n\n".join(parts)

    def __call__(self, text, **_):
        return {"input_ids": [], "attention_mask": []}


def _run_parallel_with_progress(
    fn,
    prompts: list[str],
    concurrency: int,
    desc: str,
) -> list[str]:
    """
    Dispatch `fn` over `prompts` with up to `concurrency` threads, showing a
    tqdm bar that advances as each call completes. Results are returned in
    input order regardless of completion order.
    """
    if concurrency == 1 or len(prompts) == 1:
        return [
            fn(p)
            for p in tqdm(prompts, desc=desc, leave=False, unit="req")
        ]

    workers = min(concurrency, len(prompts))
    results: list[str] = [""] * len(prompts)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        future_to_idx = {pool.submit(fn, p): i for i, p in enumerate(prompts)}
        for fut in tqdm(
            as_completed(future_to_idx),
            total=len(prompts),
            desc=desc,
            leave=False,
            unit="req",
        ):
            results[future_to_idx[fut]] = fut.result()
    return results


# ======================================================================
# OpenAI API backend (gpt-5.4-mini, gpt-4o-mini, etc.)
# ======================================================================

class OpenAIInferenceEngine(BaseInferenceEngine):
    """
    Chat-completions backend for OpenAI models.

    GPT-5.x reasoning models ignore temperature/top_p — omit them from config.
    Standard chat models (gpt-4o-mini, etc.) accept them normally.
    """

    def __init__(self, model_key: str, config_path: str | Path) -> None:
        super().__init__(model_key, config_path)

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise InferenceEngineError(
                "The `openai` package is required for the api backend. "
                "Install with: pip install openai"
            ) from exc

        api_key = self._resolve_api_key(self.model_cfg)

        self._client = OpenAI(
            api_key=api_key,
            timeout=float(self.model_cfg.get("request_timeout", 180)),
            max_retries=int(self.model_cfg.get("max_retries", 6)),
        )
        self._model_name = str(self.model_cfg["path"])
        self._max_new_tokens = int(self.model_cfg.get("max_new_tokens", 4096))
        self._reasoning_effort = self.model_cfg.get("reasoning_effort")
        self._system_prompt = self.model_cfg.get("system_prompt")
        self._concurrency = max(1, int(self.model_cfg.get("concurrency", 8)))

        gen_cfg = dict(self.model_cfg.get("generation", {}))
        self._temperature = gen_cfg.get("temperature")
        self._top_p = gen_cfg.get("top_p")

        self._tokenizer = _PassthroughTokenizer()

        print(
            f"OpenAI engine ready: model={self._model_name} | "
            f"reasoning_effort={self._reasoning_effort} | "
            f"max_new_tokens={self._max_new_tokens} | "
            f"concurrency={self._concurrency}"
        )

    @staticmethod
    def _resolve_api_key(cfg: dict[str, Any]) -> str:
        """Read OPENAI_API_KEY from env, otherwise from .env in cfg['api_key_dir']."""
        key = os.environ.get("OPENAI_API_KEY")
        if key:
            return key
        key_dir = cfg.get("api_key_dir")
        if not key_dir:
            raise InferenceEngineError(
                "OPENAI_API_KEY is not set and no 'api_key_dir' in model config."
            )
        key_dir_path = Path(str(key_dir)).expanduser().resolve()
        if not key_dir_path.exists():
            raise InferenceEngineError(f"api_key_dir does not exist: {key_dir_path}")
        env_file = key_dir_path / ".env"
        key = _parse_env_file_for_key(env_file, "OPENAI_API_KEY")
        if not key:
            raise InferenceEngineError(
                f"OPENAI_API_KEY not found in environment or in {env_file}. "
                "Expected a line like: OPENAI_API_KEY=sk-..."
            )
        os.environ.setdefault("OPENAI_API_KEY", key)
        return key

    @property
    def tokenizer(self) -> Any:
        return self._tokenizer

    def _generate_one(self, prompt: str) -> str:
        messages: list[dict[str, str]] = []
        if self._system_prompt:
            messages.append({"role": "system", "content": self._system_prompt})
        messages.append({"role": "user", "content": prompt})

        kwargs: dict[str, Any] = {
            "model": self._model_name,
            "messages": messages,
            "max_completion_tokens": self._max_new_tokens,
        }
        if self._reasoning_effort:
            kwargs["reasoning_effort"] = self._reasoning_effort
        if self._temperature is not None:
            kwargs["temperature"] = float(self._temperature)
        if self._top_p is not None:
            kwargs["top_p"] = float(self._top_p)

        resp = self._client.chat.completions.create(**kwargs)
        if not resp.choices:
            return ""
        return resp.choices[0].message.content or ""

    def generate(self, prompts: list[str]) -> list[str]:
        if not prompts:
            return []
        return _run_parallel_with_progress(
            self._generate_one,
            prompts,
            self._concurrency,
            desc=self._model_name,
        )


# ======================================================================
# Anthropic API backend (claude-opus-4-7, claude-sonnet-4-6, etc.)
# ======================================================================

class AnthropicInferenceEngine(BaseInferenceEngine):
    """
    Messages API backend for Anthropic Claude models.

    Config fields:
      - path:           Claude model ID, e.g. "claude-opus-4-7"
      - max_new_tokens: int, mapped to the API's max_tokens
      - thinking:       dict passed through to the API's `thinking` field.
                        Opus 4.7+:  {type: "adaptive"}
                        Opus 4.6 and older: {type: "enabled", budget_tokens: N}
      - effort:         optional str in {"low","medium","high","xhigh","max"}.
                        Sent as output_config={"effort": ...}.
      - generation:     {temperature, top_p} — only sent if thinking is off.
                        Opus 4.7 rejects non-default values; omit for 4.7.
    """

    def __init__(self, model_key: str, config_path: str | Path) -> None:
        super().__init__(model_key, config_path)

        try:
            from anthropic import Anthropic
        except ImportError as exc:
            raise InferenceEngineError(
                "The `anthropic` package is required for the anthropic backend. "
                "Install with: pip install anthropic"
            ) from exc

        api_key = self._resolve_api_key(self.model_cfg)

        self._client = Anthropic(
            api_key=api_key,
            timeout=float(self.model_cfg.get("request_timeout", 180)),
            max_retries=int(self.model_cfg.get("max_retries", 6)),
        )
        self._model_name = str(self.model_cfg["path"])
        self._max_new_tokens = int(self.model_cfg.get("max_new_tokens", 4096))
        self._system_prompt = self.model_cfg.get("system_prompt")
        self._concurrency = max(1, int(self.model_cfg.get("concurrency", 8)))

        gen_cfg = dict(self.model_cfg.get("generation", {}))
        self._temperature = gen_cfg.get("temperature")
        self._top_p = gen_cfg.get("top_p")

        thinking_cfg = self.model_cfg.get("thinking")
        self._thinking = dict(thinking_cfg) if thinking_cfg else None
        self._effort = self.model_cfg.get("effort")

        self._tokenizer = _PassthroughTokenizer()

        print(
            f"Anthropic engine ready: model={self._model_name} | "
            f"thinking={self._thinking} | "
            f"effort={self._effort} | "
            f"max_new_tokens={self._max_new_tokens} | "
            f"concurrency={self._concurrency}"
        )

    @staticmethod
    def _resolve_api_key(cfg: dict[str, Any]) -> str:
        """Read ANTHROPIC_API_KEY from env, otherwise from .env in cfg['api_key_dir']."""
        key = os.environ.get("ANTHROPIC_API_KEY")
        if key:
            return key
        key_dir = cfg.get("api_key_dir")
        if not key_dir:
            raise InferenceEngineError(
                "ANTHROPIC_API_KEY is not set and no 'api_key_dir' in model config."
            )
        key_dir_path = Path(str(key_dir)).expanduser().resolve()
        if not key_dir_path.exists():
            raise InferenceEngineError(f"api_key_dir does not exist: {key_dir_path}")
        env_file = key_dir_path / ".env"
        key = _parse_env_file_for_key(env_file, "ANTHROPIC_API_KEY")
        if not key:
            raise InferenceEngineError(
                f"ANTHROPIC_API_KEY not found in environment or in {env_file}. "
                "Expected a line like: ANTHROPIC_API_KEY=sk-ant-..."
            )
        os.environ.setdefault("ANTHROPIC_API_KEY", key)
        return key

    @property
    def tokenizer(self) -> Any:
        return self._tokenizer

    def _generate_one(self, prompt: str) -> str:
        kwargs: dict[str, Any] = {
            "model": self._model_name,
            "max_tokens": self._max_new_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if self._system_prompt:
            kwargs["system"] = self._system_prompt
        if self._thinking is not None:
            kwargs["thinking"] = self._thinking
        if self._effort:
            kwargs["output_config"] = {"effort": self._effort}
        if self._thinking is None:
            if self._temperature is not None:
                kwargs["temperature"] = float(self._temperature)
            if self._top_p is not None:
                kwargs["top_p"] = float(self._top_p)

        resp = self._client.messages.create(**kwargs)
        return "".join(
            block.text for block in resp.content
            if getattr(block, "type", None) == "text"
        )

    def generate(self, prompts: list[str]) -> list[str]:
        if not prompts:
            return []
        return _run_parallel_with_progress(
            self._generate_one,
            prompts,
            self._concurrency,
            desc=self._model_name,
        )


# ======================================================================
# Factory
# ======================================================================

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
    if backend in {"api", "openai"}:
        return OpenAIInferenceEngine(model_key, config_path)
    if backend == "anthropic":
        return AnthropicInferenceEngine(model_key, config_path)

    raise InferenceEngineError(
        f"Unsupported backend '{backend}' for model '{model_key}'. "
        f"Supported: ['hf', 'vllm', 'api', 'anthropic']"
    )

    


# from __future__ import annotations
# from abc import ABC, abstractmethod
# from pathlib import Path
# from typing import Any

# import torch
# import yaml
# from transformers import AutoTokenizer

# from model_loader import load_model


# class InferenceEngineError(Exception):
#     """Raised when an inference backend cannot be initialized or used."""


# def _read_models_config(config_path: str | Path) -> dict[str, Any]:
#     config_path = Path(config_path).expanduser().resolve()
#     if not config_path.exists():
#         raise FileNotFoundError(f"Model config not found: {config_path}")
#     with config_path.open("r", encoding="utf-8") as handle:
#         data = yaml.safe_load(handle)
#     if not isinstance(data, dict):
#         raise InferenceEngineError("Top-level models YAML must be a mapping")
#     models = data.get("models")
#     if not isinstance(models, dict) or not models:
#         raise InferenceEngineError("Expected non-empty 'models' mapping in models_config.yaml")
#     return data


# def _get_model_cfg(model_key: str, config_path: str | Path) -> dict[str, Any]:
#     config = _read_models_config(config_path)
#     models = config["models"]
#     if model_key not in models:
#         raise InferenceEngineError(
#             f"Unknown model '{model_key}'. Available: {sorted(models.keys())}"
#         )
#     cfg = models[model_key]
#     if not isinstance(cfg, dict):
#         raise InferenceEngineError(f"Model entry '{model_key}' must be a mapping")
#     return cfg


# class BaseInferenceEngine(ABC):
#     def __init__(self, model_key: str, config_path: str | Path) -> None:
#         self.model_key = model_key
#         self.config_path = Path(config_path).expanduser().resolve()
#         self.model_cfg = _get_model_cfg(model_key, self.config_path)

#     @property
#     @abstractmethod
#     def tokenizer(self) -> Any:
#         raise NotImplementedError

#     @abstractmethod
#     def generate(self, prompts: list[str]) -> list[str]:
#         raise NotImplementedError


# class HFInferenceEngine(BaseInferenceEngine):
#     def __init__(self, model_key: str, config_path: str | Path) -> None:
#         super().__init__(model_key, config_path)
#         self.model, self._tokenizer, self.model_cfg = load_model(
#             model_key, config_path=self.config_path
#         )
#         self.model.eval()
#         self.generation_kwargs = {
#             "max_new_tokens": int(self.model_cfg["max_new_tokens"]),
#             **dict(self.model_cfg.get("generation", {})),
#         }
#         self._device = next(self.model.parameters()).device

#     @property
#     def tokenizer(self) -> Any:
#         return self._tokenizer

#     def generate(self, prompts: list[str]) -> list[str]:
#         if not prompts:
#             return []
#         inputs = self._tokenizer(prompts, return_tensors="pt", padding=True).to(self._device)
#         with torch.no_grad():
#             output_ids = self.model.generate(
#                 **inputs,
#                 pad_token_id=self._tokenizer.pad_token_id,
#                 **self.generation_kwargs,
#             )
#         prompt_width = inputs["input_ids"].shape[1]
#         new_token_ids = output_ids[:, prompt_width:]
#         return self._tokenizer.batch_decode(
#             new_token_ids,
#             skip_special_tokens=True,
#             clean_up_tokenization_spaces=False,
#         )


# class VLLMInferenceEngine(BaseInferenceEngine):
#     """
#     VLLM for inference.
#     """
#     def __init__(self, model_key: str, config_path: str | Path) -> None:
#         super().__init__(model_key, config_path)

#         try:
#             from vllm import LLM, SamplingParams
#         except ImportError as exc:
#             raise InferenceEngineError("vLLM is not installed") from exc

#         self._LLM = LLM
#         self._SamplingParams = SamplingParams

#         family = str(self.model_cfg.get("family", "")).strip().lower()
#         trust_remote_code = family in {"qwen", "deepseek", "gptoss"}

#         self._tokenizer = AutoTokenizer.from_pretrained(
#             self.model_cfg["path"],
#             trust_remote_code=trust_remote_code,
#         )
#         self._tokenizer.padding_side = "left"
#         if self._tokenizer.pad_token_id is None and self._tokenizer.eos_token is not None:
#             self._tokenizer.pad_token = self._tokenizer.eos_token

#         llm_kwargs: dict[str, Any] = {
#             "model": self.model_cfg["path"],
#             "tensor_parallel_size": int(self.model_cfg.get("tensor_parallel_size", 1)),
#             "dtype": self.model_cfg.get("dtype", "auto"),
#             "gpu_memory_utilization": float(self.model_cfg.get("gpu_memory_utilization", 0.92)),
#             "enforce_eager": bool(self.model_cfg.get("enforce_eager", False)),
#             "enable_prefix_caching": bool(self.model_cfg.get("enable_prefix_caching", True)),
#             "enable_chunked_prefill": bool(self.model_cfg.get("enable_chunked_prefill", True)),
#             "max_num_batched_tokens": int(self.model_cfg.get("max_num_batched_tokens", 32768)),
#             "max_num_seqs": int(self.model_cfg.get("max_num_seqs", 256)),
#         }

#         if "max_model_len" in self.model_cfg:
#             llm_kwargs["max_model_len"] = int(self.model_cfg["max_model_len"])

#         quant_cfg = dict(self.model_cfg.get("quantization", {}))
#         if quant_cfg.get("load_in_4bit", False):
#             llm_kwargs["quantization"] = "bitsandbytes"

#         self.llm = self._LLM(**llm_kwargs)

#         # Sampling (compatible with old + new vLLM)
#         gen_cfg = dict(self.model_cfg.get("generation", {}))
#         self.sampling_params = self._SamplingParams(
#             max_tokens=int(self.model_cfg["max_new_tokens"]),
#             temperature=float(gen_cfg.get("temperature", 0.0)),
#             repetition_penalty=float(gen_cfg.get("repetition_penalty", 1.0)),
#             top_p=1.0,
#         )

#     @property
#     def tokenizer(self) -> Any:
#         return self._tokenizer

#     def generate(self, prompts: list[str]) -> list[str]:
#         if not prompts:
#             return []
#         outputs = self.llm.generate(prompts, self.sampling_params)
#         return [out.outputs[0].text if out.outputs else "" for out in outputs]


# def build_inference_engine(
#     model_key: str,
#     config_path: str | Path,
# ) -> BaseInferenceEngine:
#     cfg = _get_model_cfg(model_key, config_path)
#     backend = str(cfg.get("backend", "hf")).strip().lower()

#     if backend == "hf":
#         return HFInferenceEngine(model_key, config_path)
#     if backend == "vllm":
#         return VLLMInferenceEngine(model_key, config_path)

#     raise InferenceEngineError(
#         f"Unsupported backend '{backend}' for model '{model_key}'. "
#         f"Supported: ['hf', 'vllm']"
#     )