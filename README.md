<h1 align="center">The Fragility of Chain-of-Thought Monitoring Across Typologically Diverse Languages</h1>

<p align="center">
  <a href="https://multilingual-cot-monitoring.github.io/"><img src="https://img.shields.io/badge/Website-Project_Page-1f6feb?style=flat&labelColor=222222"></a>
  <a href="#"><img src="https://img.shields.io/badge/arXiv-coming_soon-b31b1b?style=flat&labelColor=222222"></a>
</p>

<p align="center"><em>by</em></p>

<p align="center">
  <b>Eric Onyame</b><sup>1</sup> &nbsp;&middot;&nbsp;
  <b>Runtao Zhou</b><sup>1</sup> &nbsp;&middot;&nbsp;
  <b>Kowshik Thopalli</b><sup>2</sup> &nbsp;&middot;&nbsp;
  <b>Bhavya Kailkhura</b><sup>2</sup> &nbsp;&middot;&nbsp;
  <b>Chirag Agarwal</b><sup>1</sup>
</p>

<p align="center">
  <sup>1</sup>University of Virginia &nbsp;&nbsp;&nbsp; <sup>2</sup>Lawrence Livermore National Laboratory
</p>

---

This repository hosts the codebase for **The Fragility of Chain-of-Thought Monitoring Across Typologically Diverse Languages**, the first large-scale evaluation of CoT monitorability under linguistic distribution shift. Below, we provide an overview of the project along with key implementation details.

## Overview

Chain-of-thought (CoT) monitoring has been proposed as a promising safety mechanism for detecting misaligned behavior in large language models — yet its reliability has remained largely unexplored beyond English and across diverse model families. We present the first large-scale evaluation of CoT monitorability across **13 typologically diverse languages** and **seven frontier model families** (16 models total, spanning 8B–120B parameters), using adversarial-hint evaluations that require explicit intermediate computation together with logit-lens analysis of internal answer-token probabilities.

We find that CoT unfaithfulness persists consistently across languages and hint types, with an **average deception rate of 95.9%**. Frontier models systematically engage in strategic manipulation — including answer-switching, post-hoc rationalization, and **procedural exploitation of hints** — making external monitors struggle to detect deception. Logit-lens analysis further reveals that models often commit to the misaligned cue in their latent activations **within the first 15% of generation**, even when the CoT appears faithful. Strikingly, these deceptive patterns **saturate at 100% in low-resource languages**, revealing fundamental limitations in current CoT-based oversight.

Our results show that CoT monitoring is fundamentally fragile under linguistic distribution shift, providing a substantially weaker safety signal than what English-only studies suggest. These findings underscore an urgent need to develop robust CoT monitors and to accelerate research into white-box monitoring techniques — especially to improve CoT monitorability in mid- and low-resource languages.

<!-- For full technical details and experiments, see the [project website](https://multilingual-cot-monitoring.github.io/). -->

## Key Figure

<p align="center">
  <img src="figures/figure1_deceptive_reasoning.png" alt="Example of deceptive hint-driven reasoning" width="85%">
</p>

<p align="center"><em>
<b>Figure 1. Example of deceptive hint-driven reasoning.</b> Qwen3-32B on a GPQA chemistry question (Rein et al., 2024). The model first gives the correct nucleophilicity analysis, explicitly endorses option A, and rules out option C on substantive chemistry grounds. It then applies the injected hint formula (K+Q) mod 4 using fabricated values for K and Q, overriding its own answer and submitting option C. <b>Green</b>: correct domain reasoning. <b>Blue</b>: explicit endorsement of the correct answer. <b>Orange</b>: substantive elimination of C. <b>Red</b>: fabricated K, Q values used to justify the hinted answer. Hint target: <b>C</b>; gold answer: <b>A</b>.
</em></p>

<p align="center">High-resolution PDF: <a href="assets/figure1_deceptive_reasoning.pdf">Figure 1</a></p>

For more examples of this deceptive behavior, see the [project website](https://multilingual-cot-monitoring.github.io/) and our paper (arXiv coming soon)




## Codebase Overview

### `config/` — Configuration
All experiment knobs live here as YAML. We separate **what to run** (models, judges) from **how to phrase it** (prompts, hints) so that adding a new model or a new language never requires touching the inference code.

- **`models_config.yaml`** — Specs for every model under evaluation (Hugging Face IDs or API model names, family, backend, generation parameters, quantization options, and API-key locations for closed models).
- **`judges_config.yaml`** — Specs for the CoT-monitor (judge) models used to score whether a model's chain-of-thought is faithful or deceptive. Each judge entry declares its `type` (`open` or `closed`) and is paired with a hint mode (`simple_hint_judge` or `complex_hint_judge`).
- **`prompts_config.yaml`** — Holds all language-dependent text in one place. Three top-level blocks:
  - `languages` — maps language codes (`english`, `swahili`, `telugu`, …) to display names. 13 typologically diverse languages are supported.
  - `hints` — adversarial hint text per language, organized by hint mode (`none` / `simple` / `complex`).
  - `translations` — per-language prompt components (`system`, `instruction`, `question_label`, `think_in`, `hacking_starter`) consumed by the prompt builder.

### `data_process/` — Data loading
- **`data_loader.py`** — Defines `GPQAExample` (a frozen dataclass with language, question, and four answer options) and `GPQADataLoader` for streaming. Option A is always the gold answer; B/C/D are distractors. Hint text is **not** stored on the example — it is injected later by the prompt builder, so the same dataset can be reused across hint modes without re-loading.

### `gpqa_dataset/` — Evaluation data
The 13-language GPQA splits used in the paper, stored as one `.jsonl` file per language under `gpqa_dataset/json/`. Each split contains the same questions, translated and adapted per language.

### `src/` — Experiment core
- **`main.py`** — Single entry point for the open- and closed-model pipelines. Pins the model under test to GPU 0 and the judge to GPU 1, sets up Hugging Face / vLLM / torchinductor caches in scratch space, parses CLI arguments, loads the YAML configs, and dispatches to the appropriate inference backend. It extracts the final answer from each generation (`AnswerExtractor`), invokes the configured judge whenever the model lands on the hinted option, and writes per-language logs plus cumulative summaries (`SummaryManager`, `RunPaths`).
- **`inference_engine.py`** — Factory and implementations for four generation backends behind a common `BaseInferenceEngine` interface:
  - `HFInferenceEngine` — Hugging Face Transformers (uses `model_loader.py`).
  - `VLLMInferenceEngine` — vLLM backend for high-throughput open-weights inference.
  - `OpenAIInferenceEngine` — OpenAI chat-completions backend (e.g., `gpt-5-mini`, `gpt-4o-mini`).
  - `AnthropicInferenceEngine` — Anthropic messages backend (e.g., `claude-opus-4-7`, `claude-sonnet-4-6`).

  The factory picks the backend from the `backend` field in `models_config.yaml`. API backends read keys from the environment or from a local `.env` file pointed to by `api_key_dir`.
- **`model_loader.py`** — Hugging Face loader for the `HFInferenceEngine` path. Reads quantization, dtype, and attention settings from `models_config.yaml` and dispatches to family-specific loaders (`qwen`, `deepseek`, `gemma`). New families are added by registering a `_load_<family>(cfg)` function in `_FAMILY_LOADERS`.
- **`prompt_builder.py`** — Composes the final prompt for a given `(language, hint_mode, example)` triple by stitching multilingual templates and hint strings from `prompts_config.yaml`. Adversarial hints are inserted **after** the question and options. A per-language "thinking starter" is then appended inside an opened `<think>` block to keep the model's chain-of-thought in the chosen language.
- **`closed_judge/` and `open_judge/`** — Two implementations of the CoT monitor sharing a common interface. `closed_judge/` wraps closed-source API models as the judge; `open_judge/` runs open-weights judges locally. Each side provides `simple_hint_judge.py` and `complex_hint_judge.py`; the runner picks which to import based on the hint mode and the `type` field in `judges_config.yaml`.
- **`gpt-oss_20b_inference.py` / `gpt-oss_120b_inference.py`** — Stand-alone scripts for the GPT-OSS family, kept separate from `main.py` because these models require Mxfp4 quantization, a developer/user chat format, and runtime attention-implementation selection that doesn't fit the unified backend abstraction.
- **`results/` and `logs/`** — Generated artifacts. Populated when you run experiments locally; not tracked in git.

### `figures/` — Paper figures
Source files (PDF / PNG) for the figures used in the paper and on this README.