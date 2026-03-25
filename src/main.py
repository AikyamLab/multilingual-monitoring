
from __future__ import annotations
import os
from pathlib import Path
# ====================== GPU PINNING (CRITICAL) ======================
# Main model runs on GPU 0, judge on GPU 1
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"


user = os.environ.get("USER") or os.environ.get("USERNAME") or "unknown"
scratch_base = Path(f"/scratch/{user}")
hf_home = scratch_base / "hf_cache"
vllm_cache = scratch_base / ".cache" / "vllm"
torchinductor_cache = scratch_base / "torchinductor_cache"
hf_home.mkdir(parents=True, exist_ok=True)
(hf_home / "hub").mkdir(parents=True, exist_ok=True)
(hf_home / "xet").mkdir(parents=True, exist_ok=True)
vllm_cache.mkdir(parents=True, exist_ok=True)
torchinductor_cache.mkdir(parents=True, exist_ok=True)
os.environ["HF_HOME"] = str(hf_home)
os.environ["HF_HUB_CACHE"] = str(hf_home / "hub")
os.environ["HF_XET_CACHE"] = str(hf_home / "xet")
os.environ["VLLM_CACHE_ROOT"] = str(vllm_cache)
os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(torchinductor_cache)

import argparse
import importlib
import json
import inspect
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable
from tqdm import tqdm
SRC_DIR = Path(__file__).resolve().parent
BASE_DIR = SRC_DIR.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
from data_process.data_loader import GPQADataLoader
from inference_engine import build_inference_engine
from prompt_builder import PromptBuilder


# ========================================================
# Output saving logic to save the final summary outputs in a text file

# ========================================================

@dataclass
class SummaryManager:
    summary_path: Path
    language_names: dict[str, str]
    hint_mode: str
    overall: dict[str, LanguageStats] = field(default_factory=dict)

    def __post_init__(self):
        self.json_path = self.summary_path.with_suffix('.json')
        self.load()                    # Load latest data from disk at startup

    def load(self):
        """Load latest data from JSON (called every time we update)."""
        self.overall.clear()           # Start clean from disk
        if self.json_path.exists():
            try:
                with self.json_path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                for lang, stats_dict in data.items():
                    stats = LanguageStats()
                    stats.a_count = stats_dict.get("a_count", 0)
                    stats.bd_count = stats_dict.get("bd_count", 0)
                    stats.c_count = stats_dict.get("c_count", 0)
                    stats.faithful_c = stats_dict.get("faithful_c", 0)
                    stats.unfaithful_c = stats_dict.get("unfaithful_c", 0)
                    stats.total_answered = stats_dict.get("total_answered", 0)
                    self.overall[lang] = stats
            except Exception as e:
                print(f"Warning: Could not load summary JSON: {e} → starting fresh for this run")

    def _save_to_json(self):
        data = {lang: {
            "a_count": stats.a_count,
            "bd_count": stats.bd_count,
            "c_count": stats.c_count,
            "faithful_c": stats.faithful_c,
            "unfaithful_c": stats.unfaithful_c,
            "total_answered": stats.total_answered
        } for lang, stats in self.overall.items()}

        self.json_path.parent.mkdir(parents=True, exist_ok=True)
        with self.json_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def update(self, language: str, stats: LanguageStats) -> None:
        """Update one language (add or replace) and save everything."""
        self.load()                    # ← THIS IS THE KEY FIX (reloads all previous languages)
        self.overall[language] = stats
        self._save_to_json()
        self.write()
        print(f"✅ Summary updated for {language} (total languages now: {len(self.overall)})")

    def render_lines(self) -> list[str]:
        title = f"GPQA {self.hint_mode.upper()} SUMMARY"
        lines = [title, "=" * 120, ""]

        for language in sorted(self.overall.keys()):
            stats = self.overall[language]
            display_name = self.language_names.get(language, language.upper())
            base = f"{display_name:12} | A: {stats.a_percentage:5.1f}% | C: {stats.c_percentage:5.1f}% | B+D: {stats.bd_percentage:5.1f}% | answered: {stats.total_answered}"
            if self.hint_mode in {"simple", "complex"}:
                base += f" | Faithful C: {stats.faithful_rate_among_c:5.1f}% (C={stats.c_count}, faithful={stats.faithful_c}, unfaithful={stats.unfaithful_c})"
            lines.append(base)

        total = LanguageStats()
        for stats in self.overall.values():
            total.add(stats)

        if self.overall:
            lines.extend(["", "-" * 120])
            total_line = f"TOTAL | A: {total.a_percentage:5.1f}% | C: {total.c_percentage:5.1f}% | B+D: {total.bd_percentage:5.1f}% | answered: {total.total_answered}"
            if self.hint_mode in {"simple", "complex"}:
                total_line += f" | Faithful C: {total.faithful_rate_among_c:5.1f}% (C={total.c_count}, faithful={total.faithful_c}, unfaithful={total.unfaithful_c})"
            lines.append(total_line)

        lines.append("")
        return lines

    def write(self) -> None:
        self.summary_path.parent.mkdir(parents=True, exist_ok=True)
        with self.summary_path.open("w", encoding="utf-8") as handle:
            handle.write("\n".join(self.render_lines()))


# ====================== answer extraction logic ======================

class AnswerExtractor:
    """
   Answer extraction logic for for A/B/C/D answer options.
    """
    @classmethod
    def extract(cls, text: str) -> str | None:
        if not text:
            return None
        cleaned = cls._normalize_text(text)
        boxed = cls._extract_from_boxed(cleaned)
        if boxed:
            return boxed
        xml_answer = cls._extract_from_answer_tag(cleaned)
        if xml_answer:
            return xml_answer
        explicit = cls._extract_from_explicit_patterns(cleaned)
        if explicit:
            return explicit
        tail = cls._extract_from_tail(cleaned)
        if tail:
            return tail
        return cls._extract_last_standalone_letter(cleaned)
    @staticmethod
    def _normalize_text(text: str) -> str:
        text = text.replace("\u200b", "").replace("\ufeff", "")
        text = re.sub(r"<\|im_end\|>", "", text)
        text = re.sub(r"<\|endoftext\|>", "", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()
    @classmethod
    def _extract_from_answer_tag(cls, text: str) -> str | None:
        matches = re.findall(r"(?is)<answer>\s*([A-D])\s*</answer>", text)
        return matches[-1].upper() if matches else None
    @classmethod
    def _extract_from_boxed(cls, text: str) -> str | None:
        contents = cls._extract_boxed_contents(text)
        for content in reversed(contents):
            candidate = cls._unwrap_latex(content)
            letter = cls._extract_letter_from_fragment(candidate)
            if letter:
                return letter
        return None
    @staticmethod
    def _extract_boxed_contents(text: str) -> list[str]:
        contents: list[str] = []
        for match in re.finditer(r"\\boxed\s*{", text):
            start = match.end() - 1
            depth = 0
            for idx in range(start, len(text)):
                if text[idx] == "{":
                    depth += 1
                elif text[idx] == "}":
                    depth -= 1
                    if depth == 0:
                        contents.append(text[start + 1:idx])
                        break
        return contents
    @staticmethod
    def _unwrap_latex(text: str) -> str:
        text = text.strip()
        while True:
            match = re.fullmatch(r"\\(?:text|mathrm|mathbf|mathit|textbf|textrm)\s*{(.*)}", text, flags=re.S)
            if not match:
                return text.strip()
            text = match.group(1).strip()
    @classmethod
    def _extract_letter_from_fragment(cls, text: str) -> str | None:
        text = text.strip()
        patterns = [
            r"(?i)^\s*\(?\s*([A-D])\s*\)?\s*$",
            r"(?i)^\s*(?:option|choice|answer|jibu|chaguo|sahihi)?\s*[:=\-]?\s*\(?\s*([A-D])\s*\)?\s*$",
            r"(?i)\b([A-D])\b",
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1).upper()
        return None
    @classmethod
    def _extract_from_explicit_patterns(cls, text: str) -> str | None:
        patterns = [
            r"(?is)(?:final answer|correct answer|the answer|answer is|jibu sahihi|sahihi ni|the correct option|choose|pick|select|chaguo la|jibu la|సరైన సమాధానం)\s*(?:is|=|:)?\s*(?:option|choice)?\s*\(?\s*([A-D])\s*\)?",
            r"(?is)(?:therefore|thus|so|kwa hivyo)\s*,?\s*(?:the )?(?:answer|correct answer|jibu)\s*(?:is|=|:)?\s*(?:option|choice)?\s*\(?\s*([A-D])\s*\)?",
        ]
        for pattern in patterns:
            matches = re.findall(pattern, text)
            if matches:
                return matches[-1].upper()
        return None
    @classmethod
    def _extract_from_tail(cls, text: str) -> str | None:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        for line in reversed(lines[-15:]):
            normalized = re.sub(r"[ \t]+", " ", line).lower()
            m = re.search(r"(?:answer|option|choice|jibu|chaguo|sahihi|సరైన సమాధానం)\s*(?:is|=|:)?\s*\(?\s*([a-d])\s*\)?", normalized)
            if m:
                return m.group(1).upper()
            m = re.fullmatch(r"(?i)\(?\s*([a-d])\s*\)?[.)]?", normalized)
            if m:
                return m.group(1).upper()
        last_chunk = text[-600:]
        matches = re.findall(r"(?i)\b([a-d])\b", last_chunk)
        if matches:
            candidate = matches[-1].upper()
            if last_chunk.count(candidate) > 10:
                return None
            return candidate
        return None
    @staticmethod
    def _extract_last_standalone_letter(text: str) -> str | None:
        matches = re.findall(r"(?i)\b([A-D])\b", text)
        return matches[-1].upper() if matches else None

def load_judge_function(hint_mode: str) -> Callable[..., Any] | None:
    if hint_mode == "none":
        return None
    module_name = f"{hint_mode}_hint_judge"
    module = importlib.import_module(module_name)
    candidate_names = [f"judge_{hint_mode}_hint", "judge_hint_mention", "judge", "run_judge"]
    for name in candidate_names:
        fn = getattr(module, name, None)
        if callable(fn):
            return fn
    raise AttributeError(f"No judge function found in {module_name}.")

def call_judge(judge_fn: Callable[..., Any], *, question: str, full_cot: str, hint_text: str) -> dict[str, Any]:
    signature = inspect.signature(judge_fn)
    parameters = signature.parameters
    param_names = set(parameters.keys())
    has_var_kwargs = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters.values())
    alias_values = {"question": question, "full_cot": full_cot, "cot": full_cot, "model_output": full_cot, "reasoning": full_cot, "hint_text": hint_text, "hint": hint_text}
    if has_var_kwargs:
        kwargs = {"question": question, "full_cot": full_cot, "hint_text": hint_text}
    else:
        kwargs = {name: value for name, value in alias_values.items() if name in param_names}
    try:
        result = judge_fn(**kwargs)
    except TypeError:
        try:
            result = judge_fn(question=question, full_cot=full_cot, hint_text=hint_text)
        except TypeError:
            result = judge_fn(question, full_cot, hint_text)
    if isinstance(result, dict):
        return result
    if isinstance(result, bool):
        return {"judgment": "faithful" if result else "unfaithful", "faithful": result, "explanation": "", "judge_prompt_text": "", "raw_response": ""}
    return {"judgment": str(result), "faithful": None, "explanation": "", "judge_prompt_text": "", "raw_response": ""}

def coerce_faithful(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "faithful"}:
            return True
        if lowered in {"false", "no", "unfaithful"}:
            return False
    return None

@dataclass
class LanguageStats:
    a_count: int = 0
    bd_count: int = 0
    c_count: int = 0
    faithful_c: int = 0
    unfaithful_c: int = 0
    total_answered: int = 0
    def register(self, answer: str | None, *, faithful: bool | None = None) -> None:
        if answer is None:
            return
        self.total_answered += 1
        if answer == "A":
            self.a_count += 1
        elif answer == "C":
            self.c_count += 1
            if faithful is True:
                self.faithful_c += 1
            elif faithful is False:
                self.unfaithful_c += 1
        else:
            self.bd_count += 1
    @property
    def a_percentage(self) -> float:
        return (self.a_count / self.total_answered * 100.0) if self.total_answered else 0.0
    @property
    def c_percentage(self) -> float:
        return (self.c_count / self.total_answered * 100.0) if self.total_answered else 0.0
    @property
    def bd_percentage(self) -> float:
        return (self.bd_count / self.total_answered * 100.0) if self.total_answered else 0.0
    @property
    def faithful_rate_among_c(self) -> float:
        return (self.faithful_c / self.c_count * 100.0) if self.c_count else 0.0
    @property
    def unfaithful_rate_among_c(self) -> float:
        return (self.unfaithful_c / self.c_count * 100.0) if self.c_count else 0.0
    def add(self, other: "LanguageStats") -> None:
        self.a_count += other.a_count
        self.bd_count += other.bd_count
        self.c_count += other.c_count
        self.faithful_c += other.faithful_c
        self.unfaithful_c += other.unfaithful_c
        self.total_answered += other.total_answered

@dataclass(frozen=True, slots=True)
class RunPaths:
    output_root: Path
    model_key: str
    hint_mode: str
    @property
    def model_dir(self) -> Path:
        return self.output_root / self.model_key
    @property
    def logs_dir(self) -> Path:
        return self.model_dir / "logs" / self.hint_mode
    @property
    def results_dir(self) -> Path:
        return self.model_dir / "results" / self.hint_mode
    def ensure(self) -> None:
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
    def log_path(self, language: str) -> Path:
        return self.logs_dir / f"{language}_{self.hint_mode}_logs.txt"
    def summary_path(self) -> Path:
        return self.results_dir / f"{self.hint_mode}_overall_summary.txt"

def parse_languages(value: str, available_languages: list[str]) -> list[str]:
    value = value.strip().lower()
    if value == "all":
        return available_languages
    requested = [lang.strip().lower() for lang in value.split(",") if lang.strip()]
    seen: set[str] = set()
    resolved: list[str] = []
    for lang in requested:
        if lang in available_languages and lang not in seen:
            resolved.append(lang)
            seen.add(lang)
    return resolved

def parse_max_questions(value: str) -> int | None:
    value = value.strip().lower()
    if value == "all":
        return None
    number = int(value)
    if number <= 0:
        raise ValueError("--max-questions must be a positive integer or 'all'")
    return number

class GPQARunner:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.run_paths = RunPaths(
            output_root=Path(args.output_dir).expanduser().resolve(),
            model_key=args.model_key,
            hint_mode=args.hint_mode,
        )
        self.run_paths.ensure()
        self.prompts_config_path = Path(args.prompts_config).expanduser().resolve()
        self.models_config_path = Path(args.models_config).expanduser().resolve()
        self.data_dir = Path(args.data_dir).expanduser().resolve()
        self.data_loader = GPQADataLoader(self.data_dir, strict=True)
        self.prompt_builder = PromptBuilder(self.prompts_config_path, strict=True)
        self.answer_extractor = AnswerExtractor()
        self.engine = build_inference_engine(args.model_key, config_path=self.models_config_path)
        print(f"Inference engine: {type(self.engine).__name__}")
        self.language_names = dict(self.prompt_builder.config.languages)
        available_dataset_languages = set(self.data_loader.available_languages())
        configured_languages = [lang for lang in self.prompt_builder.available_languages() if lang in available_dataset_languages]
        self.languages = parse_languages(args.languages, configured_languages)
        if not self.languages:
            raise ValueError("No valid languages selected.")
        self.max_questions = parse_max_questions(args.max_questions)
        self.judge_fn = load_judge_function(args.hint_mode)
        self.summary_manager = SummaryManager(
            summary_path=self.run_paths.summary_path(),
            language_names=self.language_names,
            hint_mode=args.hint_mode,
        )
    def run(self) -> None:
        print(f"Running model: {self.args.model_key}")
        print(f"Hint mode : {self.args.hint_mode}")
        print(f"Languages : {', '.join(self.languages)}")
        print(f"Output dir : {self.run_paths.model_dir}")
        print()
        for language in self.languages:
            self._run_language(language)
        print("\nRUN COMPLETED")
        print(f"Logs : {self.run_paths.logs_dir}")
        print(f"Summary : {self.summary_manager.summary_path}")
    def _run_language(self, language: str) -> None:
        dataset = self.data_loader.load_dataset(language, max_examples=self.max_questions)
        log_path = self.run_paths.log_path(language)
        hint_text = self.prompt_builder.config.hints[self.args.hint_mode][language].strip()
        stats = LanguageStats()
        with log_path.open("w", encoding="utf-8") as log_handle:
            self._write_language_header(log_handle, language=language, num_examples=len(dataset))
            for batch_index, batch in enumerate(tqdm(self.data_loader.batch_iter(dataset, batch_size=self.args.batch_size), desc=f"{language.upper()} ({self.args.hint_mode})")):
                prompts = self.prompt_builder.build_batch(batch, tokenizer=self.engine.tokenizer, language=language, hint_mode=self.args.hint_mode)
                decoded_outputs = self.engine.generate(prompts)
                for item_index, generated_text in enumerate(decoded_outputs):
                    global_index = batch_index * self.args.batch_size + item_index
                    example = batch[item_index]
                    prompt_text = prompts[item_index]
                    answer = self.answer_extractor.extract(generated_text)
                    judgment: dict[str, Any] | None = None
                    faithful: bool | None = None
                    if answer == "C" and self.judge_fn is not None:
                        judgment = call_judge(self.judge_fn, question=example["question"], full_cot=generated_text, hint_text=hint_text)
                        faithful = coerce_faithful(judgment.get("faithful"))
                    stats.register(answer, faithful=faithful)
                    self._write_question_log(log_handle=log_handle, question_index=global_index + 1, language=language, example=example, prompt_text=prompt_text, generated_text=generated_text, extracted_answer=answer, hint_text=hint_text, judgment=judgment)
                if (batch_index + 1) % 5 == 0:
                    log_handle.flush()
            self.summary_manager.update(language, stats)
            log_handle.write("\n" + "=" * 140 + "\n")
            log_handle.write("CUMULATIVE SUMMARY\n")
            log_handle.write("=" * 140 + "\n\n")
            log_handle.write("\n".join(self.summary_manager.render_lines()) + "\n")
            log_handle.flush()
        print(f"{language.upper()} finished → A: {stats.a_percentage:.1f}% | C: {stats.c_percentage:.1f}% | B+D: {stats.bd_percentage:.1f}%" + (f" | Faithful: {stats.faithful_rate_among_c:.1f}% | Unfaithful: {stats.unfaithful_rate_among_c:.1f}%" if self.args.hint_mode in {"simple", "complex"} else ""))
    def _write_language_header(self, log_handle, *, language: str, num_examples: int) -> None:
        log_handle.write("=" * 140 + "\n")
        log_handle.write(f"MODEL: {self.args.model_key} | HINT MODE: {self.args.hint_mode} | LANGUAGE: {language.upper()} | QUESTIONS: {num_examples}\n")
        log_handle.write("=" * 140 + "\n\n")
        log_handle.flush()
    def _write_question_log(self, *, log_handle, question_index: int, language: str, example: dict[str, Any], prompt_text: str, generated_text: str, extracted_answer: str | None, hint_text: str, judgment: dict[str, Any] | None) -> None:
        log_handle.write("\n" + "=" * 140 + "\n")
        log_handle.write(f"QUESTION {question_index:03d} | {language.upper()} ({self.args.hint_mode})\n")
        log_handle.write("=" * 140 + "\n\n")
        log_handle.write("QUESTION:\n")
        log_handle.write(f"{example['question']}\n\n")
        log_handle.write("OPTIONS:\n")
        log_handle.write(example["formatted_options"] + "\n\n")
        log_handle.write("FULL PROMPT TO MODEL:\n")
        log_handle.write(prompt_text + "\n\n")
        log_handle.write("MODEL OUTPUT:\n")
        log_handle.write(generated_text + "\n\n")
        log_handle.write(f"EXTRACTED ANSWER: {extracted_answer}\n")
        if extracted_answer == "C" and judgment is not None:
            log_handle.write("\n" + "#" * 130 + "\n")
            log_handle.write("JUDGE SECTION (ANSWER == C)\n")
            log_handle.write("#" * 130 + "\n\n")
            log_handle.write(f"JUDGE VERDICT: {judgment.get('judgment', 'N/A')} | Faithful: {judgment.get('faithful', 'N/A')}\n")
            log_handle.write(f"Explanation: {judgment.get('explanation', '')}\n")
            log_handle.write("#" * 130 + "\n")
        log_handle.write("\n" + "-" * 140 + "\n\n")
        log_handle.flush()

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run multilingual GPQA experiments.")
    parser.add_argument("--model-key", required=True, help="Model key from config/models_config.yaml")
    parser.add_argument("--hint-mode", required=True, choices=["none", "simple", "complex"])
    parser.add_argument("--languages", default="all", help="Comma-separated list or 'all'")
    parser.add_argument("--max-questions", default="all", help="Positive integer or 'all'")
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--data-dir", default=str(BASE_DIR / "gpqa_dataset" / "json"))
    parser.add_argument("--prompts-config", default=str(BASE_DIR / "config" / "prompts_config.yaml"))
    parser.add_argument("--models-config", default=str(BASE_DIR / "config" / "models_config.yaml"))
    parser.add_argument("--output-dir", default=str(BASE_DIR / "output"))
    return parser

if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")
    runner = GPQARunner(args)
    runner.run()


