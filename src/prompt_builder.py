from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml
from transformers import AutoTokenizer


class PromptConfigError(Exception):
    """Raised when prompts_config.yaml is malformed or incomplete."""


@dataclass(frozen=True, slots=True)
class LanguagePromptSpec:
    system: str
    instruction: str
    question_label: str
    think_in: str
    hacking_starter: str


@dataclass(frozen=True, slots=True)
class PromptConfig:
    languages: dict[str, str]
    hints: dict[str, dict[str, str]]
    translations: dict[str, LanguagePromptSpec]


class PromptBuilder:
    """
    Builds multilingual prompts from:
      - prompts_config.yaml
      - prepared dataset items from data_loader.py
      - selected hint mode
    Design choices:
      - hint appears AFTER the question + options (complex and simple)
      - the hacker prefix is always used
      - the hacker prefix body starts inside <think>
      - prompt logic stays separate from data loading and model loading
    """

    REQUIRED_TRANSLATION_FIELDS = (
        "system",
        "instruction",
        "question_label",
        "think_in",
        "hacking_starter",
    )

    def __init__(
        self,
        config_path: str | Path,
        *,
        default_language: str = "english",
        strict: bool = True,
    ) -> None:
        self.config_path = Path(config_path).expanduser().resolve()
        self.default_language = default_language.strip().lower()
        self.strict = strict
        self.config = self._load_config()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def available_languages(self) -> tuple[str, ...]:
        return tuple(sorted(self.config.languages.keys()))

    def available_hint_modes(self) -> tuple[str, ...]:
        return tuple(sorted(self.config.hints.keys()))

    def build_messages(
        self,
        example: Mapping[str, Any] | Any,
        *,
        language: str,
        hint_mode: str,
    ) -> list[dict[str, str]]:
        """
        Build structured chat messages with the hint placed AFTER the question + options.
        
        Order:
          1. Instruction
          2. Question label + question
          3. Options
          4. Hint (wrapped in <answer> for complex mode, plain for simple mode)
          5. Think-in instruction
        """
        language = self._resolve_language(language)
        hint_mode = self._resolve_hint_mode(hint_mode)
        spec = self.config.translations[language]
        hint_text = self._get_hint(language=language, hint_mode=hint_mode)
        question = self._extract_field(example, "question")
        formatted_options = self._get_formatted_options(example)

        user_parts: list[str] = [
            spec.instruction,
            f"{spec.question_label}: {question}",
            formatted_options,
        ]

        # === HINT PLACEMENT (AFTER QUESTION + OPTIONS) ===
        if hint_text:
            hint_text = hint_text.strip()
            if hint_mode == "complex":
                user_parts.append(f"<answer>{hint_text}</answer>")
            else:
                user_parts.append(hint_text)          # simple hint stays clean

        user_parts.append(spec.think_in)

        user_content = "\n\n".join(part for part in user_parts if part.strip())

        return [
            {"role": "system", "content": spec.system},
            {"role": "user", "content": user_content},
        ]

    def build_prompt(
        self,
        example: Mapping[str, Any] | Any,
        *,
        tokenizer: Any,
        language: str,
        hint_mode: str,
    ) -> str:
        """
        Build the final rendered prompt.

        Strategy:
          1. Render system+user via the tokenizer chat template with
             add_generation_prompt=True.
          2. Insert the hacker prefix body inside <think>.
             - If the rendered prompt already ends with an open <think>,
               append only the hacker body.
             - Otherwise open <think> manually, then append the hacker body.
        """
        language = self._resolve_language(language)
        spec = self.config.translations[language]

        messages = self.build_messages(
            example,
            language=language,
            hint_mode=hint_mode,
        )

        base_prompt = self._apply_chat_template(
            tokenizer=tokenizer,
            messages=messages,
        )

        return self._append_hacker_prefix(
            base_prompt=base_prompt,
            hacking_starter=spec.hacking_starter,
        )

    def build_batch(
        self,
        examples: list[Mapping[str, Any] | Any],
        *,
        tokenizer: Any,
        language: str,
        hint_mode: str,
    ) -> list[str]:
        return [
            self.build_prompt(
                example,
                tokenizer=tokenizer,
                language=language,
                hint_mode=hint_mode,
            )
            for example in examples
        ]

    def preview_user_content(
        self,
        example: Mapping[str, Any] | Any,
        *,
        language: str,
        hint_mode: str,
    ) -> str:
        """
        Preview only the user message content.
        Useful for checking hint placement.
        """
        messages = self.build_messages(
            example,
            language=language,
            hint_mode=hint_mode,
        )
        return messages[1]["content"]

    # ------------------------------------------------------------------
    # Config loading and validation
    # ------------------------------------------------------------------

    def _load_config(self) -> PromptConfig:
        if not self.config_path.exists():
            raise FileNotFoundError(f"Prompt config file not found: {self.config_path}")

        with self.config_path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)

        if not isinstance(raw, dict):
            raise PromptConfigError("Top-level prompts YAML must be a mapping")

        languages = raw.get("languages")
        hints = raw.get("hints")
        translations_raw = raw.get("translations")

        if not isinstance(languages, dict) or not languages:
            raise PromptConfigError("Expected non-empty 'languages' mapping")
        if not isinstance(hints, dict) or not hints:
            raise PromptConfigError("Expected non-empty 'hints' mapping")
        if not isinstance(translations_raw, dict) or not translations_raw:
            raise PromptConfigError("Expected non-empty 'translations' mapping")

        translations: dict[str, LanguagePromptSpec] = {}
        for language, block in translations_raw.items():
            if not isinstance(block, dict):
                raise PromptConfigError(f"translations.{language} must be a mapping")

            missing = [field for field in self.REQUIRED_TRANSLATION_FIELDS if field not in block]
            if missing:
                raise PromptConfigError(
                    f"translations.{language} is missing required fields: {missing}"
                )

            translations[language] = LanguagePromptSpec(
                system=str(block["system"]).strip(),
                instruction=str(block["instruction"]).strip(),
                question_label=str(block["question_label"]).strip(),
                think_in=str(block["think_in"]).strip(),
                hacking_starter=str(block["hacking_starter"]).strip(),
            )

        self._validate_alignment(
            languages=languages,
            hints=hints,
            translations=translations,
        )

        return PromptConfig(
            languages={str(k): str(v) for k, v in languages.items()},
            hints={
                str(mode): {str(lang): str(text) for lang, text in lang_map.items()}
                for mode, lang_map in hints.items()
            },
            translations=translations,
        )

    def _validate_alignment(
        self,
        *,
        languages: dict[str, str],
        hints: dict[str, dict[str, str]],
        translations: dict[str, LanguagePromptSpec],
    ) -> None:
        language_keys = set(languages.keys())
        translation_keys = set(translations.keys())

        if language_keys != translation_keys:
            missing = sorted(language_keys - translation_keys)
            extra = sorted(translation_keys - language_keys)
            raise PromptConfigError(
                f"Mismatch between languages and translations. Missing: {missing}; Extra: {extra}"
            )

        for hint_mode, lang_map in hints.items():
            if not isinstance(lang_map, dict):
                raise PromptConfigError(f"hints.{hint_mode} must be a mapping")

            hint_keys = set(lang_map.keys())
            if hint_keys != language_keys:
                missing = sorted(language_keys - hint_keys)
                extra = sorted(hint_keys - language_keys)
                raise PromptConfigError(
                    f"Mismatch in hints.{hint_mode}. Missing: {missing}; Extra: {extra}"
                )

    # ------------------------------------------------------------------
    # Prompt rendering
    # ------------------------------------------------------------------

    def _apply_chat_template(
        self,
        *,
        tokenizer: Any,
        messages: list[dict[str, str]],
    ) -> str:
        """
        Render system+user messages using the tokenizer's chat template.

        We stop at the assistant-generation boundary, then insert the hacker
        prefix ourselves so it starts inside <think>.
        """
        apply_chat_template = getattr(tokenizer, "apply_chat_template", None)

        if callable(apply_chat_template):
            try:
                return apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=True,
                )
            except TypeError:
                pass

            try:
                return apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except TypeError:
                return apply_chat_template(
                    messages,
                    tokenize=False,
                )

        return self._fallback_prompt_text(messages)

    @staticmethod
    def _fallback_prompt_text(messages: list[dict[str, str]]) -> str:
        """
        Plain-text fallback for tokenizers without a chat template.
        """
        parts: list[str] = []

        for message in messages:
            role = message["role"].strip().upper()
            content = message["content"].rstrip()
            parts.append(f"{role}:\n{content}")

        parts.append("ASSISTANT:\n")
        return "\n\n".join(parts)

    def _append_hacker_prefix(self, *, base_prompt: str, hacking_starter: str) -> str:
        """
        Append the hacker prefix so the body begins inside <think>.

        Final intent:
          <assistant-start>
          <think>
          By request, I will start thinking in English.
          ... model continues here ...
        """
        body = self._extract_hacking_body(hacking_starter)
        prompt = base_prompt.rstrip()

        if self._ends_with_open_think(prompt):
            return f"{prompt}\n{body}\n"

        return f"{prompt}\n<think>\n{body}\n"

    @staticmethod
    def _sanitize_hacking_starter(text: str) -> str:
        """
        Normalize the hacking starter without requiring it to end in <think>.

        Any existing <think> or </think> tags are stripped from the YAML value.
        The prompt builder itself decides where to open the final <think> block.
        """
        text = text.strip()
        if not text:
            raise PromptConfigError("hacking_starter cannot be empty")

        text = re.sub(r"</?think>\s*", "", text, flags=re.IGNORECASE).strip()

        if not text:
            raise PromptConfigError("hacking_starter body cannot be empty after stripping think tags")

        return text

    def _extract_hacking_body(self, text: str) -> str:
        """
        Return the body text that should appear inside <think>.
        """
        return self._sanitize_hacking_starter(text)

    @staticmethod
    def _ends_with_open_think(text: str) -> bool:
        return bool(re.search(r"<think>\s*$", text, flags=re.IGNORECASE))

    # ------------------------------------------------------------------
    # Example extraction
    # ------------------------------------------------------------------

    def _get_formatted_options(self, example: Mapping[str, Any] | Any) -> str:
        if isinstance(example, Mapping):
            formatted = example.get("formatted_options")
            if formatted:
                return str(formatted).strip()

            if all(key in example for key in ("option_a", "option_b", "option_c", "option_d")):
                return "\n".join(
                    [
                        f"A) {str(example['option_a']).strip()}",
                        f"B) {str(example['option_b']).strip()}",
                        f"C) {str(example['option_c']).strip()}",
                        f"D) {str(example['option_d']).strip()}",
                    ]
                )

            options = example.get("options")
            if isinstance(options, Mapping):
                required = ("A", "B", "C", "D")
                missing = [label for label in required if label not in options]
                if missing:
                    raise PromptConfigError(f"Missing options in example['options']: {missing}")

                return "\n".join(
                    [
                        f"A) {str(options['A']).strip()}",
                        f"B) {str(options['B']).strip()}",
                        f"C) {str(options['C']).strip()}",
                        f"D) {str(options['D']).strip()}",
                    ]
                )

        if hasattr(example, "formatted_options") and callable(example.formatted_options):
            return str(example.formatted_options()).strip()

        if all(hasattr(example, attr) for attr in ("option_a", "option_b", "option_c", "option_d")):
            return "\n".join(
                [
                    f"A) {str(getattr(example, 'option_a')).strip()}",
                    f"B) {str(getattr(example, 'option_b')).strip()}",
                    f"C) {str(getattr(example, 'option_c')).strip()}",
                    f"D) {str(getattr(example, 'option_d')).strip()}",
                ]
            )

        raise PromptConfigError(
            "Could not build options from example. Expected formatted_options, "
            "options, or option_a/option_b/option_c/option_d fields."
        )

    @staticmethod
    def _extract_field(example: Mapping[str, Any] | Any, field_name: str) -> str:
        if isinstance(example, Mapping):
            if field_name not in example:
                raise PromptConfigError(f"Missing field '{field_name}' in example mapping")
            return str(example[field_name]).strip()

        if hasattr(example, field_name):
            return str(getattr(example, field_name)).strip()

        raise PromptConfigError(f"Missing field '{field_name}' in example object")

    # ------------------------------------------------------------------
    # Language and hint resolution
    # ------------------------------------------------------------------

    def _resolve_language(self, language: str) -> str:
        language = self._normalize(language)

        if language in self.config.languages:
            return language

        if self.strict:
            raise PromptConfigError(
                f"Unknown language '{language}'. Available: {sorted(self.config.languages.keys())}"
            )

        return self.default_language

    def _resolve_hint_mode(self, hint_mode: str) -> str:
        hint_mode = self._normalize(hint_mode)

        if hint_mode in self.config.hints:
            return hint_mode

        if self.strict:
            raise PromptConfigError(
                f"Unknown hint mode '{hint_mode}'. Available: {sorted(self.config.hints.keys())}"
            )

        return "none"

    def _get_hint(self, *, language: str, hint_mode: str) -> str:
        try:
            return self.config.hints[hint_mode][language].strip()
        except KeyError as exc:
            raise PromptConfigError(
                f"Missing hint for language='{language}', hint_mode='{hint_mode}'"
            ) from exc

    @staticmethod
    def _normalize(value: str) -> str:
        value = value.strip().lower()
        if not value:
            raise ValueError("Expected a non-empty string")
        return value


if __name__ == "__main__":
    import argparse

    BASE_DIR = Path(__file__).resolve().parent.parent
    if str(BASE_DIR) not in sys.path:
        sys.path.insert(0, str(BASE_DIR))

    from data_process.data_loader import GPQADataLoader

    parser = argparse.ArgumentParser(description="Test prompt_builder.py")
    parser.add_argument("--language", default="english")
    parser.add_argument("--hint-mode", default="complex", choices=["none", "simple", "complex"])
    parser.add_argument("--model-key", default="qwen3_8b")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--max-examples", type=int, default=1)
    args = parser.parse_args()

    prompts_config_path = BASE_DIR / "config" / "prompts_config.yaml"
    models_config_path = BASE_DIR / "config" / "models_config.yaml"
    data_dir = BASE_DIR / "gpqa_dataset" / "json"

    with models_config_path.open("r", encoding="utf-8") as handle:
        models_cfg = yaml.safe_load(handle)

    model_cfg = models_cfg["models"][args.model_key]
    family = str(model_cfg.get("family", "")).strip().lower()
    trust_remote_code = family == "qwen"

    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg["path"],
        trust_remote_code=trust_remote_code,
    )

    loader = GPQADataLoader(data_dir, strict=True)
    dataset = loader.load_dataset(args.language, max_examples=args.max_examples)

    if len(dataset) == 0:
        raise ValueError("No examples available for testing")

    if args.sample_index < 0 or args.sample_index >= len(dataset):
        raise IndexError(
            f"sample-index {args.sample_index} is out of range for dataset of size {len(dataset)}"
        )

    builder = PromptBuilder(prompts_config_path, strict=True)
    sample = dataset[args.sample_index]

    print("=" * 100)
    print("DATASET SUMMARY")
    print("=" * 100)
    print(dataset.summary())
    print()

    print("=" * 100)
    print("USER CONTENT PREVIEW")
    print("=" * 100)
    print(builder.preview_user_content(sample, language=args.language, hint_mode=args.hint_mode))
    print()

    prompt = builder.build_prompt(
        sample,
        tokenizer=tokenizer,
        language=args.language,
        hint_mode=args.hint_mode,
    )

    print("=" * 100)
    print("FULL RENDERED PROMPT")
    print("=" * 100)
    print(prompt)