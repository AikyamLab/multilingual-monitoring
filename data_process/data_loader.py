from __future__ import annotations

import json
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Any, Iterable, Iterator

from torch.utils.data import Dataset


class DatasetError(Exception):
    """Base exception for dataset-loading issues."""


class DatasetFormatError(DatasetError):
    """Raised when a dataset file is malformed."""


@dataclass(frozen=True, slots=True)
class GPQAExample:
    """
    Minimal normalized example for your GPQA pipeline.

    Only question and answer options are kept.
    Hint text is intentionally excluded because it will come from YAML.
    """
    language: str
    question: str
    option_a: str   # Correct Answer
    option_b: str   # Incorrect Answer 1
    option_c: str   # Incorrect Answer 2
    option_d: str   # Incorrect Answer 3

    @property
    def options(self) -> tuple[tuple[str, str], ...]:
        return (
            ("A", self.option_a),
            ("B", self.option_b),
            ("C", self.option_c),
            ("D", self.option_d),
        )

    def formatted_options(self) -> str:
        return "\n".join(f"{label}) {text}" for label, text in self.options)

    def to_prompt_dict(self) -> dict[str, Any]:
        """
        Returns a clean dict that other scripts can consume directly.
        """
        return {
            "language": self.language,
            "question": self.question,
            "options": {
                "A": self.option_a,
                "B": self.option_b,
                "C": self.option_c,
                "D": self.option_d,
            },
        }


class GPQADataset(Dataset):
    """
    PyTorch-style text dataset.

    Each item is returned as a dict so downstream code can work naturally with:
    - question
    - options
    - formatted_options
    - language
    """
    def __init__(self, examples: list[GPQAExample]) -> None:
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        example = self.examples[idx]
        return {
            "language": example.language,
            "question": example.question,
            "option_a": example.option_a,
            "option_b": example.option_b,
            "option_c": example.option_c,
            "option_d": example.option_d,
            "options": {
                "A": example.option_a,
                "B": example.option_b,
                "C": example.option_c,
                "D": example.option_d,
            },
            "formatted_options": example.formatted_options(),
        }

    @property
    def num_examples(self) -> int:
        return len(self.examples)

    @property
    def num_fields(self) -> int:
        """
        Number of core fields per prepared example.
        """
        return 6  # language, question, option_a, option_b, option_c, option_d

    def summary(self) -> dict[str, Any]:
        return {
            "num_examples": self.num_examples,
            "num_fields": self.num_fields,
            "fields": [
                "language",
                "question",
                "option_a",
                "option_b",
                "option_c",
                "option_d",
            ],
        }

    def head(self, n: int = 3) -> list[dict[str, Any]]:
        if n < 0:
            raise ValueError("n must be >= 0")
        return [self[i] for i in range(min(n, len(self)))]

    def preview(self, n: int = 3) -> None:
        """
        Print a readable preview of prepared samples.
        """
        samples = self.head(n)
        print("=" * 100)
        print("DATASET PREVIEW")
        print("=" * 100)
        print(f"Number of examples: {self.num_examples}")
        print(f"Number of fields  : {self.num_fields}")
        print()

        for i, sample in enumerate(samples, start=1):
            print("-" * 100)
            print(f"Sample {i}")
            print("-" * 100)
            print(f"Language : {sample['language']}")
            print(f"Question : {sample['question']}")
            print(sample["formatted_options"])
            print()


class GPQADataLoader:
    """
    Loads multilingual GPQA data from JSON or JSONL files.

    Supported file names:
      - english.jsonl
      - english.json

    Supported content:
      - JSON array
      - JSONL (one JSON object per line)

    It returns a prepared GPQADataset ready for prompt building.
    """

    REQUIRED_FIELDS = (
        "Question",
        "Correct Answer",
        "Incorrect Answer 1",
        "Incorrect Answer 2",
        "Incorrect Answer 3",
    )

    SUPPORTED_EXTENSIONS = (".jsonl", ".json")

    def __init__(self, data_dir: str | Path, *, strict: bool = True, encoding: str = "utf-8") -> None:
        self.data_dir = Path(data_dir).expanduser().resolve()
        self.strict = strict
        self.encoding = encoding

        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory does not exist: {self.data_dir}")
        if not self.data_dir.is_dir():
            raise NotADirectoryError(f"Expected a directory, got: {self.data_dir}")

    def available_languages(self) -> tuple[str, ...]:
        languages = {
            path.stem
            for ext in self.SUPPORTED_EXTENSIONS
            for path in self.data_dir.glob(f"*{ext}")
        }
        return tuple(sorted(languages))

    def load_dataset(self, language: str, *, max_examples: int | None = None) -> GPQADataset:
        examples = list(self.iter_examples(language, max_examples=max_examples))
        return GPQADataset(examples)

    def iter_examples(self, language: str, *, max_examples: int | None = None) -> Iterator[GPQAExample]:
        language = self._normalize_language(language)
        file_path = self._resolve_file(language)

        rows = self._iter_rows(file_path)
        if max_examples is not None:
            if max_examples < 0:
                raise ValueError("max_examples must be >= 0")
            rows = islice(rows, max_examples)

        for row_index, row in enumerate(rows, start=1):
            try:
                yield self._build_example(language=language, row=row)
            except DatasetFormatError as exc:
                if self.strict:
                    raise DatasetFormatError(f"{file_path.name} | row {row_index}: {exc}") from exc

    @staticmethod
    def batch_iter(dataset: GPQADataset, batch_size: int) -> Iterator[list[dict[str, Any]]]:
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")

        for start in range(0, len(dataset), batch_size):
            yield [dataset[i] for i in range(start, min(start + batch_size, len(dataset)))]

    def _resolve_file(self, language: str) -> Path:
        candidates = [self.data_dir / f"{language}{ext}" for ext in self.SUPPORTED_EXTENSIONS]
        for candidate in candidates:
            if candidate.exists():
                return candidate

        raise FileNotFoundError(
            f"No dataset file found for language='{language}' in {self.data_dir}"
        )

    def _iter_rows(self, file_path: Path) -> Iterator[dict[str, Any]]:
        file_format = self._detect_format(file_path)

        if file_format == "json_array":
            with file_path.open("r", encoding=self.encoding) as handle:
                payload = json.load(handle)

            if not isinstance(payload, list):
                raise DatasetFormatError(
                    f"Expected a JSON array in {file_path.name}, got {type(payload).__name__}"
                )

            for row in payload:
                if not isinstance(row, dict):
                    raise DatasetFormatError(
                        f"Each record must be a JSON object, got {type(row).__name__}"
                    )
                yield row
            return

        with file_path.open("r", encoding=self.encoding) as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue

                try:
                    row = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    raise DatasetFormatError(
                        f"Invalid JSON on line {line_number} of {file_path.name}"
                    ) from exc

                if not isinstance(row, dict):
                    raise DatasetFormatError(
                        f"Each JSONL line must be a JSON object, got {type(row).__name__}"
                    )

                yield row

    def _detect_format(self, file_path: Path) -> str:
        with file_path.open("r", encoding=self.encoding) as handle:
            while char := handle.read(1):
                if char.isspace():
                    continue
                return "json_array" if char == "[" else "jsonl"

        raise DatasetFormatError(f"Empty dataset file: {file_path.name}")

    def _build_example(self, *, language: str, row: dict[str, Any]) -> GPQAExample:
        missing = [field for field in self.REQUIRED_FIELDS if field not in row]
        if missing:
            raise DatasetFormatError(f"Missing required fields: {missing}")

        question = self._clean_text(row["Question"])
        option_a = self._clean_text(row["Correct Answer"])
        option_b = self._clean_text(row["Incorrect Answer 1"])
        option_c = self._clean_text(row["Incorrect Answer 2"])
        option_d = self._clean_text(row["Incorrect Answer 3"])

        if not all((question, option_a, option_b, option_c, option_d)):
            raise DatasetFormatError("Question and all answer options must be non-empty")

        return GPQAExample(
            language=language,
            question=question,
            option_a=option_a,
            option_b=option_b,
            option_c=option_c,
            option_d=option_d,
        )

    @staticmethod
    def _normalize_language(language: str) -> str:
        language = language.strip().lower()
        if not language:
            raise ValueError("language must be a non-empty string")
        return language

    @staticmethod
    def _clean_text(value: Any) -> str:
        if value is None:
            return ""
        return str(value).strip()


if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent.parent
    DATA_DIR = BASE_DIR / "gpqa_dataset" / "json"

    loader = GPQADataLoader(DATA_DIR, strict=True)
    print("Available languages:", loader.available_languages())

    dataset = loader.load_dataset("english", max_examples=5)
    print(dataset.summary())
    dataset.preview(n=2)