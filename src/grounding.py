"""Checks that typed function arguments can be grounded in the user prompt."""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal

from src import utils
from src.functions_manager import FunctionsDefinition


BOOLEAN_PATTERN = re.compile(r"\b(true|false)\b", re.IGNORECASE)
RESPONSE_STRUCTURE_PATTERN = re.compile(
    r"(?:\{\s*[\"']?|[\"'])(?:prompt|parameters|name)[\"']?\s*:",
    re.IGNORECASE,
)
RESPONSE_FIELD_WORD_PATTERN = re.compile(
    r"\b(?:prompt|parameters|name)\b",
    re.IGNORECASE,
)
PLACEHOLDER_PATTERN = re.compile(r"\{[A-Za-z_][A-Za-z0-9_]*\}")
QUOTED_PATH_PATTERN = re.compile(
    r"([\"'])(?P<path>(?:[A-Za-z]:[\\/]|/).*?)\1"
)
WINDOWS_PATH_PATTERN = re.compile(r"(?<!\w)[A-Za-z]:\\[^\s\"']+")
POSIX_PATH_PATTERN = re.compile(r"(?<!\w)/[^\s\"']+")


def contains_response_structure(value: str, source_prompt: str = "") -> bool:
    """Return whether a string embeds structure absent from the user prompt.

    Object-key fragments are always rejected. Reserved response-field words
    and named placeholders are accepted only when the user supplied them.
    """
    if RESPONSE_STRUCTURE_PATTERN.search(value):
        return True

    source_words = {
        match.group(0).casefold()
        for match in RESPONSE_FIELD_WORD_PATTERN.finditer(source_prompt)
    }
    for match in RESPONSE_FIELD_WORD_PATTERN.finditer(value):
        if match.group(0).casefold() not in source_words:
            return True

    return any(
        match.group(0) not in source_prompt
        for match in PLACEHOLDER_PATTERN.finditer(value)
    )


def extract_path(prompt: str) -> str | None:
    """Extract the first literal Unix or Windows path from a prompt."""
    quoted = QUOTED_PATH_PATTERN.search(prompt)
    if quoted:
        return quoted.group("path")

    for pattern in (WINDOWS_PATH_PATTERN, POSIX_PATH_PATTERN):
        match = pattern.search(prompt)
        if match:
            return match.group(0)
    return None


def infer_string_parameters(prompt: str) -> dict[str, str]:
    """Extract unambiguous string arguments from common request phrasing.

    The returned keys are semantic parameter names. An empty mapping means
    that the prompt was not recognized and constrained generation should fall
    back to the language model.

    Args:
        prompt: User request to match against supported unambiguous forms.

    Returns:
        Extracted values keyed by semantic parameter name, or an empty mapping.
    """
    substitute = re.fullmatch(
        r"\s*substitute(?:\s+the\s+word)?\s+(['\"])(.*?)\1\s+with\s+"
        r"(['\"])(.*?)\3\s+in\s+(['\"])(.*?)\5\s*[.!?]?\s*",
        prompt,
        flags=re.IGNORECASE,
    )
    if substitute:
        searched, replacement, source = (
            substitute.group(2),
            substitute.group(4),
            substitute.group(6),
        )
        return {
            "source_string": source,
            "regex": re.escape(searched),
            "replacement": replacement,
        }

    replace = re.fullmatch(
        r"\s*replace\s+all\s+(numbers|vowels)\s+in\s+"
        r"(['\"])(.*?)\2\s+with\s+(.+?)\s*[.!?]?\s*",
        prompt,
        flags=re.IGNORECASE,
    )
    if replace:
        kind, source, replacement = (
            replace.group(1).lower(),
            replace.group(3),
            replace.group(4).strip(" \t'\""),
        )
        regex = "[0-9]+" if kind == "numbers" else "[aeiouAEIOU]"
        return {
            "source_string": source,
            "regex": regex,
            "replacement": replacement,
        }

    greet = re.fullmatch(
        r"\s*greet\s+(['\"]?)(.+?)\1\s*[.!?]?\s*",
        prompt,
        flags=re.IGNORECASE,
    )
    if greet:
        return {"name": greet.group(2)}

    reverse = re.fullmatch(
        r"\s*reverse(?:\s+the)?\s+string\s+(['\"])(.*?)\1\s*[.!?]?\s*",
        prompt,
        flags=re.IGNORECASE,
    )
    if reverse:
        return {"s": reverse.group(2)}

    path = extract_path(prompt)
    if path is not None:
        return {"path": path}

    return {}


@dataclass(frozen=True)
class PromptValues:
    """Typed literals explicitly available in one user prompt."""

    numbers: list[Decimal]
    booleans: list[bool]


def collect_prompt_values(prompt: str) -> PromptValues:
    """Extract typed literals once using the grounding rules."""
    booleans = [
        match.group(1).lower() == "true"
        for match in BOOLEAN_PATTERN.finditer(prompt)
    ]
    return PromptValues(
        numbers=utils.extract_numbers(prompt),
        booleans=booleans,
    )


def validate_prompt_capacity(
    functions_def: FunctionsDefinition,
    prompt: str,
    function_name: str,
) -> None:
    """Ensure a prompt contains enough typed literals for a selected function.

    Args:
        functions_def: Available function definitions.
        prompt: Original user request containing source values.
        function_name: Function selected during constrained generation.

    Raises:
        ValueError: If the prompt lacks required number or boolean literals.
    """
    parameters = functions_def.get_function_parameters_by_name(function_name)
    required_numbers = sum(
        parameter.type in {"number", "integer"}
        for parameter in parameters.values()
    )
    required_booleans = sum(
        parameter.type == "boolean" for parameter in parameters.values()
    )
    available = collect_prompt_values(prompt)

    shortages = []
    if len(available.numbers) < required_numbers:
        shortages.append(
            f"{required_numbers} number parameter(s) required, "
            f"{len(available.numbers)} found"
        )
    if len(available.booleans) < required_booleans:
        shortages.append(
            f"{required_booleans} boolean parameter(s) required, "
            f"{len(available.booleans)} found"
        )

    if shortages:
        details = "; ".join(shortages)
        raise ValueError(
            f"Cannot call function {function_name!r} from this prompt: "
            f"{details}"
        )
