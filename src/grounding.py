"""Extract and validate function arguments grounded in the user prompt."""

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
QUOTED_STRING_PATTERN = re.compile(r"([\"'])(?P<value>.+?)\1")
TOKEN_PATTERN = re.compile(r"\S+")
EDGE_PUNCTUATION = ",;:!?()[]"
WINDOWS_PATH_TOKEN_PATTERN = re.compile(r"^[A-Za-z]:[\\/]")


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


def collect_prompt_string_candidates(
    prompt: str,
    parameter_name: str | None = None,
    max_length: int = 80,
) -> tuple[str, ...]:
    """Return exact prompt spans suitable for an extractive string value.

    Parameter labels provide a strong generic signal when present. A unique
    path-shaped literal is also unambiguous. Unknown forms return no candidate
    so transformations such as ``asterisks`` to ``*`` remain generative.

    Args:
        prompt: Original user request containing source values.
        parameter_name: Optional schema parameter currently being generated.
        max_length: Longest candidate accepted by the string FSM.

    Returns:
        Deduplicated candidates in their order of discovery.
    """
    if not prompt or max_length < 1:
        return ()

    if parameter_name:
        label = re.escape(parameter_name).replace("_", r"[ _]")
        bounded_label = rf"(?<![\w{{]){label}(?![\w}}])"
        colon_match = re.search(
            rf"{bounded_label}\s*:\s*(?P<value>.+?)\s*$",
            prompt,
            re.IGNORECASE,
        )
        if colon_match:
            value = colon_match.group("value")
            if len(value) <= max_length:
                return (value,)

        label_match = re.search(bounded_label, prompt, re.IGNORECASE)
        if label_match:
            suffix = prompt[label_match.end():]
            quoted = QUOTED_STRING_PATTERN.search(suffix)
            if quoted and not suffix[:quoted.start()].strip():
                value = quoted.group("value")
                if len(value) <= max_length:
                    return (value,)

            prefix_tokens = TOKEN_PATTERN.findall(prompt[:label_match.start()])
            if prefix_tokens:
                value = prefix_tokens[-1].strip(EDGE_PUNCTUATION + "\"'")
                if value and len(value) <= max_length:
                    return (value,)

    path_candidates: list[str] = []
    for raw_token in TOKEN_PATTERN.findall(prompt):
        value = raw_token.strip(EDGE_PUNCTUATION + "\"'")
        if (
            value.startswith("/")
            or WINDOWS_PATH_TOKEN_PATTERN.match(value)
        ) and len(value) <= max_length:
            path_candidates.append(value)

    return tuple(dict.fromkeys(path_candidates))


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
