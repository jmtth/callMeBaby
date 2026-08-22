"""Checks that typed function arguments can be grounded in the user prompt."""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal

from src import utils
from src.functions_manager import FunctionsDefinition


BOOLEAN_PATTERN = re.compile(r"\b(true|false)\b", re.IGNORECASE)


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
    """Reject a selected function lacking enough typed source literals."""
    parameters = functions_def.get_function_parameters_by_name(function_name)
    required_numbers = sum(
        parameter.type == "number" for parameter in parameters.values()
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
