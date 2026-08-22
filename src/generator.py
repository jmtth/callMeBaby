"""Constrained token generation with one canonical response buffer."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Protocol, cast

import numpy as np

from src.JSONStateMachine import JSONStateMachine
from src.functions_manager import FunctionsDefinition
from src.models import JSONState
from src.token_vocabulary import TokenModel, TokenVocabulary


class GeneratorModel(TokenModel, Protocol):
    """Public model operations required by the generator."""

    def get_logits_from_input_ids(self, input_ids: list[int]) -> list[float]:
        """Return logits for the token following ``input_ids``."""
        ...


class GenerationLimitError(ValueError):
    """Raised when a complete constrained response exceeds its token budget."""


@dataclass
class GenerationBuffer:
    """Own the only generated-token list and enforce its exact budget."""

    prompt_ids: list[int]
    max_response_tokens: int
    response_ids: list[int] = field(default_factory=list)

    @property
    def context_ids(self) -> list[int]:
        """Return prompt and generated IDs as the model context."""
        return [*self.prompt_ids, *self.response_ids]

    @property
    def remaining(self) -> int:
        return self.max_response_tokens - len(self.response_ids)

    def append(self, token_ids: int | Sequence[int]) -> None:
        """Append generated IDs if the complete addition fits the budget."""
        ids = [token_ids] if isinstance(token_ids, int) else list(token_ids)
        if len(ids) > self.remaining:
            raise GenerationLimitError(
                "Constrained response exceeds max_res_tokens="
                f"{self.max_response_tokens}"
            )
        self.response_ids.extend(ids)


def normalize_encoded_ids(encoded: object) -> list[int]:
    """Normalize the first encoded batch from tensors, arrays, or lists."""
    first_batch = cast(Sequence[object], encoded)[0]
    to_list = getattr(first_batch, "tolist", None)
    if callable(to_list):
        first_batch = to_list()
    return [int(token_id) for token_id in cast(Sequence[int], first_batch)]


def select_next_token(
    model: GeneratorModel,
    current_ids: list[int],
    allowed_ids: set[int],
) -> int:
    """Select the allowed token ID with the highest model logit."""
    if not allowed_ids:
        raise ValueError("No allowed tokens available for selection")

    logits = np.asarray(model.get_logits_from_input_ids(current_ids))
    indices = np.fromiter(allowed_ids, dtype=int)
    if np.any(indices < 0) or np.any(indices >= logits.size):
        raise ValueError("Allowed token ID is outside the model vocabulary")
    best_index = int(np.argmax(logits[indices]))
    return int(indices[best_index])


def generate_constrained_response(
    model: GeneratorModel,
    token_to_id: dict[str, int],
    functions_def: FunctionsDefinition,
    prompt: str,
    input_prompt: str,
    max_res_tokens: int,
    *,
    vocabulary: TokenVocabulary | None = None,
    token_selector: Callable[
        [GeneratorModel, list[int], set[int]], int
    ] = select_next_token,
    fsm_factory: Callable[..., JSONStateMachine] = JSONStateMachine,
) -> str:
    """Generate one schema-constrained JSON response."""
    if max_res_tokens < 1:
        raise ValueError("max_res_tokens must be greater than zero")

    prompt_ids = normalize_encoded_ids(model.encode(prompt))
    buffer = GenerationBuffer(prompt_ids, max_res_tokens)
    vocabulary = vocabulary or TokenVocabulary(model, token_to_id)
    fsm = fsm_factory(
        model,
        functions_def,
        token_to_id,
        input_prompt,
        vocabulary,
    )
    generation_steps = 0
    max_generation_steps = max_res_tokens * 2 + 32

    while fsm.state != JSONState.STOP:
        generation_steps += 1
        if generation_steps > max_generation_steps:
            raise GenerationLimitError(
                "Constrained generation did not converge within "
                f"{max_generation_steps} state-machine steps"
            )
        if fsm.is_in_fixed_sequence():
            target_ids = fsm.get_target_tokens_for_current_state()
            if not target_ids:
                fsm.complete_empty_fixed_sequence()
                continue
            buffer.append(target_ids)
            for token_id in target_ids:
                fsm.update(token_id)
            continue

        allowed_ids = fsm.get_allowed_tokens()
        if not allowed_ids:
            raise ValueError(
                f"No token allowed while generating state {fsm.state.name}"
            )

        if len(allowed_ids) == 1:
            token_id = next(iter(allowed_ids))
        else:
            token_id = token_selector(
                model,
                buffer.context_ids,
                allowed_ids,
            )

        if fsm.update(token_id):
            buffer.append(token_id)

    return model.decode(buffer.response_ids)
