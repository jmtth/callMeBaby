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


@dataclass(frozen=True)
class TokenCandidate:
    """One allowed token considered during a model decision."""

    token_id: int
    text: str
    logit: float


@dataclass(frozen=True)
class GenerationStep:
    """Read-only snapshot emitted after one generation decision."""

    index: int
    state: str
    kind: str
    selected_id: int | None
    selected_text: str
    generated_text: str
    response_tokens: int
    allowed_count: int
    candidates: tuple[TokenCandidate, ...] = ()


GenerationObserver = Callable[[GenerationStep], None]


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
        """Return the number of response tokens still available."""
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


def rank_allowed_tokens(
    model: GeneratorModel,
    current_ids: list[int],
    allowed_ids: set[int],
    vocabulary: TokenVocabulary,
    limit: int = 20,
) -> tuple[int, tuple[TokenCandidate, ...]]:
    """Rank allowed tokens by model logit and select the best candidate.

    Args:
        model: Model providing next-token logits.
        current_ids: Token IDs forming the current model context.
        allowed_ids: Token IDs permitted by the state machine.
        vocabulary: Vocabulary used to decode candidate IDs.
        limit: Maximum number of ranked candidates to retain.

    Returns:
        The selected token ID and a descending snapshot of candidates.

    Raises:
        ValueError: If no tokens are allowed or an ID is outside the logits.
    """
    if not allowed_ids:
        raise ValueError("No allowed tokens available for selection")
    logits = np.asarray(model.get_logits_from_input_ids(current_ids))
    indices = np.fromiter(allowed_ids, dtype=int)
    if np.any(indices < 0) or np.any(indices >= logits.size):
        raise ValueError("Allowed token ID is outside the model vocabulary")
    ranked_ids = sorted(
        (int(token_id) for token_id in indices),
        key=lambda token_id: float(logits[token_id]),
        reverse=True,
    )[:limit]
    candidates = tuple(
        TokenCandidate(
            token_id=token_id,
            text=vocabulary.text(token_id),
            logit=float(logits[token_id]),
        )
        for token_id in ranked_ids
    )
    return ranked_ids[0], candidates


def generate_constrained_response(
    model: GeneratorModel,
    token_to_id: dict[str, int],
    functions_def: FunctionsDefinition,
    prompt: str,
    input_prompt: str,
    max_res_tokens: int,
    *,
    vocabulary: TokenVocabulary | None = None,
    observer: GenerationObserver | None = None,
    token_selector: Callable[
        [GeneratorModel, list[int], set[int]], int
    ] = select_next_token,
    fsm_factory: Callable[..., JSONStateMachine] = JSONStateMachine,
) -> str:
    """Generate one JSON response under state-machine constraints.

    The function maintains a single response buffer, emits fixed structural
    sequences atomically, and delegates ambiguous choices to model logits.

    Args:
        model: Model used for encoding, decoding, and token logits.
        token_to_id: Mapping from tokenizer spellings to token IDs.
        functions_def: Function schemas that constrain the response.
        prompt: Complete instruction prompt passed to the model.
        input_prompt: Original user request used by grounding rules.
        max_res_tokens: Maximum number of generated response tokens.
        vocabulary: Optional reusable token-vocabulary cache.
        observer: Optional callback receiving generation snapshots.
        token_selector: Strategy used for ambiguous token choices.
        fsm_factory: Factory used to construct the JSON state machine.

    Returns:
        The decoded schema-constrained JSON response.

    Raises:
        GenerationLimitError: If the token budget or step guard is exceeded.
        ValueError: If generation reaches a state with no valid token.
    """
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
            state_name = fsm.state.name
            target_ids = fsm.get_target_tokens_for_current_state()
            if not target_ids:
                fsm.complete_empty_fixed_sequence()
                continue
            buffer.append(target_ids)
            for token_id in target_ids:
                fsm.update(token_id)
            if observer is not None:
                observer(GenerationStep(
                    index=generation_steps,
                    state=state_name,
                    kind="fixed",
                    selected_id=None,
                    selected_text=model.decode(target_ids),
                    generated_text=model.decode(buffer.response_ids),
                    response_tokens=len(buffer.response_ids),
                    allowed_count=len(target_ids),
                ))
            continue

        allowed_ids = fsm.get_allowed_tokens()
        if not allowed_ids:
            raise ValueError(
                f"No token allowed while generating state {fsm.state.name}"
            )

        state_name = fsm.state.name
        candidates: tuple[TokenCandidate, ...] = ()
        if len(allowed_ids) == 1:
            token_id = next(iter(allowed_ids))
            kind = "deterministic"
        elif observer is not None and token_selector is select_next_token:
            token_id, candidates = rank_allowed_tokens(
                model,
                buffer.context_ids,
                allowed_ids,
                vocabulary,
            )
            kind = "model"
        else:
            token_id = token_selector(
                model,
                buffer.context_ids,
                allowed_ids,
            )
            kind = "model"

        if fsm.update(token_id):
            buffer.append(token_id)
        if observer is not None:
            observer(GenerationStep(
                index=generation_steps,
                state=state_name,
                kind=kind,
                selected_id=token_id,
                selected_text=vocabulary.text(token_id),
                generated_text=model.decode(buffer.response_ids),
                response_tokens=len(buffer.response_ids),
                allowed_count=len(allowed_ids),
                candidates=candidates,
            ))

    return model.decode(buffer.response_ids)
