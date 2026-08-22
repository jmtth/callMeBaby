"""Token vocabulary indexes used by constrained decoding."""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any, Protocol

from src import utils


class TokenDecoder(Protocol):
    """Small part of the model API needed to inspect tokens."""

    def decode(self, ids: list[int]) -> str:
        """Decode token IDs into text."""
        ...


class TokenModel(TokenDecoder, Protocol):
    """Tokenizer operations required by the state machine."""

    def encode(self, text: str) -> Any:
        """Encode text into a batch containing token IDs."""


class TokenVocabulary:
    """Cache the decoded text and useful classifications of model tokens.

    Token IDs are the canonical representation inside the generation pipeline.
    Text is decoded here, once per ID, only when a constraint needs it.
    """

    def __init__(
        self,
        model: TokenDecoder,
        token_to_id: dict[str, int],
    ) -> None:
        self._model = model
        self._token_to_id = token_to_id
        self.all_ids = set(token_to_id.values())
        self._decoded_text: dict[int, str] = {}
        self._exact_text_ids: dict[str, set[int]] = {}
        self._number_ids: set[int] | None = None
        self._json_string_ids: set[int] | None = None

    def text(self, token_id: int) -> str:
        """Return and cache the text actually produced by one token ID."""
        if token_id not in self._decoded_text:
            self._decoded_text[token_id] = self._model.decode([token_id])
        return self._decoded_text[token_id]

    def exact_ids(self, expected_text: str) -> set[int]:
        """Return IDs whose decoded representation exactly matches text."""
        if expected_text not in self._exact_text_ids:
            self._exact_text_ids[expected_text] = {
                token_id
                for token_id in self._candidate_ids(expected_text)
                if self.text(token_id) == expected_text
            }
        return set(self._exact_text_ids[expected_text])

    def ids_continuing(self, target: str, generated: str) -> set[int]:
        """Return IDs that can extend ``generated`` towards ``target``."""
        if not target.startswith(generated):
            return set()
        remaining = target[len(generated):]
        if not remaining:
            return set()
        return {
            token_id
            for token_id in self._candidate_ids(remaining)
            if (token_text := self.text(token_id))
            and remaining.startswith(token_text)
        }

    def number_fragment_ids(self) -> set[int]:
        """Return IDs whose decoded text can participate in a number."""
        if self._number_ids is None:
            self._number_ids = {
                token_id
                for token_id in self.all_ids
                if (token_text := self.text(token_id))
                and " " not in token_text
                and utils.is_valid_number_fragment(token_text)
            }
        return set(self._number_ids)

    def number_terminator_ids(self) -> set[int]:
        """Return IDs that terminate a number without joining its value."""
        terminators: set[int] = set()
        for token_text in (" ", "  ", ",", ", ", " ,", "}"):
            terminators.update(self.exact_ids(token_text))
        return terminators

    def json_string_content_ids(self) -> set[int]:
        """Return IDs safe to append unescaped inside a JSON string."""
        if self._json_string_ids is None:
            self._json_string_ids = {
                token_id
                for token_id in self.all_ids
                if self._is_safe_json_string_fragment(self.text(token_id))
            }
        return set(self._json_string_ids)

    @staticmethod
    def _is_safe_json_string_fragment(token_text: str) -> bool:
        if not token_text or '"' in token_text or "\\" in token_text:
            return False
        escaped = json.dumps(token_text, ensure_ascii=False)[1:-1]
        return escaped == token_text

    def _candidate_ids(self, text_prefix: str) -> Iterable[int]:
        """Narrow common tokenizer spellings before verifying with decode()."""
        candidates: set[int] = set()
        for raw_token, token_id in self._token_to_id.items():
            normalized = raw_token.replace("Ġ", " ").replace("▁", " ")
            if (
                text_prefix.startswith(normalized)
                or normalized.startswith(text_prefix)
            ):
                candidates.add(token_id)

        # Synthetic vocabularies used by callers may not expose tokenizer
        # markers. Falling back to all IDs preserves correctness; decode() is
        # still the final authority.
        return candidates or self.all_ids
