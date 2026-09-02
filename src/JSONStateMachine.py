import json
from typing import cast

from src.models import JSONState
from src import utils
from src.functions_manager import FunctionsDefinition, Parameter
from src.grounding import infer_string_parameters, validate_prompt_capacity
from src.token_vocabulary import TokenModel, TokenVocabulary


MAX_STRING_LENGTH = 80


class JSONStateMachine:
    """Constrain token generation to a valid function-call JSON object."""

    def __init__(self,
                 model: TokenModel,
                 functions_def: FunctionsDefinition,
                 token_to_id: dict[str, int],
                 prompt: str = "",
                 vocabulary: TokenVocabulary | None = None):
        """Initialize the state machine and pre-encode fixed JSON fragments.

        Args:
            model: Model used to encode fixed fragments and decode tokens.
            functions_def: Available function definitions.
            token_to_id: Mapping from tokenizer spellings to token IDs.
            prompt: Original user request used for value grounding.
            vocabulary: Optional reusable token-vocabulary cache.
        """
        self.model = model
        self.state = JSONState.START
        self.current_text = ""
        self.current_function_name = ""

        # Keep every function from functions_definition.json available.
        # LLM logits choose the function; the FSM only constrains the choice
        # to names that exist in the input schema.
        self.functions_names = functions_def.list_functions_name()
        self.functions = functions_def
        self.prompt = prompt
        self.grounded_string_parameters = infer_string_parameters(prompt)
        self.token_to_id = token_to_id
        self.vocabulary = vocabulary or TokenVocabulary(model, token_to_id)

        # Normalize encodings (support lists, numpy arrays, tensors)
        def _norm_encode(s: str) -> list[int]:
            """Normalize one encoded string to a list of token IDs."""
            enc0 = model.encode(s)[0]
            if hasattr(enc0, "tolist"):
                return [int(token_id) for token_id in cast(list[int],
                                                           enc0.tolist())]
            return [int(token_id) for token_id in enc0]

        # Escape user prompt for insertion inside a JSON string value.
        escaped_prompt = json.dumps(prompt, ensure_ascii=False)[1:-1]

        # Targets encodes
        self.targets = {
            JSONState.START: _norm_encode("{"),
            JSONState.PROMPT_KEY: _norm_encode('"prompt": "'),
            JSONState.NAME_KEY: _norm_encode('", "name": "'),
            JSONState.PARAMS_KEY: _norm_encode('", "parameters": {"'),
            JSONState.EMPTY_PARAMS: _norm_encode('", "parameters": {}}'),
            JSONState.PROMPT_VAL: _norm_encode(escaped_prompt),
            JSONState.PARAM_COLON: _norm_encode('": '),
            JSONState.PARAM_COMMA: _norm_encode(', "'),
            JSONState.END: _norm_encode("}}"),
        }

        self.progress = 0
        self.prompt_list = prompt.split()
        self.current_param_nb = 0
        self.total_params = 0  # Set when function name is known
        self.prompt_decimal_counts = utils.extract_decimal_counts(prompt)
        self.prompt_numbers = utils.extract_numbers(prompt)

    def _get_all_token_ids(self) -> set[int]:
        """Return every unique token ID in the vocabulary."""
        return set(self.vocabulary.all_ids)

    def _get_adjusted_param_index(self) -> int:
        """Get the current parameter index, adjusted for PARAM_VAL state.

        When in PARAM_VAL state (filling the parameter value), we need to look
        at the parameter we're currently filling, not the next one.

        Returns:
            The zero-based index of the parameter currently being processed.
        """
        idx = self.current_param_nb
        if self.state == JSONState.PARAM_VAL and idx > 0:
            idx -= 1
        return idx

    def _get_current_function_params(self) -> dict | None:
        """Return current function parameters, or ``None`` if unavailable."""
        if self.current_function_name not in self.functions_names:
            return None
        params = self.functions.get_function_parameters_by_name(
            self.current_function_name
        )
        if not isinstance(params, dict):
            return None
        return params

    def _get_current_param_type(self) -> str | None:
        """Get the type of the current parameter, or None if invalid."""
        params = self._get_current_function_params()
        if params is None:
            return None

        idx = self._get_adjusted_param_index()
        values: list[Parameter] = [*params.values()]
        if idx < 0 or idx >= len(values):
            return None

        return values[idx].type

    def _get_current_param_name(self) -> str | None:
        """Get the name of the current parameter, or None if invalid."""
        params = self._get_current_function_params()
        if params is None:
            return None

        idx = self._get_adjusted_param_index()
        values: list[str] = [*params.keys()]
        if idx < 0 or idx >= len(values):
            return None
        return values[idx]

    def _get_current_param_index(self) -> int | None:
        """Get the index of the current parameter, or None if invalid."""
        if self.current_function_name not in self.functions_names:
            return None

        idx = self._get_adjusted_param_index()
        if idx < 0:
            return None
        return idx

    def _get_target_decimals_for_current_param(self) -> int | None:
        """Return the prompt-derived decimal precision for the parameter."""
        idx = self._get_current_param_index()
        if idx is None:
            return None
        if idx < len(self.prompt_decimal_counts):
            return self.prompt_decimal_counts[idx]
        return None

    def _get_target_number_for_current_param(self) -> str | None:
        """Return the prompt-derived numeric literal for the parameter."""
        idx = self._get_current_param_index()
        if idx is None or idx >= len(self.prompt_numbers):
            return None

        value = self.prompt_numbers[idx]
        if self._get_current_param_type() == "integer":
            if value != value.to_integral_value():
                return None
            return str(int(value))
        return format(value, "f")

    def get_target_tokens_for_current_state(self) -> list[int]:
        """Return unconsumed fixed token IDs for the current state."""
        target = self.targets.get(self.state, [])
        return target[self.progress:]

    def complete_empty_fixed_sequence(self) -> None:
        """Advance a fixed state whose encoded target contains no token."""
        if self.state not in self.targets or self.targets[self.state]:
            raise ValueError("Current state is not an empty fixed sequence")
        self._update_state()
        self.current_text = ""
        self.progress = 0

    def is_in_fixed_sequence(self) -> bool:
        """Return whether the current state emits a fixed token sequence."""
        return self.state in self.targets

    def get_allowed_tokens(self) -> set[int]:
        """Compute token IDs allowed by the current state and partial value.

        Returns:
            Token IDs that may legally follow the generated prefix.
        """
        # 1. Sequence fixe (JSON)
        if self.state in self.targets:
            target = self.targets[self.state]
            if self.progress < len(target):
                return {target[self.progress]}

        # 2. Cas dynamique
        if self.state == JSONState.NAME_VAL:
            return self._allowed_tokens_for_function_name()
        if self.state == JSONState.PARAM_NAME:
            return self._allowed_tokens_for_parameter_name()
        if self.state == JSONState.PARAM_VAL:
            return self._allowed_tokens_for_parameter_value()

        return self._get_all_token_ids()

    def _allowed_tokens_for_parameter_name(self) -> set[int]:
        """Get the allowed token ids for the current parameter name."""
        allowed_tokens: set[int] = set()
        params = self._get_current_function_params()
        if params is not None:
            param_names = [*params.keys()]
            if self.current_param_nb < len(param_names):
                param_name = param_names[self.current_param_nb]
                allowed_tokens.update(
                    self._get_allowed_tokens_for_string(
                        param_name,
                        self.current_text,
                    )
                )
        return allowed_tokens

    def _allowed_tokens_for_parameter_value(self) -> set[int]:
        """Get the allowed token ids for the current parameter value."""
        allowed_tokens: set[int] = set()
        param_type = self._get_current_param_type()

        if param_type == "string":
            return self._allowed_tokens_for_param_string()

        elif param_type in {"number", "integer"}:
            return self._allowed_tokens_for_param_number()

        elif param_type == "boolean":
            for literal in ("true", "false"):
                allowed_tokens.update(
                    self._get_allowed_tokens_for_string(
                        literal,
                        self.current_text,
                    )
                )
            return allowed_tokens

        return allowed_tokens

    def _allowed_tokens_for_param_string(self) -> set[int]:
        """Get the allowed token ids for the current string parameter value."""
        quote_ids = self.vocabulary.exact_ids('"')

        if not self.current_text:
            # Start string with opening quote.
            return quote_ids

        if not self.current_text.startswith('"'):
            # Prevent generating unquoted strings.
            return set()

        if utils.get_repeating_pattern(self.current_text):
            return quote_ids

        param_name = self._get_current_param_name()
        grounded_value = self.grounded_string_parameters.get(param_name or "")
        if grounded_value is not None:
            allowed_tokens = self._allowed_tokens_for_grounded_string(
                grounded_value
            )
        elif param_name == 'replacement':
            allowed_tokens = self._allowed_tokens_for_replacement()
        else:
            allowed_tokens = self.vocabulary.json_string_content_ids()

        safe_tokens = {
            token_id
            for token_id in allowed_tokens
            if self._is_safe_string_continuation(token_id)
        }
        safe_tokens.update(quote_ids)
        if grounded_value is not None:
            generated = self.current_text[1:]
            if generated != grounded_value:
                safe_tokens.difference_update(quote_ids)
        return safe_tokens

    def _allowed_tokens_for_grounded_string(self, value: str) -> set[int]:
        """Return token fragments that continue an extracted string value."""
        generated = self.current_text[1:]
        return self._get_allowed_tokens_for_string(value, generated)

    def _is_safe_string_continuation(self, token_id: int) -> bool:
        """Return whether a token safely extends the current string value."""
        candidate = self.current_text + self.vocabulary.text(token_id)
        if len(candidate) > MAX_STRING_LENGTH:
            return False
        return not utils.get_repeating_pattern(candidate)

    def _allowed_tokens_for_param_number(self) -> set[int]:
        """Get the allowed token ids for the current number parameter value."""
        text = self.current_text
        has_dot = "." in text
        frac_len = len(text.split(".", 1)[1]) if has_dot else 0
        target_decimals = self._get_target_decimals_for_current_param()
        is_integer = self._get_current_param_type() == "integer"
        target_number = self._get_target_number_for_current_param()

        if target_number is not None:
            if text == target_number:
                return self.vocabulary.number_terminator_ids()
            return self._get_allowed_tokens_for_string(target_number, text)

        digit_tokens = set()
        for token_id in self.vocabulary.number_fragment_ids():
            token_text = self.vocabulary.text(token_id)
            candidate = text + token_text

            if not utils.is_valid_number_fragment(candidate):
                continue

            if is_integer and ("." in candidate or "e" in candidate):
                continue

            # If prompt has numeric literals, keep their decimal precision.
            if target_decimals is not None:
                if "e" in candidate:
                    continue
                if target_decimals == 0 and "." in candidate:
                    continue
                if target_decimals > 0 and "." in candidate:
                    candidate_frac_len = len(candidate.split(".", 1)[1])
                    if candidate_frac_len > target_decimals:
                        continue
            else:
                # Fallback when prompt has no numeric literal.
                if has_dot and frac_len >= 2:
                    continue

            digit_tokens.add(token_id)

        if not utils.is_complete_number(text):
            return digit_tokens

        # Number is complete.
        # Only allow termination when precision target is met.
        if is_integer:
            pass
        elif target_decimals is not None:
            if target_decimals == 0:
                if "." in text:
                    return digit_tokens
            else:
                if "." not in text:
                    return digit_tokens
                if frac_len < target_decimals:
                    return digit_tokens
        else:
            if has_dot and frac_len < 2:
                return digit_tokens

        terminator_tokens = self.vocabulary.number_terminator_ids()
        if terminator_tokens:
            # Once a complete value has reached the expected precision, stop
            # extending it. Otherwise greedy decoding can emit digits until the
            # global response-token limit is reached, leaving invalid JSON.
            return terminator_tokens
        return digit_tokens

    def _allowed_tokens_for_function_name(self) -> set[int]:
        """Get the allowed token ids for the current function name."""
        allowed_tokens: set[int] = set()
        still_possible = [
            s for s in self.functions_names
            if s.startswith(self.current_text)
        ]
        for s in still_possible:
            allowed_tokens.update(
                self._get_allowed_tokens_for_string(
                    s,
                    self.current_text,
                )
            )
        if self.current_text in self.functions_names:
            boundary_state = self._state_after_function_name()
            if boundary_state is not None:
                target = self.targets[boundary_state]
                if target:
                    allowed_tokens.add(target[0])
        return allowed_tokens

    def _allowed_tokens_for_replacement(self) -> set[int]:
        """Get the allowed token ids for the 'replacement' parameter value."""
        allowed_tokens: set[int] = set()
        generated = self.current_text[1:]  # Skip the opening quote.
        still_possible = [
            value for value in self.prompt_list if value.startswith(generated)
        ]
        for s in still_possible:
            allowed_tokens.update(
                self._get_allowed_tokens_for_string(
                    s,
                    generated,
                )
            )
        if generated in self.prompt_list:
            allowed_tokens.update(self.vocabulary.exact_ids('"'))
        return allowed_tokens

    def _get_allowed_tokens_for_string(
        self,
        target_string: str,
        current_generated_text: str,
    ) -> set[int]:
        """Return token IDs that continue a target string from its prefix."""
        return self.vocabulary.ids_continuing(
            target_string,
            current_generated_text,
        )

    def update(self, token_id: int) -> bool:
        """Consume one token and advance the state machine when appropriate.

        Structural delimiters terminating numeric values advance the machine
        but are not retained because the following fixed state emits them.

        Args:
            token_id: Token ID selected by constrained generation.

        Returns:
            Whether the caller should append the token to the response.

        Raises:
            ValueError: If a token violates a fixed sequence.
        """
        token_text = self.vocabulary.text(token_id)

        if self.state == JSONState.NAME_VAL:
            boundary_state = self._state_after_function_name()
            if (
                boundary_state is not None
                and self.targets[boundary_state]
                and token_id == self.targets[boundary_state][0]
            ):
                self._commit_current_function()
                self.state = boundary_state
                self.current_text = ""
                self.progress = 0

        if self.state == JSONState.PARAM_VAL:
            param_type = self._get_current_param_type()
            if (
                param_type in {"number", "integer"}
                and utils.is_number_terminator_token(token_text)
                and utils.is_complete_number(self.current_text)
            ):
                self._update_state()
                self.current_text = ""
                return False

        self.current_text += token_text

        if self.state in self.targets:
            target = self.targets[self.state]
            if token_id == target[self.progress]:
                self.progress += 1
                if self.progress == len(target):
                    self._update_state()
                    self.current_text = ""
                    self.progress = 0
            else:
                raise ValueError("Invalid token in fixed sequence")

        elif self.state == JSONState.NAME_VAL:
            self.current_function_name = self.current_text

        elif self.state == JSONState.PARAM_NAME:
            params = self._get_current_function_params()
            if params is not None:
                param_names = [*params.keys()]
                if self.current_text in param_names:
                    self.current_param_nb += 1
                    self._update_state()
                    self.current_text = ""

        elif self.state == JSONState.PARAM_VAL:
            param_type = self._get_current_param_type()
            if (
                param_type == "boolean"
                and self.current_text in {"true", "false"}
            ):
                self._update_state()
                self.current_text = ""
            elif (
                param_type == "string"
                and len(self.current_text) > 1
                and self.current_text.startswith('"')
                and token_text == '"'
            ):
                self._update_state()
                self.current_text = ""

        return True

    def _state_after_function_name(self) -> JSONState | None:
        """Return the structural state selected by a complete function name."""
        if self.current_text not in self.functions_names:
            return None
        if self.functions.get_nb_parameters(self.current_text) == 0:
            return JSONState.EMPTY_PARAMS
        return JSONState.PARAMS_KEY

    def _commit_current_function(self) -> None:
        """Store the function selected by an explicit boundary token."""
        validate_prompt_capacity(
            self.functions,
            self.prompt,
            self.current_text,
        )
        self.current_function_name = self.current_text
        self.total_params = self.functions.get_nb_parameters(
            self.current_function_name
        )

    def _update_state(self) -> None:
        """Advance to the next structural JSON generation state."""
        if self.state == JSONState.START:
            self.state = JSONState.PROMPT_KEY
        elif self.state == JSONState.PROMPT_KEY:
            self.state = JSONState.PROMPT_VAL
        elif self.state == JSONState.PROMPT_VAL:
            self.state = JSONState.NAME_KEY
        elif self.state == JSONState.NAME_KEY:
            self.state = JSONState.NAME_VAL
        elif self.state == JSONState.EMPTY_PARAMS:
            self.state = JSONState.STOP
        elif self.state == JSONState.NAME_VAL:
            self.state = JSONState.PARAMS_KEY
        elif self.state == JSONState.PARAMS_KEY:
            self.state = JSONState.PARAM_NAME
        elif self.state == JSONState.PARAM_NAME:
            self.state = JSONState.PARAM_COLON
        elif self.state == JSONState.PARAM_COLON:
            self.state = JSONState.PARAM_VAL
        elif self.state == JSONState.PARAM_VAL:
            if self.current_param_nb < self.total_params:
                self.state = JSONState.PARAM_COMMA
            else:
                self.state = JSONState.END
        elif self.state == JSONState.PARAM_COMMA:
            self.state = JSONState.PARAM_NAME
        elif self.state == JSONState.END:
            self.state = JSONState.STOP
        else:
            raise ValueError("Invalid state transition")
