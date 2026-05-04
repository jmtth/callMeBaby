from src.models import JSONState
from src import utils


class JSONStateMachine:
    def __init__(self, model, functions_def, token_to_id, prompt=""):
        """State machine to track the generation of a JSON function call.

        Attributes:
            model: The language model instance.
            state: The current state of the state machine.
            buffer_tokens: A list of token IDs representing the generated text.
            current_text: The current text being generated.
            current_function_name: The name of the function being called.
        """
        self.model = model
        self.state = JSONState.START
        self.buffer_tokens: list[int] = []
        self.current_text = ""
        self.current_function_name = ""

        self.functions_names = functions_def.list_functions_name()
        self.functions = functions_def
        self.token_to_id = token_to_id

        # Normalize encodings (support lists, numpy arrays, tensors)
        def _norm_encode(s: str) -> list[int]:
            enc0 = model.encode(s)[0]
            if hasattr(enc0, "tolist"):
                return enc0.tolist()
            return list(enc0)

        # Targets encodes
        self.targets = {
            JSONState.START: _norm_encode("{"),
            JSONState.PROMPT_KEY: _norm_encode('"prompt": "'),
            JSONState.NAME_KEY: _norm_encode('", "name": "'),
            JSONState.PARAMS_KEY: _norm_encode('", "parameters": {"'),
            JSONState.PROMPT_VAL: _norm_encode(prompt),
            JSONState.PARAM_COLON: _norm_encode('": '),
            JSONState.PARAM_COMMA: _norm_encode(', "'),
            JSONState.END: _norm_encode("}}"),
        }

        self.progress = 0
        self.current_param_nb = 0
        self.total_params = 0  # Set when function name is known
        self.prompt_decimal_counts = utils.extract_decimal_counts(prompt)

    def _get_all_token_ids(self) -> set[int]:
        return set(self.token_to_id.values())

    def _get_adjusted_param_index(self) -> int:
        """Get the current parameter index, adjusted for PARAM_VAL state.

        When in PARAM_VAL state (filling the parameter value), we need to look
        at the parameter we're currently filling, not the next one.

        returns:
            int : idx of the parameter
        """
        idx = self.current_param_nb
        if self.state == JSONState.PARAM_VAL and idx > 0:
            idx -= 1
        return idx

    def _get_current_function_params(self) -> dict | None:
        """Get the parameters dict for the current function,
        or None if invalid.
        """
        if self.current_function_name not in self.functions_names:
            return None
        params = self.functions.get_function_parameters_by_name(
            self.current_function_name
        )
        if not isinstance(params, dict):
            return None
        return params

    def _get_current_param_type(self):
        params = self._get_current_function_params()
        if params is None:
            return None

        idx = self._get_adjusted_param_index()
        values = [*params.values()]
        if idx < 0 or idx >= len(values):
            return None

        return values[idx].type

    def _get_current_param_index(self) -> int | None:
        if self.current_function_name not in self.functions_names:
            return None

        idx = self._get_adjusted_param_index()
        if idx < 0:
            return None
        return idx

    def _get_target_decimals_for_current_param(self) -> int | None:
        idx = self._get_current_param_index()
        if idx is None:
            return None
        if idx < len(self.prompt_decimal_counts):
            return self.prompt_decimal_counts[idx]
        return None

    def get_target_tokens_for_current_state(self) -> list[int]:
        """"Get the target token ids for the current state."""
        return self.targets.get(self.state, [])

    def is_in_fixed_sequence(self) -> bool:
        """Check if we are currently in a fixed sequence of tokens
        (like the JSON structure or the prompt)."""
        return self.state in self.targets

    def get_allowed_tokens(self) -> set[int]:
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
        allowed_tokens = set()
        params = self._get_current_function_params()
        if params is not None:
            param_names = [*params.keys()]
            if self.current_param_nb < len(param_names):
                param_name = param_names[self.current_param_nb]
                allowed_tokens.update(
                    self._get_allowed_tokens_for_string(
                        param_name,
                        self.current_text,
                        self.token_to_id,
                    )
                )
        return allowed_tokens

    def _allowed_tokens_for_parameter_value(self) -> set[int]:
        allowed_tokens = set()
        param_type = self._get_current_param_type()

        if param_type == "string":
            # For strings, allow any chars until closing quote
            quote_id = self.token_to_id.get('"')
            if quote_id is not None:
                # If already in string, allow quote to close it
                if self.current_text.startswith('"'):
                    allowed_tokens.add(quote_id)
                else:
                    # Start string, allow opening quote
                    allowed_tokens.add(quote_id)
            # Allow any character tokens inside string
            allowed_tokens.update(self._get_all_token_ids())
            return allowed_tokens
        elif param_type == "number":
            text = self.current_text
            has_dot = "." in text
            frac_len = len(text.split(".", 1)[1]) if has_dot else 0
            target_decimals = self._get_target_decimals_for_current_param()

            digit_tokens = set()
            for token_id in utils.get_number_token_ids(self.token_to_id):
                token_text = self.model.decode([token_id])
                candidate = text + token_text

                if not utils.is_valid_number_fragment(candidate):
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
            if target_decimals is not None:
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

            terminator_tokens = utils.get_number_terminator_token_ids(
                self.token_to_id)
            if terminator_tokens:
                return digit_tokens | terminator_tokens
            return digit_tokens

        elif param_type == "boolean":
            # Only allow 'true' or 'false'
            allowed_tokens = set()
            for token_str, token_id in self.token_to_id.items():
                clean_token = token_str.replace('Ġ', ' ').replace(' ', ' ')
                if clean_token in {"true", "false"}:
                    allowed_tokens.add(token_id)
            return allowed_tokens

        return allowed_tokens

    def _allowed_tokens_for_function_name(self) -> set[int]:
        allowed_tokens = set()
        still_possible = [
            s for s in self.functions_names
            if s.startswith(self.current_text)
        ]
        for s in still_possible:
            allowed_tokens.update(
                self._get_allowed_tokens_for_string(
                    s,
                    self.current_text,
                    self.token_to_id,
                )
            )
        return allowed_tokens

    def _get_allowed_tokens_for_string(
        self,
        target_string: str,
        current_generated_text: str,
        token_to_id: dict[str, int],
    ) -> set[int]:
        if current_generated_text == target_string:
            space_id = token_to_id.get(" ")
            return {space_id} if space_id is not None else set()

        remaining = target_string[len(current_generated_text):]

        allowed = set()
        for token_str, t_id in token_to_id.items():
            clean_token = token_str.replace('Ġ', ' ').replace(' ', ' ')
            if remaining.startswith(clean_token) and clean_token != "":
                allowed.add(t_id)
        return allowed

    def update(self, token_id: int) -> bool:
        """Update the state machine with the new token.
        check if the token is valid in the current state,
        update the state accordingly,
        and return whether to keep the token or not.

        Args:
            token_id: the new token id to update the state machine with.

        Returns:
            bool: whether to keep the token (True) or not (False)."""
        token_text = self.model.decode([token_id])

        if self.state == JSONState.PARAM_VAL:
            param_type = self._get_current_param_type()
            if (
                param_type == "number"
                and utils.is_number_terminator_token(token_text)
                and utils.is_complete_number(self.current_text)
            ):
                self._update_state()
                self.current_text = ""
                return False

        self.buffer_tokens.append(token_id)
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
            if self.current_text in self.functions_names:
                self.total_params = self.functions.get_nb_parameters(
                    self.current_function_name
                )
                self._update_state()
                self.current_text = ""

        elif self.state == JSONState.PARAMS_KEY:
            params = self._get_current_function_params()
            if params is not None:
                params_names = [*params.keys()]
                if self.current_text in params_names:
                    self._update_state()
                    self.current_text = ""

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
            if param_type == "string" and self.current_text.endswith('"'):
                self._update_state()
                self.current_text = ""

        return True

    def _update_state(self):
        if self.state == JSONState.START:
            self.state = JSONState.PROMPT_KEY
        elif self.state == JSONState.PROMPT_KEY:
            self.state = JSONState.PROMPT_VAL
        elif self.state == JSONState.PROMPT_VAL:
            self.state = JSONState.NAME_KEY
        elif self.state == JSONState.NAME_KEY:
            self.state = JSONState.NAME_VAL
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
