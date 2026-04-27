from enum import Enum, auto
import re


class JSONState(Enum):
    START = auto()          # {
    PROMPT_KEY = auto()     # "prompt": "
    PROMPT_VAL = auto()     # [Valeur du prompt]
    NAME_KEY = auto()       # ", "name": "
    NAME_VAL = auto()       # [Nom de la fonction]
    PARAMS_KEY = auto()     # ", "parameters": {
    PARAM_NAME = auto()     # "nom_du_parametre"
    PARAM_COLON = auto()    # :
    PARAM_VAL = auto()      # [Valeur selon le type]
    PARAM_COMMA = auto()    # ,
    END = auto()            # }
    STOP = auto()           # Token de fin de génération


class JSONStateMachine:
    def __init__(self, model, functions_def, token_to_id, prompt=""):
        self.model = model
        self.state = JSONState.START
        self.buffer_tokens: list[int] = []
        self.current_text = ""
        self.current_function_name = ""

        self.functions_names = functions_def.list_functions_name()
        self.functions = functions_def
        self.token_to_id = token_to_id

        # Targets encodes
        self.targets = {
            JSONState.START: model.encode("{")[0].tolist(),
            JSONState.PROMPT_KEY: model.encode('"prompt": "')[0].tolist(),
            JSONState.NAME_KEY: model.encode('", "name": "')[0].tolist(),
            JSONState.PARAMS_KEY: model.encode('", "parameters": {"')[0].tolist(),
            JSONState.PROMPT_VAL: model.encode(prompt)[0].tolist(),
            JSONState.PARAM_COLON: model.encode('": ')[0].tolist(),
            JSONState.PARAM_COMMA: model.encode(', "')[0].tolist(),
            JSONState.END: model.encode("}")[0].tolist(),
        }

        self.progress = 0
        self.current_param_nb = 0
        self.prompt_decimal_counts = self._extract_prompt_decimal_counts(prompt)

    def _extract_prompt_decimal_counts(self, prompt: str) -> list[int]:
        counts = []
        for match in re.finditer(r"-?\d+(?:\.(\d+))?", prompt):
            frac = match.group(1)
            counts.append(len(frac) if frac is not None else 1)
        return counts

    def _is_valid_number_fragment(self, text: str) -> bool:
        if text == "":
            return True

        chars = set("0123456789-.e")
        if any(ch not in chars for ch in text):
            return False

        if text.count("e") > 1:
            return False
        if text.count(".") > 1:
            return False

        e_pos = text.find("e")
        if e_pos != -1 and text.find(".", e_pos) != -1:
            return False

        if "-" in text:
            for i, ch in enumerate(text):
                if ch != "-":
                    continue
                if i == 0:
                    continue
                if i > 0 and text[i - 1] == "e":
                    continue
                return False

        return True

    def _is_complete_number(self, text: str) -> bool:
        if text == "":
            return False

        if text[-1] in {"-", ".", "e"}:
            return False

        if "e" in text:
            left, right = text.split("e", 1)
            if left in {"", "-", ".", "-."}:
                return False
            if right in {"", "+", "-"}:
                return False
            if right[0] in {"+", "-"}:
                right = right[1:]
            return right.isdigit()

        if text in {"-", ".", "-."}:
            return False

        return any(ch.isdigit() for ch in text)

    def _get_number_token_ids(self) -> set[int]:
        allowed = set()
        for token_str, token_id in self.token_to_id.items():
            clean_t = token_str.replace('Ġ', ' ').replace(' ', ' ')
            if clean_t == "":
                continue
            if " " in clean_t:
                continue
            if self._is_valid_number_fragment(clean_t):
                allowed.add(token_id)
        return allowed

    def _get_number_terminator_token_ids(self) -> set[int]:
        terminators = set()
        for exact_text in [" ", "  ", ",", ", ", " ,", "}"]:
            terminators.update(self._get_exact_token_ids(exact_text))
        return terminators

    def _is_number_terminator_token(self, token_text: str) -> bool:
        return token_text in {" ", "  ", ",", ", ", " ,", "}"}

    def _get_exact_token_ids(self, exact_text: str) -> set[int]:
        allowed = set()
        for token_str, token_id in self.token_to_id.items():
            clean_t = token_str.replace('Ġ', ' ').replace(' ', ' ')
            if clean_t == exact_text:
                allowed.add(token_id)
        return allowed

    def _get_current_param_type(self):
        if self.current_function_name not in self.functions_names:
            return None

        params = self.functions.get_function_parameters_by_name(
            self.current_function_name
        )
        if not isinstance(params, dict):
            return None

        idx = self.current_param_nb
        if self.state == JSONState.PARAM_VAL and idx > 0:
            idx -= 1

        values = [*params.values()]
        if idx < 0 or idx >= len(values):
            return None

        return values[idx].type

    def _get_current_param_index(self) -> int | None:
        if self.current_function_name not in self.functions_names:
            return None

        idx = self.current_param_nb
        if self.state == JSONState.PARAM_VAL and idx > 0:
            idx -= 1

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
        return self.targets.get(self.state, [])

    def is_in_fixed_sequence(self) -> bool:
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

        return set(range(self.token_to_id.__len__()))

    def _allowed_tokens_for_parameter_name(self) -> set[int]:
        allowed_tokens = set()
        if self.current_function_name in self.functions_names:
            params = self.functions.get_function_parameters_by_name(
                self.current_function_name
            )
            if isinstance(params, dict):
                param_name = [*params.keys()][self.current_param_nb]
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
            pass
        elif param_type == "number":
            text = self.current_text
            has_dot = "." in text
            frac_len = len(text.split(".", 1)[1]) if has_dot else 0
            target_decimals = self._get_target_decimals_for_current_param()

            digit_tokens = set()
            for token_id in self._get_number_token_ids():
                token_text = self.model.decode([token_id])
                candidate = text + token_text

                if not self._is_valid_number_fragment(candidate):
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
                    # Fallback when prompt has no numeric literal for this param.
                    if has_dot and frac_len >= 2:
                        continue

                digit_tokens.add(token_id)

            if not self._is_complete_number(text):
                return digit_tokens

            # Number is complete. Only allow termination when precision target is met.
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

            terminator_tokens = self._get_number_terminator_token_ids()
            if terminator_tokens:
                return digit_tokens | terminator_tokens
            return digit_tokens

        elif param_type == "boolean":
            pass

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
        token_text = self.model.decode([token_id])

        if self.state == JSONState.PARAM_VAL:
            param_type = self._get_current_param_type()
            if (
                param_type == "number"
                and self._is_number_terminator_token(token_text)
                and self._is_complete_number(self.current_text)
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
                self.param_nb = self.functions.get_nb_parameters(
                    self.current_function_name
                )
                self._update_state()
                self.current_text = ""

        elif self.state == JSONState.PARAMS_KEY:
            params = self.functions.get_function_parameters_by_name(
                self.current_function_name
            )
            params_names = [*params.keys()]
            if self.current_text in params_names:
                self._update_state()
                self.current_text = ""

        elif self.state == JSONState.PARAM_NAME:
            params = self.functions.get_function_parameters_by_name(
                self.current_function_name
            )
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
            if self.current_param_nb < self.param_nb:
                self.state = JSONState.PARAM_COMMA
            else:
                self.state = JSONState.END
        elif self.state == JSONState.PARAM_COMMA:
            self.state = JSONState.PARAM_NAME
        elif self.state == JSONState.END:
            self.state = JSONState.STOP
        else:
            raise ValueError("Invalid state transition")
