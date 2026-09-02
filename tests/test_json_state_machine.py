from src.JSONStateMachine import JSONStateMachine
from llm_sdk import Small_LLM_Model
from src.functions_manager import FunctionsDefinition
from src.models import JSONState
from typing import cast


class DummyFunctionsDef:
    """Provide the DummyFunctionsDef test double."""
    def list_functions_name(self):
        """Return function names exposed by the test double."""
        return []


class DummyParam:
    """Provide the DummyParam test double."""
    def __init__(self, type_: str):
        """Initialize the test double."""
        self.type = type_


class StringParamFunctionsDef:
    """Provide the StringParamFunctionsDef test double."""
    def list_functions_name(self):
        """Return function names exposed by the test double."""
        return ["fn_echo"]

    def get_function_parameters_by_name(self, name: str):
        """Return parameters exposed by the test double."""
        return {"text": DummyParam("string")}


class NotParamFunctionsDef:
    """Provide the NotParamFunctionsDef test double."""
    def list_functions_name(self):
        """Return function names exposed by the test double."""
        return ["fn_not_good"]

    def get_function_parameters_by_name(self, name: str):
        """Return parameters exposed by the test double."""
        return None


class NumberParamFunctionsDef:
    """Provide the NumberParamFunctionsDef test double."""
    def list_functions_name(self):
        """Return function names exposed by the test double."""
        return ["fn_add"]

    def get_function_parameters_by_name(self, name: str):
        """Return parameters exposed by the test double."""
        return {"value": DummyParam("number")}


class IntegerParamFunctionsDef:
    """Provide the IntegerParamFunctionsDef test double."""
    def list_functions_name(self):
        """Return function names exposed by the test double."""
        return ["fn_even"]

    def get_function_parameters_by_name(self, name: str):
        """Return parameters exposed by the test double."""
        return {"value": DummyParam("integer")}


class BooleanParamFunctionsDef:
    """Provide the BooleanParamFunctionsDef test double."""
    def list_functions_name(self):
        """Return function names exposed by the test double."""
        return ["fn_toggle"]

    def get_function_parameters_by_name(self, name: str):
        """Return parameters exposed by the test double."""
        return {"enabled": DummyParam("boolean")}


class UnsupportedParamFunctionsDef:
    """Provide the UnsupportedParamFunctionsDef test double."""
    def list_functions_name(self):
        """Return function names exposed by the test double."""
        return ["fn_collect"]

    def get_function_parameters_by_name(self, name: str):
        """Return parameters exposed by the test double."""
        return {"items": DummyParam("array")}


class EmptyParamFunctionsDef:
    """Provide the EmptyParamFunctionsDef test double."""
    def list_functions_name(self):
        """Return function names exposed by the test double."""
        return ["fn_ping"]

    def get_function_parameters_by_name(self, name: str):
        """Return parameters exposed by the test double."""
        return {}

    def get_nb_parameters(self, name: str):
        """Return the test function parameter count."""
        return 0


class PrefixFunctionsDef:
    """Provide the PrefixFunctionsDef test double."""
    def list_functions_name(self):
        """Return function names exposed by the test double."""
        return ["get", "get_weather"]

    def get_function_parameters_by_name(self, name: str):
        """Return parameters exposed by the test double."""
        return {}

    def get_nb_parameters(self, name: str):
        """Return the test function parameter count."""
        return 0


class FakeModel:
    """Provide the FakeModel test double."""
    def encode(self, s: str):
        # return a list-like structure where [0] is a list of ints
        # This simulates the behavior of a model that returns Ids
        # ord() is used to convert characters to their ASCII integer
        """Encode text into token IDs for the test double."""
        return [[ord(c) for c in s]]

    def decode(self, ids: list[int]) -> str:
        """Decode token IDs into text for the test double."""
        return ''.join(chr(i) for i in ids)


class MappedFakeModel(FakeModel):
    """Provide the MappedFakeModel test double."""

    def __init__(self, token_to_id: dict[str, int]):
        """Initialize the test double."""
        self.id_to_token = {
            token_id: token for token, token_id in token_to_id.items()
        }

    def decode(self, ids: list[int]) -> str:
        """Decode token IDs into text for the test double."""
        return ''.join(
            self.id_to_token.get(token_id, chr(token_id))
            for token_id in ids
        )


class TensorLikeEncoding:
    """Provide the TensorLikeEncoding test double."""

    def __init__(self, values: list[int]):
        """Initialize the test double."""
        self.values = values

    def tolist(self):
        """Return the wrapped values as a list."""
        return self.values


class TensorLikeModel():
    """Provide the TensorLikeModel test double."""
    def encode(self, s: str):
        """Encode text into token IDs for the test double."""
        return [TensorLikeEncoding([ord(c) for c in s])]


def test_norm_encode_converts_tensor_like_encoding_to_list():
    """Norm encode converts tensor like encoding to list."""
    model = TensorLikeModel()
    funcs = DummyFunctionsDef()
    token_to_id = {chr(i): i for i in range(32, 128)}

    sm = JSONStateMachine(
        cast(Small_LLM_Model, model),
        cast(FunctionsDefinition, funcs), token_to_id)

    assert sm.targets[JSONState.START] == [ord("{")]
    assert isinstance(sm.targets[JSONState.START], list)
    assert all(isinstance(token_id, int)
               for token_id in sm.targets[JSONState.START])


def test_get_all_token_ids_removes_duplicate_ids():
    """Get all token ids removes duplicate ids."""
    model = FakeModel()
    funcs = DummyFunctionsDef()
    token_to_id = {"first": 10, "second": 20, "alias": 10}

    sm = JSONStateMachine(
        cast(Small_LLM_Model, model),
        cast(FunctionsDefinition, funcs), token_to_id)

    assert sm._get_all_token_ids() == {10, 20}


def test_get_adjusted_param_index():
    """Get adjusted param index."""
    model = FakeModel()
    funcs = DummyFunctionsDef()
    token_to_id = {chr(i): i for i in range(32, 128)}

    sm = JSONStateMachine(
        cast(Small_LLM_Model, model),
        cast(FunctionsDefinition, funcs), token_to_id)

    sm.current_param_nb = 2
    sm.state = JSONState.PARAM_VAL
    assert sm._get_adjusted_param_index() == 1
    sm.current_param_nb = 0
    assert sm._get_adjusted_param_index() == 0


def test_extract_decimal_counts():
    """Extract decimal counts."""
    model = FakeModel()
    funcs = DummyFunctionsDef()
    token_to_id = {chr(i): i for i in range(32, 128)}

    # prompt with integer, float with 3 decimals, float with 1 decimal
    prompt = "Value A: 12, Value B: 3.456, Value C: -7.0"
    sm = JSONStateMachine(
        cast(Small_LLM_Model, model),
        cast(FunctionsDefinition, funcs), token_to_id, prompt=prompt)

    # Expect: 12 -> 1 decimal, 3.456 -> 3, -7.0 -> 1
    assert sm.prompt_decimal_counts == [1, 3, 1]


def test_get_not_found_current_function_parameters():
    """Get not found current function parameters."""
    model = FakeModel()
    funcs = StringParamFunctionsDef()
    token_to_id = {chr(i): i for i in range(32, 128)}

    sm = JSONStateMachine(
        cast(Small_LLM_Model, model),
        cast(FunctionsDefinition, funcs), token_to_id)

    sm.current_function_name = "fn_not_good"
    params = sm._get_current_function_params()
    params_type = sm._get_current_param_type()
    assert params is None
    assert params_type is None


def test_get_found_current_function_without_parameters():
    """Get found current function without parameters."""
    model = FakeModel()
    funcs = NotParamFunctionsDef()
    token_to_id = {chr(i): i for i in range(32, 128)}

    sm = JSONStateMachine(
        cast(Small_LLM_Model, model),
        cast(FunctionsDefinition, funcs), token_to_id)

    sm.current_function_name = "fn_not_good"
    params = sm._get_current_function_params()
    params_type = sm._get_current_param_type()

    assert params is None
    assert params_type is None


def test_get_current_param_type_returns_none_for_out_of_range_index():
    """Get current param type returns none for out of range index."""
    model = FakeModel()
    funcs = StringParamFunctionsDef()
    token_to_id = {chr(i): i for i in range(32, 128)}

    sm = JSONStateMachine(
        cast(Small_LLM_Model, model),
        cast(FunctionsDefinition, funcs), token_to_id)
    sm.current_function_name = "fn_echo"
    sm.state = JSONState.PARAM_VAL
    sm.current_param_nb = 2

    assert sm._get_adjusted_param_index() == 1
    assert sm._get_current_param_type() is None


def test_get_not_found_current_function_param_name():
    """Get not found current function param name."""
    model = cast(Small_LLM_Model, FakeModel())
    funcs = cast(FunctionsDefinition, StringParamFunctionsDef())
    token_to_id = {chr(i): i for i in range(32, 128)}

    sm = JSONStateMachine(model, funcs, token_to_id)

    sm.current_function_name = "fn_not_good"
    param_name = sm._get_current_param_name()
    assert param_name is None
    sm.current_function_name = "fn_echo"
    sm.current_param_nb = 2
    param_name = sm._get_current_param_name()
    assert param_name is None


def test_get_not_found_current_function_param_index():
    """Get not found current function param index."""
    model = cast(Small_LLM_Model, FakeModel())
    funcs = cast(FunctionsDefinition, StringParamFunctionsDef())
    token_to_id = {chr(i): i for i in range(32, 128)}

    sm = JSONStateMachine(model, funcs, token_to_id)

    sm.current_function_name = "fn_not_good"
    param_index = sm._get_current_param_index()
    assert param_index is None
    sm.current_function_name = "fn_echo"
    sm.current_param_nb = -1
    param_index = sm._get_current_param_index()
    assert param_index is None


def test_get_not_found_current_function_target_decimals():
    """Get not found current function target decimals."""
    model = cast(Small_LLM_Model, FakeModel())
    funcs = cast(FunctionsDefinition, NumberParamFunctionsDef())
    token_to_id = {chr(i): i for i in range(32, 128)}

    sm = JSONStateMachine(model, funcs, token_to_id)

    sm.current_function_name = "fn_not_good"
    target_decimals = sm._get_target_decimals_for_current_param()
    assert target_decimals is None
    sm.current_function_name = "fn_add"
    sm.current_param_nb = 2
    target_decimals = sm._get_target_decimals_for_current_param()
    assert target_decimals is None


def test_is_in_fixed_sequence_state():
    """Is in fixed sequence state."""
    model = cast(Small_LLM_Model, FakeModel())
    funcs = cast(FunctionsDefinition, DummyFunctionsDef())
    token_to_id = {chr(i): i for i in range(32, 128)}

    sm = JSONStateMachine(model, funcs, token_to_id)

    # Fixed sequence states
    fixed_states = [
        JSONState.START,
        JSONState.PROMPT_KEY,
        JSONState.NAME_KEY,
        JSONState.PARAMS_KEY,
        JSONState.EMPTY_PARAMS,
        JSONState.PROMPT_VAL,
        JSONState.PARAM_COLON,
        JSONState.PARAM_COMMA,
        JSONState.END
    ]

    for state in fixed_states:
        sm.state = state
        assert sm.is_in_fixed_sequence() is True

    # Non-fixed sequence states
    non_fixed_states = [

        JSONState.NAME_VAL,
        JSONState.PARAM_NAME,
        JSONState.PARAM_VAL
    ]

    for state in non_fixed_states:
        sm.state = state
        assert sm.is_in_fixed_sequence() is False


def test_get_allowed_tokens_returns_next_token_of_fixed_sequence():
    """Get allowed tokens returns next token of fixed sequence."""
    model = cast(Small_LLM_Model, FakeModel())
    funcs = cast(FunctionsDefinition, DummyFunctionsDef())
    token_to_id = {chr(i): i for i in range(32, 128)}

    sm = JSONStateMachine(model, funcs, token_to_id)
    sm.state = JSONState.START
    sm.progress = 0

    assert sm.get_allowed_tokens() == {ord("{")}


def test_get_allowed_tokens_for_function_name_state():
    """Get allowed tokens for function name state."""
    model = cast(Small_LLM_Model, FakeModel())
    funcs = cast(FunctionsDefinition, StringParamFunctionsDef())
    token_to_id = {chr(i): i for i in range(32, 128)}

    sm = JSONStateMachine(model, funcs, token_to_id)
    sm.state = JSONState.NAME_VAL
    sm.current_text = ""

    assert sm.get_allowed_tokens() == {ord("f")}


def test_get_allowed_tokens_for_parameter_name_state():
    """Get allowed tokens for parameter name state."""
    model = cast(Small_LLM_Model, FakeModel())
    funcs = cast(FunctionsDefinition, StringParamFunctionsDef())
    token_to_id = {chr(i): i for i in range(32, 128)}

    sm = JSONStateMachine(model, funcs, token_to_id)
    sm.state = JSONState.PARAM_NAME
    sm.current_function_name = "fn_echo"
    sm.current_param_nb = 0
    sm.current_text = ""

    assert sm.get_allowed_tokens() == {ord("t")}


def test_get_allowed_tokens_falls_back_to_all_vocabulary_ids():
    """Get allowed tokens falls back to all vocabulary ids."""
    model = cast(Small_LLM_Model, FakeModel())
    funcs = cast(FunctionsDefinition, DummyFunctionsDef())
    token_to_id = {"first": 10, "second": 20, "alias": 10}

    sm = JSONStateMachine(model, funcs, token_to_id)
    sm.state = JSONState.STOP

    assert sm.get_allowed_tokens() == {10, 20}


def test_allowed_tokens_for_boolean_parameter():
    """Allowed tokens for boolean parameter."""
    token_to_id = {
        "true": 10,
        "false": 20,
        "null": 30,
        "yes": 40,
    }
    model = cast(Small_LLM_Model, MappedFakeModel(token_to_id))
    funcs = cast(FunctionsDefinition, BooleanParamFunctionsDef())

    sm = JSONStateMachine(model, funcs, token_to_id)
    sm.state = JSONState.PARAM_VAL
    sm.current_function_name = "fn_toggle"
    sm.current_param_nb = 0

    assert sm.get_allowed_tokens() == {10, 20}


def test_boolean_parameter_transitions_after_complete_literal():
    """Boolean parameter transitions after complete literal."""
    token_to_id = {"true": 10, "false": 20}
    model = cast(Small_LLM_Model, MappedFakeModel(token_to_id))
    funcs = cast(FunctionsDefinition, BooleanParamFunctionsDef())
    sm = JSONStateMachine(model, funcs, token_to_id)
    sm.state = JSONState.PARAM_VAL
    sm.current_function_name = "fn_toggle"
    sm.current_param_nb = 1
    sm.total_params = 1

    assert sm.update(10) is True
    assert sm.state == JSONState.END
    assert sm.current_text == ""


def test_allowed_tokens_for_unsupported_parameter_type_is_empty():
    """Allowed tokens for unsupported parameter type is empty."""
    model = cast(Small_LLM_Model, FakeModel())
    funcs = cast(FunctionsDefinition, UnsupportedParamFunctionsDef())
    token_to_id = {"[": 10, "]": 20}

    sm = JSONStateMachine(model, funcs, token_to_id)
    sm.state = JSONState.PARAM_VAL
    sm.current_function_name = "fn_collect"
    sm.current_param_nb = 0

    assert sm.get_allowed_tokens() == set()


def test_allowed_tokens_for_repeat_pattern():
    """Allowed tokens for repeat pattern."""
    token_to_id = {"a": 10, "b": 20, "c": 30, "\"": 40}
    model = cast(Small_LLM_Model, MappedFakeModel(token_to_id))
    funcs = cast(FunctionsDefinition, StringParamFunctionsDef())

    sm = JSONStateMachine(model, funcs, token_to_id)
    sm.state = JSONState.PARAM_VAL
    sm.current_function_name = "fn_echo"
    sm.current_param_nb = 0
    sm.current_text = '"abcabc'

    allowed_tokens = sm.get_allowed_tokens()

    assert allowed_tokens == {40}


def test_prompt_target_escapes_quotes_for_json_string():
    """Prompt target escapes quotes for JSON string."""
    model = FakeModel()
    funcs = DummyFunctionsDef()
    token_to_id = {chr(i): i for i in range(32, 128)}

    prompt = 'Replace "Hello 34 I\'m 233 years old" '
    sm = JSONStateMachine(
        cast(Small_LLM_Model, model),
        cast(FunctionsDefinition, funcs), token_to_id, prompt=prompt)

    encoded_prompt = sm.targets[JSONState.PROMPT_VAL]
    decoded_prompt = model.decode(encoded_prompt)

    assert decoded_prompt == 'Replace \\"Hello 34 I\'m 233 years old\\" '


def test_allowed_tokens_for_string_value_uses_actual_token_ids():
    """Allowed tokens for string value uses actual token ids."""
    token_to_id = {"x": 10, '"': 42, "y": 99}
    model = MappedFakeModel(token_to_id)
    funcs = StringParamFunctionsDef()

    sm = JSONStateMachine(
        cast(Small_LLM_Model, model),
        cast(FunctionsDefinition, funcs), token_to_id, prompt="")
    sm.state = JSONState.PARAM_VAL
    sm.current_function_name = "fn_echo"
    sm.current_param_nb = 0

    allowed_tokens = sm.get_allowed_tokens()

    # Before opening quote, only quote is allowed.
    assert allowed_tokens == {42}

    sm.current_text = '"'
    allowed_tokens = sm.get_allowed_tokens()

    # Once inside the string, regular tokens and closing quote are allowed.
    assert allowed_tokens == {10, 42, 99}


def test_string_value_rejects_fragments_that_need_json_escaping():
    """String value rejects fragments that need JSON escaping."""
    token_to_id = {'"': 10, "safe": 20, "\n": 30, "\\": 40}
    model = MappedFakeModel(token_to_id)
    funcs = StringParamFunctionsDef()
    sm = JSONStateMachine(
        cast(Small_LLM_Model, model),
        cast(FunctionsDefinition, funcs),
        token_to_id,
    )
    sm.state = JSONState.PARAM_VAL
    sm.current_function_name = "fn_echo"
    sm.current_text = '"'

    assert sm.get_allowed_tokens() == {10, 20}


def test_number_value_allows_only_terminators_after_precision_is_met():
    """Number value allows only terminators after precision is met."""
    token_to_id = {"1": 11, "2": 22, ",": 33, " ": 44, "}": 55, ".": 66}
    model = MappedFakeModel(token_to_id)
    funcs = NumberParamFunctionsDef()

    sm = JSONStateMachine(
        cast(Small_LLM_Model, model),
        cast(FunctionsDefinition, funcs), token_to_id, prompt="3.45")
    sm.state = JSONState.PARAM_VAL
    sm.current_function_name = "fn_add"
    sm.current_param_nb = 0
    sm.current_text = "3.45"

    allowed_tokens = sm.get_allowed_tokens()

    assert allowed_tokens == {33, 44, 55}


def test_integer_value_disallows_decimal_fragments_and_can_terminate():
    """Integer values remain integral and consume their terminator."""
    token_to_id = {"1": 11, "2": 22, ",": 33, "}": 44, ".": 55,
                   "e": 66}
    model = MappedFakeModel(token_to_id)
    funcs = IntegerParamFunctionsDef()
    sm = JSONStateMachine(
        cast(Small_LLM_Model, model),
        cast(FunctionsDefinition, funcs), token_to_id, prompt="Is 4 even?")
    sm.state = JSONState.PARAM_VAL
    sm.current_function_name = "fn_even"
    sm.current_param_nb = 0
    sm.current_text = "4"

    assert sm.get_allowed_tokens() == {33, 44}
    assert sm.update(44) is False
    assert sm.state == JSONState.END
    assert sm.current_text == ""


def test_integer_value_cannot_terminate_before_full_prompt_number():
    """A multi-digit integer must match its complete prompt literal."""
    token_to_id = {"2": 22, "3": 33, ",": 44, "}": 55}
    model = MappedFakeModel(token_to_id)
    funcs = IntegerParamFunctionsDef()
    sm = JSONStateMachine(
        cast(Small_LLM_Model, model),
        cast(FunctionsDefinition, funcs), token_to_id, prompt="for 23 years")
    sm.state = JSONState.PARAM_VAL
    sm.current_function_name = "fn_even"
    sm.current_param_nb = 0

    assert sm.get_allowed_tokens() == {22}
    sm.update(22)
    assert sm.get_allowed_tokens() == {33}
    sm.update(33)
    assert sm.get_allowed_tokens() == {44, 55}


def test_empty_parameter_function_generates_complete_json_suffix():
    """Empty parameter function generates complete JSON suffix."""
    model = FakeModel()
    funcs = EmptyParamFunctionsDef()
    token_to_id = {chr(i): i for i in range(32, 128)}
    sm = JSONStateMachine(
        cast(Small_LLM_Model, model),
        cast(FunctionsDefinition, funcs), token_to_id, prompt="nonsense")

    sm.state = JSONState.NAME_VAL
    for char in "fn_ping":
        sm.update(ord(char))

    assert sm.current_function_name == "fn_ping"
    boundary_token = next(iter(sm.get_allowed_tokens()))
    sm.update(boundary_token)
    assert sm.state == JSONState.EMPTY_PARAMS
    assert model.decode(sm.get_target_tokens_for_current_state()) == (
        ', "parameters": {}}'
    )


def test_function_name_prefix_can_continue_or_terminate():
    """Function name prefix can continue or terminate."""
    model = FakeModel()
    funcs = PrefixFunctionsDef()
    token_to_id = {chr(i): i for i in range(32, 128)}
    sm = JSONStateMachine(
        cast(Small_LLM_Model, model),
        cast(FunctionsDefinition, funcs),
        token_to_id,
    )
    sm.state = JSONState.NAME_VAL

    for char in "get":
        sm.update(ord(char))

    assert sm.state == JSONState.NAME_VAL
    assert ord("_") in sm.get_allowed_tokens()
    assert ord('"') in sm.get_allowed_tokens()

    for char in "_weather":
        sm.update(ord(char))
    boundary_id = ord('"')
    assert sm.get_allowed_tokens() == {boundary_id}

    sm.update(boundary_id)
    assert sm.current_function_name == "get_weather"
    assert sm.state == JSONState.EMPTY_PARAMS
