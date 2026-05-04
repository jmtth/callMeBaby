from src.JSONStateMachine import JSONStateMachine
from src.models import JSONState


class DummyFunctionsDef:
    def list_functions_name(self):
        return []


class DummyParam:
    def __init__(self, type_: str):
        self.type = type_


class StringParamFunctionsDef:
    def list_functions_name(self):
        return ["fn_echo"]

    def get_function_parameters_by_name(self, name: str):
        return {"text": DummyParam("string")}


class NumberParamFunctionsDef:
    def list_functions_name(self):
        return ["fn_add"]

    def get_function_parameters_by_name(self, name: str):
        return {"value": DummyParam("number")}


class FakeModel:
    def encode(self, s: str):
        # return a list-like structure where [0] is a list of ints
        return [[ord(c) for c in s]]

    def decode(self, ids: list[int]) -> str:
        return ''.join(chr(i) for i in ids)


def test_extract_decimal_counts():
    model = FakeModel()
    funcs = DummyFunctionsDef()
    token_to_id = {chr(i): i for i in range(32, 128)}

    # prompt with integer, float with 3 decimals, float with 1 decimal
    prompt = "Value A: 12, Value B: 3.456, Value C: -7.0"
    sm = JSONStateMachine(model, funcs, token_to_id, prompt=prompt)

    # Expect: 12 -> 1 decimal, 3.456 -> 3, -7.0 -> 1
    assert sm.prompt_decimal_counts == [1, 3, 1]


def test_allowed_tokens_for_string_value_uses_actual_token_ids():
    model = FakeModel()
    funcs = StringParamFunctionsDef()
    token_to_id = {"x": 10, '"': 42, "y": 99}

    sm = JSONStateMachine(model, funcs, token_to_id, prompt="")
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


def test_number_value_allows_digits_and_terminators_after_precision_is_met():
    model = FakeModel()
    funcs = NumberParamFunctionsDef()
    token_to_id = {"1": 11, "2": 22, ",": 33, " ": 44, "}": 55, ".": 66}

    sm = JSONStateMachine(model, funcs, token_to_id, prompt="3.45")
    sm.state = JSONState.PARAM_VAL
    sm.current_function_name = "fn_add"
    sm.current_param_nb = 0
    sm.current_text = "3.45"

    allowed_tokens = sm.get_allowed_tokens()

    assert allowed_tokens == {33, 44, 55}
