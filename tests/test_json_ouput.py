from src.functions_manager import FunctionsDefinition
import pytest
from pydantic import ValidationError

good_f_str_json_output = {
    "prompt": "What is the sum of 2 and 3?",
    "name": "fn_add_numbers",
    "parameters": {
      "a": 2.0,
      "b": 3.0
    }
  }
bad_f_str_json_output = {
    "prompt": "What is the sum of 2 and 3?",
    "name": "fn_add_numbers",
    "parameters": {
      "a": "two",
      "b": "three"
    }
  }

good_str_f_json_output = {
    "prompt": "Greet john",
    "name": "fn_greet",
    "parameters": {
      "name": "john"
    }
  }
bad_str_f_json_output = {
    "prompt": "Greet john",
    "name": "fn_greet",
    "parameters": {
      "name": 123
    }
  }


def test_bad_json_f_str_output():
    """Bad JSON f str output."""
    functions_def = FunctionsDefinition.from_json(
        "./tests/data/valid_functions_definition.json")
    OutputModel = functions_def.get_output_function_model("fn_add_numbers")
    with pytest.raises(ValidationError) as exc:
        OutputModel.model_validate(bad_f_str_json_output)
    assert "Input should be a valid number" in str(exc.value)


def test_good_json_f_str_output():
    """Good JSON f str output."""
    functions_def = FunctionsDefinition.from_json(
        "./tests/data/valid_functions_definition.json")
    OutputModel = functions_def.get_output_function_model("fn_add_numbers")
    validated = OutputModel.model_validate(good_f_str_json_output)
    assert validated.model_dump()["parameters"]["a"] == 2.0
    assert validated.model_dump()["parameters"]["b"] == 3.0


def test_output_rejects_extra_keys():
    """Output rejects extra keys."""
    functions_def = FunctionsDefinition.from_json(
        "./tests/data/valid_functions_definition.json")
    OutputModel = functions_def.get_output_function_model("fn_add_numbers")
    output_with_extra_key = {
        **good_f_str_json_output,
        "error": "unexpected",
    }

    with pytest.raises(ValidationError):
        OutputModel.model_validate(output_with_extra_key)


def test_bad_json_str_f_output():
    """Bad JSON str f output."""
    functions_def = FunctionsDefinition.from_json(
        "./tests/data/valid_functions_definition.json")
    OutputModel = functions_def.get_output_function_model("fn_greet")
    with pytest.raises(ValidationError) as exc:
        OutputModel.model_validate(bad_str_f_json_output)
    assert "Input should be a valid string" in str(exc.value)


def test_good_json_str_f_output():
    """Good JSON str f output."""
    functions_def = FunctionsDefinition.from_json(
        "./tests/data/valid_functions_definition.json")
    OutputModel = functions_def.get_output_function_model("fn_greet")
    validated = OutputModel.model_validate(good_str_f_json_output)
    assert validated.model_dump()["parameters"]["name"] == "john"
