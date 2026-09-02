
from pydantic import ValidationError

from src.functions_manager import (
    FunctionSchema,
    FunctionsDefinition,
    Parameter,
)
from src.call_me_maybe import main
import pytest


def test_main_runs_without_error():
    """Main runs without error."""
    from src import call_me_maybe

    called = {}

    def fake_run_cli(functions_definition_path,
                     input_path=None, output_path=None):
        """Record command-line arguments without running generation."""
        called["args"] = (functions_definition_path, input_path, output_path)
        return []

    original_run_cli = call_me_maybe.run_cli
    call_me_maybe.run_cli = fake_run_cli
    try:
        assert main([]) == 0
    finally:
        call_me_maybe.run_cli = original_run_cli

    assert called["args"] == (
        "data/input/functions_definition.json",
        "data/input/function_calling_tests.json",
        "data/output/function_calling_results.json",
    )


def test_main_with_nonexistent_functions_definition():
    """Main with nonexistent functions definition."""
    return_code = main([
        "--functions_definition",
        "./tests/data/nonexistent_functions_definition.json",
    ])

    assert return_code == 1


def test_wrong_functions_definition():
    """Wrong functions definition."""
    with pytest.raises(ValueError) as exc:
        FunctionsDefinition.from_json(
            "tests/data/wrong_functions_definition.json")
    assert "Invalid function definition" in str(exc.value)


def test_empty_functions_definition():
    """Empty functions definition."""
    with pytest.raises(ValueError) as exc:
        FunctionsDefinition.from_json(
            "tests/data/empty_functions_definition.json")
    assert "Invalid JSON file" in str(exc.value)


def test_nonexistent_functions_definition():
    """Nonexistent functions definition."""
    with pytest.raises(ValueError) as exc:
        FunctionsDefinition.from_json(
            "tests/data/nonexistent_functions_definition.json")
    assert "File not found" in str(exc.value)


def test_valid_functions_definition():
    """Valid functions definition."""
    functions_def = FunctionsDefinition.from_json(
        "./tests/data/valid_functions_definition.json")
    assert len(functions_def.functions) == 5
    functions_names = ["fn_add_numbers",
                       "fn_greet",
                       "fn_reverse_string",
                       "fn_get_square_root",
                       "fn_substitute_string_with_regex"]
    assert functions_def.list_functions_name() == functions_names


def test_parameter_rejects_unsupported_type():
    """Parameter rejects unsupported type."""
    with pytest.raises(ValidationError):
        Parameter(type="array")  # type: ignore[arg-type]


def test_functions_definition_rejects_duplicate_names():
    """Functions definition rejects duplicate names."""
    duplicate = FunctionSchema(name="duplicate")

    with pytest.raises(ValueError, match="Function names must be unique"):
        FunctionsDefinition([duplicate, duplicate])


def test_get_function_by_name():
    """Get function by name."""
    functions_def = FunctionsDefinition.from_json(
        "./tests/data/valid_functions_definition.json")
    func = functions_def.get_function_by_name("fn_greet")
    assert func.name == "fn_greet"
    assert func.description ==\
        "Generate a greeting message for a person by name."
    assert func.parameters == {'name': Parameter(type='string')}


def test_get_function_by_name_not_found():
    """Get function by name not found."""
    functions_def = FunctionsDefinition.from_json(
        "./tests/data/valid_functions_definition.json")
    with pytest.raises(ValueError) as exc:
        functions_def.get_function_by_name("nonexistent_function")
    assert "Function with name 'nonexistent_function' not found" in str(
        exc.value)


def test_get_function_description_by_name():
    """Get function description by name."""
    functions_def = FunctionsDefinition.from_json(
        "./tests/data/valid_functions_definition.json")
    description = functions_def.get_function_description_by_name(
        "fn_reverse_string")
    assert description == "Reverse a string and return the reversed result."


def test_get_function_parameters_by_name():
    """Get function parameters by name."""
    functions_def = FunctionsDefinition.from_json(
        "./tests/data/valid_functions_definition.json")
    params = functions_def.get_function_parameters_by_name("fn_add_numbers")
    assert params == {'a': Parameter(type='number'),
                      'b': Parameter(type='number')}


def test_get_nb_parameters():
    """Get nb parameters."""
    functions_def = FunctionsDefinition.from_json(
        "./tests/data/valid_functions_definition.json")
    nb_params = functions_def.get_nb_parameters(
        "fn_substitute_string_with_regex")
    assert nb_params == 3


def test_get_nb_parameters_nonexistent_function():
    """Get nb parameters nonexistent function."""
    functions_def = FunctionsDefinition.from_json(
        "./tests/data/valid_functions_definition.json")
    with pytest.raises(ValueError) as exc:
        functions_def.get_nb_parameters("nonexistent_function")
    assert "Function with name 'nonexistent_function' not found" in str(
        exc.value)


def test_get_functions_prompt():
    """Get functions prompt."""
    functions_def = FunctionsDefinition.from_json(
        "./tests/data/valid_functions_definition.json")
    prompt = functions_def.get_functions_prompt()
    expected_prompt = (
        "Here are the available functions:\n\n"
        " - Function Name: fn_add_numbers\n"
    )
    assert expected_prompt in prompt
