
from src.functions_manager import FunctionsDefinition
from src.call_me_maybe import main
import pytest


def test_main_runs_without_error():
    from src import call_me_maybe

    called = {}

    def fake_run_cli(functions_definition_path,
                     input_path=None, output_path=None):
        called["args"] = (functions_definition_path, input_path, output_path)
        return []

    original_run_cli = call_me_maybe.run_cli
    call_me_maybe.run_cli = fake_run_cli
    try:
        assert main([]) == 0
    finally:
        call_me_maybe.run_cli = original_run_cli

    assert called["args"] == (
        "data/input/functions_definition.json", None, None)


def test_main_with_nonexistent_functions_definition():
    with pytest.raises(ValueError) as exc:
        main(["--functions_definition",
              "./tests/data/nonexistent_functions_definition.json"])
    assert "File not found" in str(exc.value)


def test_wrong_functions_definition():
    with pytest.raises(ValueError) as exc:
        FunctionsDefinition.from_json(
            "tests/data/wrong_functions_definition.json")
    assert "Invalid function definition" in str(exc.value)


def test_empty_functions_definition():
    with pytest.raises(ValueError) as exc:
        FunctionsDefinition.from_json(
            "tests/data/empty_functions_definition.json")
    assert "Invalid JSON file" in str(exc.value)


def test_nonexistent_functions_definition():
    with pytest.raises(ValueError) as exc:
        FunctionsDefinition.from_json(
            "tests/data/nonexistent_functions_definition.json")
    assert "File not found" in str(exc.value)


def test_valid_functions_definition():
    functions_def = FunctionsDefinition.from_json(
        "./tests/data/valid_functions_definition.json")
    assert len(functions_def.functions) == 5
    functions_names = ["fn_add_numbers",
                       "fn_greet",
                       "fn_reverse_string",
                       "fn_get_square_root",
                       "fn_substitute_string_with_regex"]
    assert functions_def.list_functions_name() == functions_names
