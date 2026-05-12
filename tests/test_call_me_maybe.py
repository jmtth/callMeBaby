import src.call_me_maybe as cmm
from unittest.mock import MagicMock
import pytest

def test_build_prompt_includes_functions():
    fakeFunctionsDefinition = MagicMock()
    fakeFunctionsDefinition.get_functions_prompt.return_value = (
        "Here are the available functions:\n\n"
        " - Function Name: add\n"
        "   Description: Add two numbers\n"
    )

    result = cmm.build_prompt(fakeFunctionsDefinition, "What is 1+1?")

    assert "You are a assistant that helps with function calls.\n\n" in result
    assert "Here are the available functions:" in result
    assert "What is 1+1?" in result
    assert "add" in result
    assert "Add two numbers" in result


def test_build_token_to_id_shapeA():
    """Test the build_token_to_id function with a sample vocab."""

    vocab_shape = {"0": "hello", "1": "world", "2": "Ġhello", "3": "Ġworld"}
    token_to_id = cmm.build_token_to_id(vocab_shape)
    assert token_to_id == {"hello": 0, "world": 1, "Ġhello": 2, "Ġworld": 3}


def test_build_token_to_id_shapeB():
    """Test the build_token_to_id function with a sample vocab."""

    vocab_shape = {"hello": 0, "world": 1, "Ġhello": 2, "Ġworld": 3}
    token_to_id = cmm.build_token_to_id(vocab_shape)
    assert token_to_id == {"hello": 0, "world": 1, "Ġhello": 2, "Ġworld": 3}


def test_build_token_to_id_raises_on_empty():
    """Empty vocab should raise ValueError."""
    with pytest.raises(ValueError,
                       match="Vocabulary is empty, cannot build token_to_id"):
        cmm.build_token_to_id({})


def test_build_token_to_id_raises_on_bad_keys():
    """Non-convertible keys should raise ValueError."""
    with pytest.raises(ValueError, match="Unsupported vocab format"):
        cmm.build_token_to_id({"hello": None, "world": None})


def test_build_token_to_id_raises_on_bad_values():
    """Non-convertible values should raise ValueError."""
    with pytest.raises(ValueError, match="Unsupported vocab format"):
        cmm.build_token_to_id({"0": None, "1": None})


def test_build_token_to_id_raises_on_unicode_digits():
    """Unicode digits like ² pass isdigit() but fail int()."""
    with pytest.raises(ValueError, match="invalid literal for int()"):
        cmm.build_token_to_id({"²": "hello", "³": "world"})
