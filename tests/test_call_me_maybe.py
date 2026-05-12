import src.call_me_maybe as cmm
from unittest.mock import MagicMock, patch, mock_open
import pytest
import json


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


def test_next_token_selection():
    """Test that the next token selection logic
    correctly handles end of sequence."""
    fake_model = MagicMock()
    fake_model.get_logits_from_input_ids.return_value = [0.1, 0.2, 0.3, 0.4]
    fake_allowed_ids = {2, 3}
    fake_current_ids = [0, 1, 3]
    new_token_id = cmm.next_token_selection(
        fake_model, fake_current_ids, fake_allowed_ids)
    assert new_token_id == 3


def test_next_token_selection_no_allowed_tokens():
    """Test that the next token selection logic
    correctly handles no allowed tokens."""
    fake_model = MagicMock()
    fake_model.get_logits_from_input_ids.return_value = [0.1, 0.2, 0.3, 0.4]
    fake_allowed_ids = set()
    fake_current_ids = [0, 1, 3]
    with pytest.raises(ValueError,
                       match="No allowed tokens available for selection"):
        cmm.next_token_selection(
            fake_model, fake_current_ids, fake_allowed_ids)

def test_load_model():
    """Test that the load_model function returns an object
    with the expected methods."""
    model = cmm.load_model()
    assert hasattr(model[0], "encode")
    assert callable(model[0].encode)
    assert hasattr(model[0], "decode")
    assert callable(model[0].decode)
    assert hasattr(model[0], "get_logits_from_input_ids")
    assert callable(model[0].get_logits_from_input_ids)

@patch("src.call_me_maybe.Small_LLM_Model")
@patch("src.call_me_maybe.platform.system")
def test_load_model_linux(mock_system, mock_model_class):
    """Test load_model on Linux uses cpu device."""
    # Simulate Linux
    mock_system.return_value = "Linux"

    # Simulate model
    mock_model = MagicMock()
    mock_model.get_path_to_vocab_file.return_value = "/fake/vocab.json"
    mock_model_class.return_value = mock_model

    # Simulate vocab file
    fake_vocab = {"hello": 0, "world": 1}
    with patch("builtins.open", mock_open(read_data=json.dumps(fake_vocab))):
        model, token_to_id = cmm.load_model()

    # Verify device is cpu on Linux
    mock_model_class.assert_called_with(device="cpu",
                                        cache_dir="./.hf_cache",
                                        local_files_only=True)
    assert token_to_id == {"hello": 0, "world": 1}


@patch("src.call_me_maybe.Small_LLM_Model")
@patch("src.call_me_maybe.platform.system")
def test_load_model_mac(mock_system, mock_model_class):
    """Test load_model on Mac uses mps device."""
    # Simulate Mac
    mock_system.return_value = "Darwin"

    mock_model = MagicMock()
    mock_model.get_path_to_vocab_file.return_value = "/fake/vocab.json"
    mock_model_class.return_value = mock_model

    fake_vocab = {"hello": 0, "world": 1}
    with patch("builtins.open", mock_open(read_data=json.dumps(fake_vocab))):
        model, token_to_id = cmm.load_model()

    # Verify device is mps on Mac
    mock_model_class.assert_called_with(device="mps",
                                        cache_dir="./.hf_cache",
                                        local_files_only=True)


@patch("src.call_me_maybe.Small_LLM_Model")
@patch("src.call_me_maybe.platform.system")
def test_load_model_fallback(mock_system, mock_model_class):
    """Test load_model fallback when local files not found."""
    mock_system.return_value = "Linux"

    mock_model = MagicMock()
    mock_model.get_path_to_vocab_file.return_value = "/fake/vocab.json"

    # First call raises exception, second call succeeds
    mock_model_class.side_effect = [Exception("No local files"), mock_model]

    fake_vocab = {"hello": 0, "world": 1}
    with patch("builtins.open", mock_open(read_data=json.dumps(fake_vocab))):
        model, token_to_id = cmm.load_model()

    # Verify fallback call with local_files_only=False
    assert mock_model_class.call_count == 2
    mock_model_class.assert_called_with(device="cpu",
                                        cache_dir="./.hf_cache",
                                        local_files_only=False)
