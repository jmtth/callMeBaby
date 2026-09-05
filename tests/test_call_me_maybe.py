import src.call_me_maybe as cmm
from src.grounding import collect_prompt_string_candidates
from src.models import JSONState
from unittest.mock import MagicMock, patch, mock_open
import pytest
import json

from src.generator import GenerationBuffer, GenerationLimitError
from src.functions_manager import FunctionSchema, Parameter


def test_build_prompt_includes_functions():
    """Build prompt includes functions."""
    fakeFunctionsDefinition = MagicMock()
    fakeFunctionsDefinition.get_functions_prompt.return_value = (
        "Here are the available functions:\n\n"
        " - Function Name: add\n"
        "   Description: Add two numbers\n"
    )

    result = cmm.build_prompt(fakeFunctionsDefinition, "What is 1+1?")

    assert "You are a function calling router" in result
    assert "Here are the available functions:" in result
    assert "What is 1+1?" in result
    assert "add" in result
    assert "Add two numbers" in result


def test_build_prompt_can_show_only_the_committed_function():
    """Build prompt delegates selected-function filtering."""
    functions_def = MagicMock()
    functions_def.get_functions_prompt.return_value = "- fn_read: Read a file"

    result = cmm.build_prompt(functions_def, "Read /tmp/a", "fn_read")

    functions_def.get_functions_prompt.assert_called_once_with("fn_read")
    assert "- fn_read: Read a file" in result


def test_build_token_to_id_shapeA():
    """Build token to id shapea."""

    vocab_shape = {"0": "hello", "1": "world", "2": "Ġhello", "3": "Ġworld"}
    token_to_id = cmm.build_token_to_id(vocab_shape)
    assert token_to_id == {"hello": 0, "world": 1, "Ġhello": 2, "Ġworld": 3}


def test_build_token_to_id_shapeB():
    """Build token to id shapeb."""

    vocab_shape = {"hello": 0, "world": 1, "Ġhello": 2, "Ġworld": 3}
    token_to_id = cmm.build_token_to_id(vocab_shape)
    assert token_to_id == {"hello": 0, "world": 1, "Ġhello": 2, "Ġworld": 3}


def test_build_token_to_id_raises_on_empty():
    """Build token to id raises on empty."""
    with pytest.raises(ValueError,
                       match="Vocabulary is empty, cannot build token_to_id"):
        cmm.build_token_to_id({})


def test_build_token_to_id_raises_on_bad_keys():
    """Build token to id raises on bad keys."""
    with pytest.raises(ValueError, match="Unsupported vocab format"):
        cmm.build_token_to_id({"hello": None, "world": None})


def test_build_token_to_id_raises_on_bad_values():
    """Build token to id raises on bad values."""
    with pytest.raises(ValueError, match="Unsupported vocab format"):
        cmm.build_token_to_id({"0": None, "1": None})


def test_build_token_to_id_raises_on_unicode_digits():
    """Build token to id raises on unicode digits."""
    with pytest.raises(ValueError, match="invalid literal for int()"):
        cmm.build_token_to_id({"²": "hello", "³": "world"})


def test_select_next_token():
    """Next token selection."""
    fake_model = MagicMock()
    fake_model.get_logits_from_input_ids.return_value = [0.1, 0.2, 0.3, 0.4]
    fake_allowed_ids = {2, 3}
    fake_current_ids = [0, 1, 3]
    new_token_id = cmm.select_next_token(
        fake_model, fake_current_ids, fake_allowed_ids)
    assert new_token_id == 3


def test_select_next_token_no_allowed_tokens():
    """Next token selection no allowed tokens."""
    fake_model = MagicMock()
    fake_model.get_logits_from_input_ids.return_value = [0.1, 0.2, 0.3, 0.4]
    fake_allowed_ids = set()
    fake_current_ids = [0, 1, 3]
    with pytest.raises(ValueError,
                       match="No allowed tokens available for selection"):
        cmm.select_next_token(
            fake_model, fake_current_ids, fake_allowed_ids)


def test_generation_buffer_enforces_an_atomic_token_budget():
    """Generation buffer enforces an atomic token budget."""
    buffer = GenerationBuffer(prompt_ids=[1, 2], max_response_tokens=2)
    buffer.append([3, 4])

    with pytest.raises(GenerationLimitError, match="max_res_tokens=2"):
        buffer.append(5)

    assert buffer.response_ids == [3, 4]
    assert buffer.context_ids == [1, 2, 3, 4]


def test_grounded_numeric_parameters_accept_prompt_values():
    """Grounded numeric parameters accept prompt values."""
    functions_def = cmm.FunctionsDefinition.from_json(
        "tests/data/valid_functions_definition.json"
    )

    cmm.validate_grounded_parameters(
        functions_def,
        "What is the sum of 34 and 2?",
        "fn_add_numbers",
        {"a": 34.0, "b": 2.0},
    )


def test_grounded_numeric_parameters_reject_invented_value():
    """Grounded numeric parameters reject invented value."""
    functions_def = cmm.FunctionsDefinition.from_json(
        "tests/data/valid_functions_definition.json"
    )

    with pytest.raises(ValueError, match="was not found in the user prompt"):
        cmm.validate_grounded_parameters(
            functions_def,
            "What is the sum of 34 and toto?",
            "fn_add_numbers",
            {"a": 34.0, "b": 3.0},
        )


def test_grounded_integer_parameter_rejects_invented_value():
    """Grounded integer parameters must occur in the prompt."""
    functions_def = cmm.FunctionsDefinition([
        FunctionSchema(
            name="even",
            parameters={"value": Parameter(type="integer")},
        )
    ])

    with pytest.raises(ValueError, match="was not found in the user prompt"):
        cmm.validate_grounded_parameters(
            functions_def,
            "Is 4 even?",
            "even",
            {"value": 7},
        )


def test_grounded_boolean_detects_literal_before_punctuation():
    """Grounded boolean detects literal before punctuation."""
    functions_def = cmm.FunctionsDefinition([
        FunctionSchema(
            name="toggle",
            description="Toggle a setting.",
            parameters={"enabled": Parameter(type="boolean")},
        )
    ])

    cmm.validate_grounded_parameters(
        functions_def,
        "Set enabled to true.",
        "toggle",
        {"enabled": True},
    )


def test_string_parameter_rejects_embedded_response_structure():
    """String parameters reject leaked response-object fields."""
    functions_def = cmm.FunctionsDefinition([
        FunctionSchema(
            name="echo",
            parameters={"text": Parameter(type="string")},
        )
    ])

    with pytest.raises(ValueError, match="response-structure fragment"):
        cmm.validate_grounded_parameters(
            functions_def,
            "Echo hello",
            "echo",
            {"text": "hello {prompt: 'Echo hello'"},
        )


def test_string_parameter_rejects_invented_field_word_and_placeholders():
    """String parameters reject metadata and placeholders not in the prompt."""
    functions_def = cmm.FunctionsDefinition([
        FunctionSchema(
            name="format",
            parameters={"template": Parameter(type="string")},
        )
    ])
    generated = (
        "Say hello to {name} with the prompt: {prompt} "
        "and {name} is the name of the {id}"
    )

    with pytest.raises(ValueError, match="response-structure fragment"):
        cmm.validate_grounded_parameters(
            functions_def,
            'Format template: Say "hello" to {name}',
            "format",
            {"template": generated},
        )


def test_string_parameter_allows_plain_structure_words():
    """Plain words matching output-field names remain valid string values."""
    functions_def = cmm.FunctionsDefinition([
        FunctionSchema(
            name="echo",
            parameters={"text": Parameter(type="string")},
        )
    ])

    cmm.validate_grounded_parameters(
        functions_def,
        "Echo the name and parameters",
        "echo",
        {"text": "the name and parameters"},
    )


def test_string_parameter_rejects_value_absent_from_prompt():
    """Every grounded string must be an exact source span."""
    functions_def = cmm.FunctionsDefinition([
        FunctionSchema(
            name="read",
            parameters={"path": Parameter(type="string")},
        )
    ])

    with pytest.raises(ValueError, match="was not extracted exactly"):
        cmm.validate_grounded_parameters(
            functions_def,
            "Read the file at /home/user/data.json",
            "read",
            {"path": "generated/path.json"},
        )


def test_string_parameter_accepts_exact_source_span():
    """Generic grounding accepts an exact string from the prompt."""
    functions_def = cmm.FunctionsDefinition([
        FunctionSchema(
            name="read",
            parameters={"path": Parameter(type="string")},
        )
    ])

    cmm.validate_grounded_parameters(
        functions_def,
        "Read the file at /home/user/data.json",
        "read",
        {"path": "/home/user/data.json"},
    )


def test_string_candidates_preserve_windows_path_verbatim():
    """Candidate extraction never normalizes Windows separators."""
    prompt = r"Read C:\Users\john\config.ini with latin-1 encoding"

    candidates = collect_prompt_string_candidates(prompt, "path")

    assert r"C:\Users\john\config.ini" in candidates
    assert "C:/Users/john/config.ini" not in candidates


def test_string_candidates_use_parameter_label_before_colon():
    """A schema label followed by a colon identifies the complete tail."""
    prompt = 'Format template: Say "hello" to {name}'

    assert collect_prompt_string_candidates(prompt, "template") == (
        'Say "hello" to {name}',
    )


def test_unknown_string_form_retains_generative_fallback():
    """Derived strings remain possible when no exact source is inferable."""
    functions_def = cmm.FunctionsDefinition([
        FunctionSchema(
            name="replace",
            parameters={"replacement": Parameter(type="string")},
        )
    ])

    assert collect_prompt_string_candidates(
        "Replace all vowels with asterisks",
        "replacement",
    ) == ()
    cmm.validate_grounded_parameters(
        functions_def,
        "Replace all vowels with asterisks",
        "replace",
        {"replacement": "*"},
    )


@patch("src.call_me_maybe.Small_LLM_Model")
def test_load_model(mock_model_class):
    """Load model."""
    mock_model = MagicMock()
    mock_model.get_path_to_vocab_file.return_value = "/fake/vocab.json"
    mock_model_class.return_value = mock_model

    fake_vocab = {"hello": 0, "world": 1}
    with patch("builtins.open", mock_open(read_data=json.dumps(fake_vocab))):
        model = cmm.load_model()

    assert hasattr(model[0], "encode")
    assert callable(model[0].encode)
    assert hasattr(model[0], "decode")
    assert callable(model[0].decode)
    assert hasattr(model[0], "get_logits_from_input_ids")
    assert callable(model[0].get_logits_from_input_ids)


@patch("src.call_me_maybe.Small_LLM_Model")
@patch("platform.system")
def test_load_model_linux(mock_system, mock_model_class):
    """Load model linux."""
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
    mock_model_class.assert_called_once_with(device="cpu")
    assert token_to_id == {"hello": 0, "world": 1}


@patch("src.call_me_maybe.Small_LLM_Model")
@patch("platform.system")
def test_load_model_mac(mock_system, mock_model_class):
    """Load model mac."""
    # Simulate Mac
    mock_system.return_value = "Darwin"

    mock_model = MagicMock()
    mock_model.get_path_to_vocab_file.return_value = "/fake/vocab.json"
    mock_model_class.return_value = mock_model

    fake_vocab = {"hello": 0, "world": 1}
    with patch("builtins.open", mock_open(read_data=json.dumps(fake_vocab))):
        model, token_to_id = cmm.load_model()

    # Verify device is mps on Mac
    mock_model_class.assert_called_once_with(device="mps")


@patch("src.call_me_maybe.Small_LLM_Model")
@patch("platform.system")
def test_load_model_fallback(mock_system, mock_model_class):
    """Load model fallback."""
    mock_system.return_value = "Linux"
    mock_model_class.side_effect = Exception("No local files")

    with pytest.raises(SystemExit) as exc:
        cmm.load_model()

    assert exc.value.code == 1
    mock_model_class.assert_called_once_with(device="cpu")


def make_fake_llm():
    """Return a fake model and token-to-ID mapping."""
    fake_model = MagicMock()

    # encode(prompt)[0].tolist() -> [1, 2, 3]
    fake_tensor = MagicMock()
    fake_tensor.tolist.return_value = [1, 2, 3]
    fake_model.encode.return_value = [fake_tensor]  # encode()[0] = fake_tensor

    fake_model.decode.return_value = '{"name": "fn_add_numbers"}'

    fake_token_to_id = {"hello": 0, "world": 1}

    return (fake_model, fake_token_to_id)


def make_fake_fsm(states: list):
    """Return a configurable JSON state-machine test double."""
    fake_fsm = MagicMock()
    fake_fsm.is_in_fixed_sequence.return_value = False
    fake_fsm.get_allowed_tokens.return_value = {1, 2, 3}
    fake_fsm.update.return_value = True
    fake_fsm.param_repeat_pattern = None

    # Simule les états successifs
    fake_fsm.state = MagicMock()
    type(fake_fsm).state = MagicMock(side_effect=states)

    return fake_fsm


@patch("src.call_me_maybe.JSONStateMachine")
def test_generate_response_basic(mock_fsm_class):
    """Generate response basic."""
    fake_llm = make_fake_llm()
    fake_functions_def = MagicMock()
    fake_functions_def.get_functions_prompt.return_value = "functions prompt"

    # FSM termine immédiatement sur END
    fake_fsm = MagicMock()
    fake_fsm.state = JSONState.STOP
    mock_fsm_class.return_value = fake_fsm

    result = cmm.generate_response(
        fake_functions_def,
        "What is 1+1?",
        llm=fake_llm
    )

    assert isinstance(result, str)
    fake_llm[0].encode.assert_called_once()


@patch("src.call_me_maybe.JSONStateMachine")
def test_generate_response_fixed_sequence(mock_fsm_class):
    """Generate response fixed sequence."""
    fake_llm = make_fake_llm()
    fake_llm[0].decode.return_value = '{"name":'
    fake_functions_def = MagicMock()

    fake_fsm = MagicMock()
    fake_fsm.state = JSONState.START
    fake_fsm.is_in_fixed_sequence.return_value = True
    fake_fsm.get_target_tokens_for_current_state.return_value = [10, 11]

    def finish_after_target(token_id):
        """Stop the fake state machine after its fixed target."""
        if token_id == 11:
            fake_fsm.state = JSONState.STOP
        return True

    fake_fsm.update.side_effect = finish_after_target
    mock_fsm_class.return_value = fake_fsm

    result = cmm.generate_response(
        fake_functions_def,
        "What is 1+1?",
        llm=fake_llm
    )

    fake_fsm.get_target_tokens_for_current_state.assert_called_once()
    assert isinstance(result, str)


@patch("src.call_me_maybe.JSONStateMachine")
def test_generate_response_rejects_fixed_sequence_over_budget(mock_fsm_class):
    """Generate response rejects fixed sequence over budget."""
    fake_llm = make_fake_llm()
    fake_functions_def = MagicMock()
    fake_fsm = MagicMock()
    fake_fsm.state = JSONState.START
    fake_fsm.is_in_fixed_sequence.return_value = True
    fake_fsm.get_target_tokens_for_current_state.return_value = [10, 11]
    mock_fsm_class.return_value = fake_fsm

    with pytest.raises(GenerationLimitError, match="max_res_tokens=1"):
        cmm.generate_response(
            fake_functions_def,
            "test",
            llm=fake_llm,
            max_res_tokens=1,
        )

    fake_fsm.update.assert_not_called()


@patch("src.call_me_maybe.select_next_token")
@patch("src.call_me_maybe.JSONStateMachine")
def test_generate_response_forces_single_allowed_token(mock_fsm_class,
                                                       mock_next_token):
    """Generate response forces single allowed token."""
    fake_llm = make_fake_llm()
    fake_functions_def = MagicMock()

    fake_fsm = MagicMock()
    fake_fsm.state = JSONState.NAME_VAL
    fake_fsm.is_in_fixed_sequence.return_value = False
    fake_fsm.get_allowed_tokens.return_value = {2}

    def finish_generation(token_id):
        """Stop the fake state machine after token selection."""
        fake_fsm.state = JSONState.STOP
        return True

    fake_fsm.update.side_effect = finish_generation
    mock_fsm_class.return_value = fake_fsm

    cmm.generate_response(fake_functions_def, "test", llm=fake_llm)

    mock_next_token.assert_not_called()
    fake_fsm.update.assert_called_once_with(2)


@patch("src.call_me_maybe.load_model")
@patch("src.call_me_maybe.JSONStateMachine")
def test_generate_response_loads_model_if_none(mock_fsm_class,
                                               mock_load_model):
    """Generate response loads model if none."""
    fake_llm = make_fake_llm()
    mock_load_model.return_value = fake_llm
    fake_functions_def = MagicMock()

    fake_fsm = MagicMock()
    fake_fsm.state = JSONState.STOP
    mock_fsm_class.return_value = fake_fsm

    cmm.generate_response(fake_functions_def, "What is 1+1?")

    mock_load_model.assert_called_once()
