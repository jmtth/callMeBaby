import json

import pytest

from src.functions_manager import (
    FunctionSchema,
    FunctionsDefinition,
    Parameter,
)
from src.generator import (
    GenerationLimitError,
    GenerationStep,
    generate_constrained_response,
)
from src.models import JSONState


class CharacterModel:
    """Provide the CharacterModel test double."""

    def encode(self, text: str) -> list[list[int]]:
        """Encode text into token IDs for the test double."""
        return [[ord(character) for character in text]]

    def decode(self, ids: list[int]) -> str:
        """Decode token IDs into text for the test double."""
        return "".join(chr(token_id) for token_id in ids)

    def get_logits_from_input_ids(self, input_ids: list[int]) -> list[float]:
        """Return deterministic logits for the test double."""
        logits = [0.0] * 128
        logits[ord("t")] = 1.0
        return logits


def test_complete_boolean_generation_pipeline():
    """Complete boolean generation pipeline."""
    model = CharacterModel()
    token_to_id = {chr(token_id): token_id for token_id in range(32, 128)}
    functions = FunctionsDefinition([
        FunctionSchema(
            name="toggle",
            parameters={"enabled": Parameter(type="boolean")},
        )
    ])
    input_prompt = "Set enabled to true"

    response = generate_constrained_response(
        model,
        token_to_id,
        functions,
        prompt="Select a function",
        input_prompt=input_prompt,
        max_res_tokens=128,
    )

    assert json.loads(response) == {
        "prompt": input_prompt,
        "name": "toggle",
        "parameters": {"enabled": True},
    }


def test_generation_observer_receives_decisions_and_final_json():
    """Generation observer receives decisions and final json."""
    model = CharacterModel()
    token_to_id = {chr(token_id): token_id for token_id in range(32, 128)}
    functions = FunctionsDefinition([
        FunctionSchema(
            name="toggle",
            parameters={"enabled": Parameter(type="boolean")},
        )
    ])
    events: list[GenerationStep] = []

    response = generate_constrained_response(
        model,
        token_to_id,
        functions,
        prompt="Select a function",
        input_prompt="Set enabled to true",
        max_res_tokens=128,
        observer=events.append,
    )

    assert events
    assert events[-1].generated_text == response
    assert {event.kind for event in events} >= {"fixed", "model"}
    model_events = [event for event in events if event.kind == "model"]
    assert model_events
    assert model_events[0].candidates
    assert model_events[0].candidates[0].token_id == (
        model_events[0].selected_id
    )


def test_generation_stops_when_selected_function_lacks_prompt_values():
    """Generation stops when selected function lacks prompt values."""
    model = CharacterModel()
    token_to_id = {chr(token_id): token_id for token_id in range(32, 128)}
    functions = FunctionsDefinition([
        FunctionSchema(
            name="add",
            parameters={
                "a": Parameter(type="number"),
                "b": Parameter(type="number"),
            },
        )
    ])

    with pytest.raises(
        ValueError,
        match=r"'add'.*2 number parameter\(s\) required, 1 found",
    ):
        generate_constrained_response(
            model,
            token_to_id,
            functions,
            prompt="Select a function",
            input_prompt="add 3",
            max_res_tokens=128,
        )


class StuckStateMachine:
    """Provide the StuckStateMachine test double."""

    def __init__(self, *args, **kwargs):
        """Initialize the test double."""
        self.state = JSONState.NAME_VAL

    def is_in_fixed_sequence(self) -> bool:
        """Return whether the test double is in a fixed sequence."""
        return False

    def get_allowed_tokens(self) -> set[int]:
        """Return token IDs allowed by the test double."""
        return {ord("a")}

    def update(self, token_id: int) -> bool:
        """Record a token update in the test double."""
        return False


def test_generation_has_a_step_guard_even_when_no_token_is_kept():
    """Generation has a step guard even when no token is kept."""
    model = CharacterModel()

    with pytest.raises(GenerationLimitError, match="did not converge"):
        generate_constrained_response(
            model,
            {"a": ord("a")},
            FunctionsDefinition([]),
            prompt="test",
            input_prompt="test",
            max_res_tokens=1,
            fsm_factory=StuckStateMachine,
        )
