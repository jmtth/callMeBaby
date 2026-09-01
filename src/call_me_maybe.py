from __future__ import annotations
import argparse
from decimal import Decimal, InvalidOperation
import json
import sys
from pathlib import Path
from pydantic import ValidationError


from llm_sdk import Small_LLM_Model, logging

from src.JSONStateMachine import JSONStateMachine
from src.functions_manager import FunctionsDefinition
from src.generator import (
    GenerationObserver,
    generate_constrained_response,
    select_next_token,
)
from src.grounding import collect_prompt_values
from src.token_vocabulary import TokenVocabulary
import timeit


def build_prompt(functions_def: FunctionsDefinition, prompt: str) -> str:
    """Start with a base instruction, then append the functions definition,
    and finally the user prompt.

    Args:
        functions_def (FunctionsDefinition): The definitions of the functions
        to include in the prompt.
        prompt (str): The user prompt to which the model should respond.

    Returns:
        str: The complete prompt to send to the model,
        including instructions, function definitions, and the user prompt.
    """
    new_prompt = (
        "Select exactly one of the available functions whose name and "
        "description best match the user request. Extract a value for every "
        "required parameter according to its declared type.\n\n"
    )
    new_prompt += functions_def.get_functions_prompt()
    new_prompt += "Now, answer the following question:\n"
    new_prompt += prompt
    return new_prompt


def build_token_to_id(vocab: dict) -> dict[str, int]:
    """
    Convert a common vocab JSON shape.
    Verify the shape of the vocab and convert it
    to a consistent token->id mapping.

    Args:
        vocab (dict): The vocabulary mapping, which can be in one of two shapes

    Returns:
        dict[str, int]: A consistent mapping of token strings
        to their corresponding IDs.
    """
    if not vocab:
        raise ValueError("Vocabulary is empty, cannot build token_to_id")
    # Shape A: {"0": "!", "1": "the", ...}
    if all(isinstance(k, str) and k.isdecimal() and isinstance(v, str)
           for k, v in vocab.items()):
        return {v: int(k) for k, v in vocab.items()}

    # Shape B: {"!": 0, "the": 1, ...}
    if all(isinstance(k, str) and isinstance(v, (int, str))
           for k, v in vocab.items()):
        return {k: int(v) for k, v in vocab.items()}

    raise ValueError("Unsupported vocab format for conversion")


next_token_selection = select_next_token


def load_model() -> tuple[Small_LLM_Model, dict[str, int]]:
    """Load the small LLM model.

    Args:
        cache_dir: Directory to use for caching model files.

    Returns:
        Tuple of (model, token_to_id mapping).
    """
    logging.disable_progress_bar()
    system = __import__("platform").system().lower()
    if system == 'linux':
        device = "cpu"
    else:
        device = "mps"
    try:
        model = Small_LLM_Model(device=device)
    except Exception as exc:
        print(f"Error loading model: {exc}.")
        sys.exit(1)

    vocab_path = model.get_path_to_vocab_file()
    with open(vocab_path, encoding="utf-8") as f:
        vocab = json.load(f)
    token_to_id = build_token_to_id(vocab)
    return (model, token_to_id)


def load_prompts(input_path: str | None) -> list[str]:
    """Load prompts from input path or stdin.

    If input_path is provided, it must be valid JSON containing prompts.
    Raises ValueError if file is missing or JSON is invalid.

    Args:
        input_path: Path to JSON file with prompts, or None for stdin.

    Returns:
        List of prompt strings.

    Raises:
        ValueError: If file not found or JSON invalid.
    """
    if input_path is None:
        return [input("input_prompt:")]

    try:
        raw_text = Path(input_path).read_text(encoding="utf-8").strip()
        if not raw_text:
            raise ValueError(f"Input file is empty: {input_path}")
    except FileNotFoundError as exc:
        raise ValueError(f"Input file not found: {input_path}") from exc

    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {input_path}: {exc}") from exc
    # prompt like "What is the sum of 2 and 3?"
    if isinstance(data, str):
        return [data]
    # prompt like {"prompt": "What is the sum of 2 and 3?"}
    if isinstance(data, dict):
        if "prompt" in data:
            return [str(data["prompt"])]
        if "prompts" in data and isinstance(data["prompts"], list):
            return [str(item) for item in data["prompts"]]
        raise ValueError(
            f"JSON dict must contain 'prompt' or 'prompts' key: {input_path}")
    # prompt like [{"prompt": "What is the sum of 2 and 3?"}, ...]
    if isinstance(data, list):
        prompts: list[str] = []
        for item in data:
            if isinstance(item, dict) and "prompt" in item:
                prompts.append(str(item["prompt"]))
            elif isinstance(item, str):
                prompts.append(item)
            else:
                raise ValueError(
                    f"List items must be strings or dicts \
                        with 'prompt' key: {input_path}")
        return prompts

    raise ValueError(f"JSON must be string, dict, or list: {input_path}")


def validate_grounded_parameters(functions_def: FunctionsDefinition,
                                 prompt: str,
                                 function_name: str,
                                 parameters: dict[str, object]) -> None:
    """Reject typed values that were invented instead of extracted.

    Function selection remains an LLM decision. This validation only checks
    that generated number and boolean arguments have explicit source values in
    the user prompt.

    Args:
        functions_def: Definitions loaded from the supplied JSON file.
        prompt: Original user request.
        function_name: Function selected by the LLM.
        parameters: Validated generated parameter values.

    Raises:
        ValueError: If a number or boolean value is absent from the prompt.
    """
    schema = functions_def.get_function_parameters_by_name(function_name)
    available = collect_prompt_values(prompt)
    available_numbers = list(available.numbers)
    available_booleans = list(available.booleans)

    for parameter_name, parameter_schema in schema.items():
        value = parameters[parameter_name]
        if parameter_schema.type == "number":
            try:
                number = Decimal(str(value))
            except InvalidOperation as exc:
                raise ValueError(
                    f"Parameter {parameter_name!r} is not a valid number"
                ) from exc
            if number not in available_numbers:
                raise ValueError(
                    f"Numeric parameter {parameter_name!r}={value!r} "
                    "was not found in the user prompt"
                )
            available_numbers.remove(number)
        elif parameter_schema.type == "boolean":
            boolean = bool(value)
            if boolean not in available_booleans:
                raise ValueError(
                    f"Boolean parameter {parameter_name!r}={value!r} "
                    "was not found in the user prompt"
                )
            available_booleans.remove(boolean)


def generate_response(functions_def: FunctionsDefinition,
                      input_prompt: str,
                      llm: tuple[
                          Small_LLM_Model,
                          dict[str, int]
                          ] | None = None,
                      max_res_tokens: int = 512,
                      vocabulary: TokenVocabulary | None = None,
                      observer: GenerationObserver | None = None) -> str:
    """Generate a response based on the model,the functions definition,
    and the user prompt.

    Args:
        functions_def (FunctionsDefinition): The definitions of the functions
        to include in the prompt.
        input_prompt (str): The user prompt to which the model should respond.
        llm (tuple[Small_LLM_Model, dict[str, int]] | None):
        Optional pre-loaded model and token mapping to use for generation.

    Returns:
        str : the JSON response generated by the model
    """
    prompt = build_prompt(functions_def, input_prompt)
    if llm is None:
        llm = load_model()
    return generate_constrained_response(
        llm[0],
        llm[1],
        functions_def,
        prompt,
        input_prompt,
        max_res_tokens,
        vocabulary=vocabulary,
        observer=observer,
        token_selector=next_token_selection,
        fsm_factory=JSONStateMachine,
    )


def validate_generated_response(
    functions_def: FunctionsDefinition,
    prompt: str,
    response: str,
) -> dict[str, object]:
    """Parse and validate a response against its function schema."""
    try:
        response_dict = json.loads(response)
    except json.JSONDecodeError as exc:
        raise ValueError(
            "Constrained generation produced invalid JSON for "
            f"prompt {prompt!r}: {exc}"
        ) from exc

    try:
        function_name = response_dict["name"]
        OutputModel = functions_def.get_output_function_model(
            function_name
        )
        validated_response = OutputModel.model_validate(response_dict)
        validated_dict: dict[str, object] = validated_response.model_dump()
        parameters = validated_dict["parameters"]
        if not isinstance(parameters, dict):
            raise ValueError("Generated parameters must be an object")
        validate_grounded_parameters(
            functions_def,
            prompt,
            function_name,
            parameters,
        )
        return validated_dict
    except (KeyError, ValueError, ValidationError) as exc:
        raise ValueError(
            "Generated response does not match a supplied function "
            f"schema for prompt {prompt!r}:\n{exc}"
        ) from exc


def run_cli(functions_definition_path: str,
            input_path: str | None = None,
            output_path: str | None = None
            ) -> list[dict[str, object]]:
    """Run CLI function calling pipeline with error handling.

    Args:
        functions_definition_path: Path to JSON file with function definitions.
        input_path: Path to JSON file with prompts, or None for stdin.
        output_path: Path to write output JSON, or None for stdout.

    Returns:
        List of results (dict with prompt and response).
    """
    try:
        functions_def = FunctionsDefinition.from_json(
            functions_definition_path)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error loading functions definition: {exc}",
              file=__import__("sys").stderr)
        raise

    try:
        prompts = load_prompts(input_path)
    except ValueError as exc:
        print(f"Error loading input prompts: {exc}",
              file=__import__("sys").stderr)
        raise

    print("Loading model...")
    llm = load_model()
    print("Model loaded.")
    vocabulary = TokenVocabulary(llm[0], llm[1])
    results: list[dict[str, object]] = []
    start_time = timeit.default_timer()
    for prompt in prompts:
        response = generate_response(
            functions_def,
            prompt,
            llm=llm,
            vocabulary=vocabulary,
        )
        validated_dict = validate_generated_response(
            functions_def,
            prompt,
            response,
        )
        results.append(validated_dict)

    end_time = timeit.default_timer()
    minutes = (end_time - start_time) / 60
    seconds = (end_time - start_time) % 60
    message = f"Total execution time: {minutes:.0f}"
    message += f" minutes and {seconds:.0f} seconds"
    print(message)

    if output_path is not None:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(json.dumps(results,
                                          indent=2,
                                          ensure_ascii=False
                                          ), encoding="utf-8")
    elif len(results) == 1:
        print(results[0])
    else:
        print(json.dumps(results, indent=2, ensure_ascii=False))

    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--functions_definition",
        default="data/input/functions_definition.json",
        help="Path to the function definitions JSON file.",
    )
    parser.add_argument(
        "--input",
        dest="input_path",
        default="data/input/function_calling_tests.json",
        # default=None,
        help="Path to the list of prompts JSON file.",
    )
    parser.add_argument(
        "--output",
        dest="output_path",
        default="data/output/function_calling_results.json",
        # default=None,
        help="Path where the generated responses should be written.",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Open the Textual constrained-generation visualizer.",
    )
    args = parser.parse_args(argv)

    try:
        if args.visualize:
            from src.visualizer import run_visualizer

            run_visualizer(
                args.functions_definition,
                args.input_path,
                args.output_path,
            )
        else:
            run_cli(
                args.functions_definition,
                args.input_path,
                args.output_path,
            )
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
