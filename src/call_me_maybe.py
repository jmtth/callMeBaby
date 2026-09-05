from __future__ import annotations
import argparse
from decimal import Decimal, InvalidOperation
import json
import sys
from pathlib import Path
from pydantic import ValidationError
from src.models.prompt import PromptSchema
from llm_sdk import Small_LLM_Model, logging
from src.JSONStateMachine import JSONStateMachine
from src.functions_manager import FunctionsDefinition
from src.generator import (
    GenerationObserver,
    generate_constrained_response,
    select_next_token,
)
from src.grounding import (
    collect_prompt_string_candidates,
    collect_prompt_values,
    contains_response_structure,
)
from src.token_vocabulary import TokenVocabulary
import timeit
from src.utils.logger import CallMeLogger

logger = CallMeLogger()


def build_prompt(functions_def: FunctionsDefinition,
                 prompt: str,
                 function_name: str | None = None) -> str:
    """Build the model prompt from instructions, function schemas, and input.

    Args:
        functions_def: Function definitions available to the model.
        prompt: User request to append to the model instructions.
        function_name: Optional committed function to show by itself.

    Returns:
        The complete prompt to send to the model.
    """
    functions_prompt = (
        functions_def.get_functions_prompt()
        if function_name is None
        else functions_def.get_functions_prompt(function_name)
    )
    system_prompt = (
        "You are a function calling router. "
        "Available functions:\n"
        f"{functions_prompt}\n. "
        "Return a JSON object with the name of the function that matches "
        "the user request. The parameter VALUES must be extracted DIRECTLY "
        "and LITERALLY from the user prompt when possible."
    )
    return (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{prompt}<|im_end|>\n"
        "<|im_start|>assistant\n"
        "<think>\n\n</think>\n\n"
    )


def build_token_to_id(vocab: dict) -> dict[str, int]:
    """Normalize a supported vocabulary shape to a token-to-ID mapping.

    Args:
        vocab: Mapping from IDs to tokens or from tokens to IDs.

    Returns:
        A mapping from token strings to integer IDs.

    Raises:
        ValueError: If the vocabulary is empty or has an unsupported shape.
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
    """Load the small language model and its token vocabulary.

    Returns:
        The model and its normalized token-to-ID mapping.
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
        Prompt strings in their input order.

    Raises:
        ValueError: If the file is missing, empty, malformed, or has an
            unsupported JSON structure.
    """
    if input_path is None:
        return [input("input_prompt:")]

    try:
        raw_text = Path(input_path).read_text(encoding="utf-8").strip()
        if not raw_text:
            raise ValueError(f"Input file is empty: {input_path}")
    except FileNotFoundError as exc:
        raise ValueError(f"Input file not found: {input_path}") from exc
    except Exception as exc:
        raise ValueError(
            f"Error reading input prompt file {input_path}: {exc}"
            ) from exc

    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {input_path}: {exc}") from exc

    def parse_prompt_item(item: object) -> str:
        """Normalize one supported prompt item to its string value."""
        if isinstance(item, str):
            return item
        if isinstance(item, dict):
            try:
                return PromptSchema(**item).prompt
            except ValidationError as exc:
                raise ValueError(f"Invalid prompt object: {exc}") from exc
        raise ValueError("List items must be strings or dicts")

    if isinstance(data, str):
        return [data]

    if isinstance(data, dict):
        if "prompt" in data:
            return [parse_prompt_item(data)]
        if "prompts" not in data:
            raise ValueError(
                "Prompt object must contain 'prompt' or 'prompts' key"
            )
        prompt_items = data["prompts"]
        if not isinstance(prompt_items, list):
            raise ValueError("'prompts' must contain a list")
        return [parse_prompt_item(item) for item in prompt_items]

    if isinstance(data, list):
        return [parse_prompt_item(item) for item in data]

    raise ValueError("JSON must be string, dict, or list")


def validate_grounded_parameters(functions_def: FunctionsDefinition,
                                 prompt: str,
                                 function_name: str,
                                 parameters: dict[str, object]) -> None:
    """Reject typed values that were invented instead of extracted.

    Function selection remains an LLM decision. This validation checks exact
    provenance for unambiguous extractive strings and for typed literals,
    while leaving derived strings generative.

    Args:
        functions_def: Definitions loaded from the supplied JSON file.
        prompt: Original user request.
        function_name: Function selected by the LLM.
        parameters: Validated generated parameter values.

    Raises:
        ValueError: If a grounded parameter value is absent from the prompt.
    """
    schema = functions_def.get_function_parameters_by_name(function_name)
    available = collect_prompt_values(prompt)
    available_numbers = list(available.numbers)
    available_booleans = list(available.booleans)

    for parameter_name, parameter_schema in schema.items():
        value = parameters[parameter_name]
        if (
            parameter_schema.type == "string"
            and isinstance(value, str)
            and contains_response_structure(value, prompt)
        ):
            raise ValueError(
                f"String parameter {parameter_name!r} contains an embedded "
                "response-structure fragment"
            )
        if parameter_schema.type == "string" and isinstance(value, str):
            candidates = collect_prompt_string_candidates(
                prompt,
                parameter_name,
            )
            if candidates and value not in candidates:
                raise ValueError(
                    f"String parameter {parameter_name!r}={value!r} "
                    "was not extracted exactly from the user prompt"
                )
        if parameter_schema.type in {"number", "integer"}:
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
    """Generate a schema-constrained response for a user prompt.

    Args:
        functions_def: Function definitions available for selection.
        input_prompt: User request to process.
        llm: Optional preloaded model and token-to-ID mapping.
        max_res_tokens: Maximum number of tokens in the generated response.
        vocabulary: Optional reusable token-vocabulary cache.
        observer: Optional callback invoked after generation decisions.

    Returns:
        The JSON response generated by the model.
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
        selected_prompt_factory=lambda function_name: build_prompt(
            functions_def,
            input_prompt,
            function_name,
        ),
    )


def validate_generated_response(
    functions_def: FunctionsDefinition,
    prompt: str,
    response: str,
) -> dict[str, object]:
    """Parse, validate, and ground a generated function-call response.

    Args:
        functions_def: Function definitions used to select the output schema.
        prompt: Original user request used for grounding checks.
        response: Generated JSON response.

    Returns:
        The validated response converted to plain Python values.

    Raises:
        ValueError: If the response is invalid JSON, violates the selected
            function schema, or contains ungrounded typed parameters.
    """
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
        Validated function-call results.

    Raises:
        ValueError: If definitions, prompts, or generated responses are
            invalid.
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

    logger.info("Loading model...")
    llm = load_model()
    logger.info("Model loaded.")
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
        logger.debug(f"Prompt: {prompt}")
        logger.debug(f"Response: {response}")
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
    logger.info(message)

    if output_path is not None:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(json.dumps(results,
                                          indent=2,
                                          ensure_ascii=False
                                          ), encoding="utf-8")
    elif len(results) == 1:
        logger.info(results[0])
    else:
        logger.info(json.dumps(results, indent=2, ensure_ascii=False))

    return results


def main(argv: list[str] | None = None) -> int:
    """Run the command-line interface and return its exit status."""
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
        help="Path to the list of prompts JSON file.",
    )
    parser.add_argument(
        "--output",
        dest="output_path",
        default="data/output/function_calling_results.json",
        help="Path where the generated responses should be written.",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Open the Textual constrained-generation visualizer.",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-0.6B",
        help="Name of the small LLM model to use.",
    )
    parser.add_argument(
        "--loglevel",
        default="INFO",
        choices=["DEBUG", "INFO"],
        help="Set the logging level.",
    )
    args = parser.parse_args(argv)

    logger.set_level(level=args.loglevel)

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
