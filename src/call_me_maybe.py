from __future__ import annotations

import argparse
import json
from pathlib import Path
from pydantic import ValidationError


import numpy as np
from llm_sdk import Small_LLM_Model

from src.JSONStateMachine import JSONStateMachine
from src.functions_manager import FunctionsDefinition
from src.models import JSONState
from src import utils


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
    new_prompt = "You are a assistant that helps with function calls.\n\n"
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
    # Shape A: {"!": 0, "the": 1, ...}
    if all(isinstance(k, str) and isinstance(v, (int, str))
           for k, v in vocab.items()):
        try:
            return {k: int(v) for k, v in vocab.items()}
        except (TypeError, ValueError):
            pass
    # Shape B: {"0": "!", "1": "the", ...}
    try:
        return {v: int(k) for k, v in vocab.items()}
    except (TypeError, ValueError) as exc:
        raise ValueError("Unsupported vocab format for conversion") from exc


def next_token_selection(model,
                         current_ids: list[int],
                         allowed_ids: set[int]
                         ) -> int:
    """Select the next token id with the best probability

    Args:
        model: llm_sdk
        current_ids: Ids list of current prompt
        allowed_ids: list of ids of allowed token

    Returns:
        int: the choosen token
    """
    logits = model.get_logits_from_input_ids(current_ids)
    logits_np = np.array(logits)

    if not allowed_ids:
        raise ValueError("No allowed tokens available for selection")

    mask = np.full_like(logits_np, float("-inf"))
    indices = list(allowed_ids)
    mask[indices] = logits_np[indices]

    return int(np.argmax(mask))


def load_model(cache_dir: str = "./.hf_cache") -> tuple[Small_LLM_Model,
                                                        dict[str, int]]:
    """Load the small LLM model.

    First, try to load with local_files_only=True to avoid downloads.
    If the model files are not found locally, local_files_only=False.

    Args:
        cache_dir: Directory to use for caching model files.

    Returns:
        Tuple of (model, token_to_id mapping).
    """
    system = __import__("platform").system().lower()
    if system == 'linux':
        device = "cpu"
    else:
        device = "mps"
    try:
        model = Small_LLM_Model(device=device,
                                cache_dir=cache_dir,
                                local_files_only=True)
    except Exception:
        model = Small_LLM_Model(device=device,
                                cache_dir=cache_dir,
                                local_files_only=False)
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


def generate_response(functions_def: FunctionsDefinition,
                      input_prompt: str,
                      llm=None,
                      max_res_tokens: int = 110) -> str:
    """Generate a response based on the model,the functions definition,
    and the user prompt.

    Args:
        functions_def (FunctionsDefinition): The definitions of the functions
        to include in the prompt.
        input_prompt (str): The user prompt to which the model should respond.
        llm: Optional pre-loaded model to use for generation.

    Returns:
        str : the JSON response generated by the model
    """
    prompt = build_prompt(functions_def, input_prompt)
    if llm is None:
        llm = load_model()
    tokens_ids = llm[0].encode(prompt)[0].tolist()
    response_tokens_ids: list[int] = []

    fsm = JSONStateMachine(llm[0], functions_def, llm[1], input_prompt)

    current_text = ""
    for i in range(max_res_tokens):
        if fsm.is_in_fixed_sequence():
            target_tokens = fsm.get_target_tokens_for_current_state()
            tokens_ids.extend(target_tokens)
            response_tokens_ids.extend(target_tokens)
            current_text += llm[0].decode(target_tokens)
            fsm.progress = max(len(target_tokens) - 1, 0)
            fsm.update(target_tokens[-1])
            if fsm.state == JSONState.STOP:
                break
        else:
            allowed_tokens = fsm.get_allowed_tokens()
            if not allowed_tokens or fsm.state == JSONState.END:
                break
            new_token_id = next_token_selection(llm[0],
                                                tokens_ids,
                                                allowed_tokens)
            keep_token = fsm.update(new_token_id)
            if keep_token:
                tokens_ids.append(new_token_id)
                response_tokens_ids.append(new_token_id)
                current_text += llm[0].decode([new_token_id])
                if fsm.param_repeat_pattern:
                    response_tokens_ids = utils.remove_repeating_pattern(
                        llm[0],
                        response_tokens_ids,
                        fsm.param_repeat_pattern)

    return llm[0].decode(response_tokens_ids)


def run_cli(functions_definition_path: str,
            input_path: str | None = None,
            output_path: str | None = None
            ) -> list[dict[str, str]]:
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

    llm = load_model()

    results: list[dict[str, str]] = []
    for prompt in prompts:
        response = generate_response(functions_def, prompt, llm=llm)
        try:
            response_dict = json.loads(response)
        except json.JSONDecodeError as exc:
            print(f"Warning: Generated response is not valid JSON: {exc}",
                  file=__import__("sys").stderr)
            results.append({"prompt": prompt,
                            "error": "Invalid generated JSON"})
            continue

        OutputModel = functions_def.get_output_function_model(
            response_dict['name'])
        try:
            OutputModel.model_validate(response_dict)
        except ValidationError as e:
            print(e)
        results.append(response_dict)

    if output_path is not None:
        Path(output_path).write_text(json.dumps(results,
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
        default=None,
        help="Path to the list of prompts JSON file.",
    )
    parser.add_argument(
        "--output",
        dest="output_path",
        default=None,
        help="Path where the generated responses should be written.",
    )
    args = parser.parse_args(argv)

    run_cli(args.functions_definition, args.input_path, args.output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
