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


def _build_token_to_id(vocab: dict) -> dict[str, int]:
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


def get_filtered_vocab_for_functions(functions_names: list[str],
                                     functions_descriptions: dict[str, str],
                                     token_to_id: dict[str, int]
                                     ) -> dict[str, int]:
    """Return a smaller filtered list of (token_str, token_id)
    that are relevant for the function names and syntax.

    Reducing the vocabulary allows for better performance.

    Args:
        functions_names: list of function names to include in the vocab.
        functions_descriptions: dictionary mapping function names.
        token_to_id: the original vocabulary mapping.

    Returns:
        dict[str, int]: The reduced vocabulary mapping.
    """

    syntax_tokens = [
        '{"', '":', ',"', '": "', '",', '[', ']', '": ', ' {',
        'true', 'false', 'null', ',', ' ', '"', '\\', '{', '}', ':'
    ]
    desciptions_token = [token for desc in functions_descriptions.values()
                         for token in desc.split()]
    alphanumeric_tokens = set("abcdefghijklmnopqrstuvwxyz"
                              "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                              "0123456789")
    filtered_vocab = dict[str, int]()
    for t_str, t_id in token_to_id.items():
        clean_t = t_str.replace('Ġ', ' ').replace(' ', ' ')
        is_alphanumeric = all(ch in alphanumeric_tokens for ch in clean_t)
        is_syntax = any(syntax == clean_t for syntax in syntax_tokens)
        is_part_of_fn = any(clean_t in fn for fn in functions_names)
        is_part_of_desc = any(clean_t in desc for desc in desciptions_token)
        is_digit = clean_t.strip().isdigit() or clean_t in [".", "-", "e"]

        if is_syntax or is_part_of_fn or is_digit or is_part_of_desc or is_alphanumeric:
            # Keep original token string as key to avoid collisions between
            # different raw tokens that normalize to the same cleaned text.
            filtered_vocab[t_str] = t_id
    print(f"vocab size: {len(token_to_id)}, filtered vocab size: {len(filtered_vocab)}t")
    return filtered_vocab


def next_token_selection(model,
                         current_ids: list[int],
                         allowed_ids: set[int]
                         ) -> int:
    """Select the next token id with the best probability

    args:
        model: llm_sdk
        current_ids: Ids list of current prompt
        allowed_ids: list of ids of allowed token
    return:
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


def load_model(cache_dir: str = "./.hf_cache") -> Small_LLM_Model:
    """Load the small LLM model.

    First, try to load with local_files_only=True to avoid downloads.
    If the model files are not found locally, local_files_only=False.
    """
    system = __import__("platform").system().lower()
    if system == 'linux':
        device = "cpu"
    else:
        device = "mps"
    try:
        return Small_LLM_Model(device=device,
                               cache_dir=cache_dir,
                               local_files_only=True)
    except Exception:
        return Small_LLM_Model(device=device,
                               cache_dir=cache_dir,
                               local_files_only=False)


def _load_prompts(input_path: str | None) -> list[str]:
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
    except FileNotFoundError as exc:
        raise ValueError(f"Input file not found: {input_path}") from exc
    
    if not raw_text:
        raise ValueError(f"Input file is empty: {input_path}")
    
    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {input_path}: {exc}") from exc

    if isinstance(data, str):
        return [data]
    if isinstance(data, dict):
        if "prompt" in data:
            return [str(data["prompt"])]
        if "prompts" in data and isinstance(data["prompts"], list):
            return [str(item) for item in data["prompts"]]
        raise ValueError(f"JSON dict must contain 'prompt' or 'prompts' key: {input_path}")
    if isinstance(data, list):
        prompts: list[str] = []
        for item in data:
            if isinstance(item, dict) and "prompt" in item:
                prompts.append(str(item["prompt"]))
            elif isinstance(item, str):
                prompts.append(item)
            else:
                raise ValueError(f"List items must be strings or dicts with 'prompt' key: {input_path}")
        return prompts
    
    raise ValueError(f"JSON must be string, dict, or list: {input_path}")


def generate_response(functions_def: FunctionsDefinition, input_prompt: str, model=None, max_res_tokens: int = 110) -> str:
    prompt = build_prompt(functions_def, input_prompt)
    if model is None:
        model = load_model()
    tokens_ids = model.encode(prompt)[0].tolist()
    response_tokens_ids: list[int] = []

    vocab_path = model.get_path_to_vocab_file()
    with open(vocab_path, encoding="utf-8") as f:
        vocab = json.load(f)
    token_to_id = _build_token_to_id(vocab)

    # functions_names = functions_def.list_functions_name()
    # functions_descriptions = {fn.name: fn.description for fn in functions_def.functions}
    # relevant_tokens = get_filtered_vocab_for_functions(functions_names, functions_descriptions, token_to_id)

    fsm = JSONStateMachine(model, functions_def, token_to_id, input_prompt)

    current_text = ""
    for i in range(max_res_tokens):
        if fsm.is_in_fixed_sequence():
            target_tokens = fsm.get_target_tokens_for_current_state()
            tokens_ids.extend(target_tokens)
            response_tokens_ids.extend(target_tokens)
            current_text += model.decode(target_tokens)
            fsm.progress = len(target_tokens) - 1 if len(target_tokens) - 1 > 0 else 0
            fsm.update(target_tokens[-1])
            if fsm.state == JSONState.STOP:
                break
        else:
            allowed_tokens = fsm.get_allowed_tokens()
            if not allowed_tokens or fsm.state == JSONState.END:
                break
            new_token_id = next_token_selection(model,
                                                tokens_ids,
                                                allowed_tokens)

            keep_token = fsm.update(new_token_id)
            if keep_token:
                tokens_ids.append(new_token_id)
                response_tokens_ids.append(new_token_id)
                current_text += model.decode([new_token_id])
                if fsm.param_repeat_pattern:
                    response_tokens_ids = utils.remove_repeating_pattern(model,
                                                                         response_tokens_ids,
                                                                         fsm.param_repeat_pattern)

    return model.decode(response_tokens_ids)


def run_cli(functions_definition_path: str, input_path: str | None = None, output_path: str | None = None) -> list[dict[str, str]]:
    """Run CLI function calling pipeline with error handling.
    
    Args:
        functions_definition_path: Path to JSON file with function definitions.
        input_path: Path to JSON file with prompts, or None for stdin.
        output_path: Path to write output JSON, or None for stdout.
        
    Returns:
        List of results (dict with prompt and response).
    """
    try:
        functions_def = FunctionsDefinition.from_json(functions_definition_path)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error loading functions definition: {exc}", file=__import__("sys").stderr)
        raise
    
    try:
        prompts = _load_prompts(input_path)
    except ValueError as exc:
        print(f"Error loading input prompts: {exc}", file=__import__("sys").stderr)
        raise
    
    model = load_model()

    results: list[dict[str, str]] = []
    for prompt in prompts:
        response = generate_response(functions_def, prompt, model=model)
        try:
            response_dict = json.loads(response)
        except json.JSONDecodeError as exc:
            print(f"Warning: Generated response is not valid JSON: {exc}", file=__import__("sys").stderr)
            results.append({"prompt": prompt, "error": "Invalid generated JSON"})
            continue

        OutputModel = functions_def.get_output_function_model(
            response_dict['name'])
        try:
            OutputModel.model_validate(response_dict)
        except ValidationError as e:
            print(e)
        results.append(response_dict)

    if output_path is not None:
        Path(output_path).write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
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
