from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from llm_sdk import Small_LLM_Model

from src.JSONStateMachine import JSONStateMachine
from src.functions_manager import FunctionsDefinition
from src.models import JSONState


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
        'true', 'false', 'null', ',', ' '
    ]
    desciptions_token = [token for desc in functions_descriptions.values()
                         for token in desc.split()]

    filtered_vocab = dict[str, int]()
    for t_str, t_id in token_to_id.items():
        clean_t = t_str.replace('Ġ', ' ').replace(' ', ' ')
        is_syntax = any(syntax == clean_t for syntax in syntax_tokens)
        is_part_of_fn = any(clean_t in fn for fn in functions_names)
        is_part_of_desc = any(clean_t in desc for desc in desciptions_token)
        is_digit = clean_t.strip().isdigit() or clean_t in [".", "-", "e"]

        if is_syntax or is_part_of_fn or is_digit or is_part_of_desc:
            filtered_vocab[clean_t] = t_id

    return filtered_vocab


def next_token_selection(model,
                         current_ids: list[int],
                         allowed_ids: set[int]
                         ) -> int:
    """Given the current token ids and a set of allowed token ids,
    return the next token id
    """
    logits = model.get_logits_from_input_ids(current_ids)
    logits_np = np.array(logits)

    if not allowed_ids:
        raise ValueError("No allowed tokens available for selection")

    mask = np.full_like(logits_np, float("-inf"))
    indices = list(allowed_ids)
    mask[indices] = logits_np[indices]

    return int(np.argmax(mask))


def load_model(device: str = "cpu",
               cache_dir: str = "./.hf_cache"
               ) -> Small_LLM_Model:
    """Load the small LLM model.
    First, try to load with local_files_only=True to avoid downloads.
    If the model files are not found locally, local_files_only=False.
    """
    try:
        return Small_LLM_Model(device=device,
                               cache_dir=cache_dir,
                               local_files_only=True)
    except Exception:
        return Small_LLM_Model(device=device,
                               cache_dir=cache_dir,
                               local_files_only=False)


def _load_prompts(input_path: str | None) -> list[str]:
    if input_path is None:
        return [input("input_prompt:")]
    try:
        raw_text = Path(input_path).read_text(encoding="utf-8").strip()
    except FileNotFoundError as exc:
        raise ValueError(f"File not found: {input_path}") from exc
    if not raw_text:
        return [""]

    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError:
        return [raw_text]

    if isinstance(data, str):
        return [data]
    if isinstance(data, dict):
        if "prompt" in data:
            return [str(data["prompt"])]
        if "prompts" in data and isinstance(data["prompts"], list):
            return [str(item) for item in data["prompts"]]
        return [json.dumps(data)]
    if isinstance(data, list):
        prompts: list[str] = []
        for item in data:
            if isinstance(item, dict) and "prompt" in item:
                prompts.append(str(item["prompt"]))
            else:
                prompts.append(str(item))
        return prompts
    return [raw_text]


def generate_response(functions_def: FunctionsDefinition, input_prompt: str, model=None, max_res_tokens: int = 30) -> str:
    prompt = build_prompt(functions_def, input_prompt)
    if model is None:
        model = load_model()
    tokens_ids = model.encode(prompt)[0].tolist()
    response_tokens_ids: list[int] = []

    vocab_path = model.get_path_to_vocab_file()
    with open(vocab_path, encoding="utf-8") as f:
        vocab = json.load(f)
    token_to_id = _build_token_to_id(vocab)

    functions_names = functions_def.list_functions_name()
    functions_descriptions = {fn.name: fn.description for fn in functions_def.functions}
    relevant_tokens = get_filtered_vocab_for_functions(functions_names, functions_descriptions, token_to_id)

    fsm = JSONStateMachine(model, functions_def, relevant_tokens, input_prompt)

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
            new_token_id = next_token_selection(model, tokens_ids, allowed_tokens)

            keep_token = fsm.update(new_token_id)
            if keep_token:
                tokens_ids.append(new_token_id)
                response_tokens_ids.append(new_token_id)
                current_text += model.decode([new_token_id])

    return model.decode(response_tokens_ids)


def run_cli(functions_definition_path: str, input_path: str | None = None, output_path: str | None = None) -> list[dict[str, str]]:
    functions_def = FunctionsDefinition.from_json(functions_definition_path)
    model = load_model()
    prompts = _load_prompts(input_path)

    results: list[dict[str, str]] = []
    for prompt in prompts:
        response = generate_response(functions_def, prompt, model=model)
        #results.append(json.loads(response) if response.startswith("{") else {"response": response})
        results.append(response)

    if output_path is not None:
        Path(output_path).write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    elif len(results) == 1:
        print(results[0]["response"])
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
