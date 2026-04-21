from llm_sdk import Small_LLM_Model
import json
import numpy as np
from functions import FunctionsDefinition

def build_prompt(functions_def: FunctionsDefinition, prompt: str) -> str:
    """Start with a base instruction, then append the functions definition,
    and finally the user prompt.
    """
    new_prompt = ""
    new_prompt = "You are a assistant that helps with function calls.\n\n"
    new_prompt += functions_def.get_functions_prompt()
    new_prompt += "Now, answer the following question:\n"
    new_prompt += prompt
    return new_prompt


def _build_token_to_id(vocab: dict) -> dict[str, int]:
    """
    Return a token->id mapping from either common vocab JSON shape.
    Verify the shape of the vocab and convert it
    to a consistent token->id mapping.
    """
    # Shape A: {"!": 0, "the": 1, ...}
    if all(isinstance(k, str) and isinstance(v, (int, str)) for k, v in vocab.items()):
        try:
            return {k: int(v) for k, v in vocab.items()}
        except (TypeError, ValueError):
            pass
    # Shape B: {"0": "!", "1": "the", ...}
    try:
        return {v: int(k) for k, v in vocab.items()}
    except (TypeError, ValueError) as exc:
        raise ValueError("Unsupported vocab format for token/id conversion") from exc


def next_token_selection(model, current_ids: list[int], allowed_ids: set[int]) -> int:
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


def get_allowed_tokens_for_string(
        target_string,
        current_generated_text,
        token_to_id
        ) -> set[int]:
    """
    Determine the next character that is expected
    and allow tokens that start with it.
    """
    # Si on a déjà fini la chaîne
    if current_generated_text == target_string:
        return {token_to_id.get(" ")}  # Ou un token de ponctuation comme ':'
    
    # Trouver ce qu'il reste à générer
    remaining = target_string[len(current_generated_text):]
    
    allowed = set()
    for token_str, t_id in token_to_id.items():
        # On autorise les tokens qui sont le début de ce qu'il reste à écrire
        # Attention : gérer les espaces de début de token (ex: 'Ġ' ou ' ')
        clean_token = token_str.replace('Ġ', ' ').replace(' ', ' ')
        if remaining.startswith(clean_token) and clean_token != "":
            allowed.add(t_id)
    return allowed

def get_filtered_vocab_for_functions(functions_names: list[str], token_to_id: dict[str, int]) -> list[tuple[str, int]]:
    """Return a filtered list of (token_str, token_id) that are relevant for the function names and syntax."""
   
    syntax_tokens = ['{"', '":', ',"', '": "', '",', '}', '[', ']', '": ', ' {', 'true', 'false', 'null']

    filtered_vocab = []
    for t_str, t_id in token_to_id.items():
        clean_t = t_str.replace('Ġ', ' ').replace(' ', ' ')
        is_syntax = any(syntax == clean_t for syntax in syntax_tokens)
        is_part_of_fn = any(clean_t in fn for fn in functions_names)
        is_digit = clean_t.strip().isdigit() or clean_t in [".", "-", "e"]
        
        if is_syntax or is_part_of_fn or is_digit:
            filtered_vocab.append((clean_t, t_id))

    return filtered_vocab

def test_small_llm_model():
    input_prompt = input("input_prompt:")
    print("input_prompt:", input_prompt)
    print("Loading functions definition...")
    functions_def = FunctionsDefinition.from_json("data/input/functions_definition.json")
    print("Building prompt...")
    prompt = build_prompt(functions_def, input_prompt)
    print("Instruction:", prompt)
    model = Small_LLM_Model(
        device="cpu",
        cache_dir="./.hf_cache",
        local_files_only=True, # First run will download the model, then True for subsequent runs
    )
    max_res_tokens = 20
    tokens_ids = model.encode(prompt)[0].tolist()
    response_tokens_ids: list[int] = []

    # Charger le vocabulaire
    vocab_path = model.get_path_to_vocab_file()
    with open(vocab_path) as f:
        vocab = json.load(f)
    token_to_id = _build_token_to_id(vocab)
    
    functions_names = functions_def.list_functions_name()
    relevant_tokens = get_filtered_vocab_for_functions(functions_names, token_to_id)

    current_text = ""
    for i in range(max_res_tokens):
        allowed_tokens = set()
        still_possible = [s for s in functions_names if s.startswith(current_text)]
        for s in still_possible:
            allowed_tokens.update(get_allowed_tokens_for_string(s, current_text, token_to_id))
        remaining_needed = [s[len(current_text):] for s in still_possible if len(s) > len(current_text)]
        # 2. Si rien n'est possible, on arrête ou on gère l'erreur
        if not remaining_needed:
            break
        for clean_t, t_id in relevant_tokens:
            if clean_t == "": 
                continue
            # Si le token correspond au début de l'un des restes attendus
            if any(rem.startswith(clean_t) for rem in remaining_needed):
                allowed_tokens.add(t_id)

        if not allowed_tokens:
            print(f"Block after : '{current_text}'")
            break
        new_token_id = next_token_selection(model, tokens_ids, allowed_tokens)
    
        tokens_ids.append(new_token_id)
        response_tokens_ids.append(new_token_id)
        current_text += model.decode([new_token_id])

    print(f"Final response token ids: {response_tokens_ids}")
    print(f"Final response: {model.decode(response_tokens_ids)}")


if __name__ == "__main__":
    test_small_llm_model()
