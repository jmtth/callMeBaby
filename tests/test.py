from llm_sdk import Small_LLM_Model
import json


def _build_token_to_id(vocab: dict) -> dict[str, int]:
    """Return a token->id mapping from either common vocab JSON shape."""
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

def next_token_constrained(model, token_ids: list[int], allowed_ids: set[int]) -> int:
    logits = model.get_logits_from_input_ids(token_ids)

    # Mettre -inf sur tous les tokens non autorisés
    for i in range(len(logits)):
        if i not in allowed_ids:
            logits[i] = float("-inf")

    # Prendre le meilleur token restant
    return logits.index(max(logits))

def test_small_llm_model():
    model = Small_LLM_Model(
        device="cpu",
        cache_dir="./.hf_cache",
        local_files_only=True, # First run will download the model, then True for subsequent runs
    )
    max_res_tokens = 20
    tokens_ids = model.encode("What is the capital of France?")[0].tolist()
    response_tokens_ids: list[int] = []

    # Charger le vocabulaire
    vocab_path = model.get_path_to_vocab_file()
    with open(vocab_path) as f:
        vocab = json.load(f)
    token_to_id = _build_token_to_id(vocab)
    allowed_tokens = set()
    # tokens simples (OK via vocab)
    simple_tokens = ["{", "}", ":", ",", '"']
    allowed_tokens |= {token_to_id[t] for t in simple_tokens if t in token_to_id}
    # mots (via encode)
    words = ["{name"]
    for word in words:
        for variant in (word, " " + word):
            token_seq = model.encode(variant)[0].tolist()
            allowed_tokens |= set(token_seq)

    print(f"Prompt tokens_ids: {tokens_ids}")

    for i in range(max_res_tokens):
        if i == 0:
            new_token_id = next_token_constrained(model, tokens_ids, allowed_tokens)
        else:
            logits = model.get_logits_from_input_ids(tokens_ids)
            new_token_id = logits.index(max(logits))
        tokens_ids.append(new_token_id)
        response_tokens_ids.append(new_token_id)

    print(f"Final response token ids: {response_tokens_ids}")
    print(f"Final response: {model.decode(response_tokens_ids)}")


if __name__ == "__main__":
    test_small_llm_model()
