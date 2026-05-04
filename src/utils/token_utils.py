from src.utils.number_utils import is_valid_number_fragment


def is_number_terminator_token(token_text: str) -> bool:
    """Check if the token text is a valid number terminator
    (whitespace, commas, braces).

    args:
        str : a token string

    returns:
        bool: true if the token is a valid terminator
    """
    return token_text in {" ", "  ", ",", ", ", " ,", "}"}


def get_number_token_ids(token_to_id: dict) -> set[int]:
    """Get token ids that can be part
    of a number (digits, '.', 'e', '-').

    returns:
        set[int]: token ids that can be part of a number
    """
    allowed = set()
    for token_str, token_id in token_to_id.items():
        clean_t = token_str.replace('Ġ', ' ').replace(' ', ' ')
        if clean_t == "":
            continue
        if " " in clean_t:
            continue
        if is_valid_number_fragment(clean_t):
            allowed.add(token_id)
    return allowed


def get_number_terminator_token_ids(token_to_id: dict) -> set[int]:
    """Get token ids that can terminate a number
    (whitespace, commas, braces).

    returns:
        set[int]: token ids that can terminate a number
    """
    terminators = set()
    for exact_text in [" ", "  ", ",", ", ", " ,", "}"]:
        terminators.update(get_exact_token_ids(token_to_id, exact_text))
    return terminators


def get_exact_token_ids(token_to_id: dict, exact_text: str) -> set[int]:
    """Get token ids that correspond exactly to the given text.
    This is used for fixed tokens like punctuation or keywords.
    """
    allowed = set()
    for token_str, token_id in token_to_id.items():
        clean_t = token_str.replace('Ġ', ' ').replace(' ', ' ')
        if clean_t == exact_text:
            allowed.add(token_id)
    return allowed
