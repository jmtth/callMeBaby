from llm_sdk import Small_LLM_Model


def get_repeating_pattern(text: str,
                          min_len: int = 3,
                          max_repeats: int = 10) -> str:
    """Check if the text has a repeating pattern of 'cat' at the end.

    It is best not to research patterns shorter 3 characters,
    and that a pattern must be repeated at least 3 time.

    args:
        text (str): the input text to check for repeating patterns.
        min_len (int): the minimum length of the pattern to check for.
        max_repeats (int): the maximum number of repeats.

    returns:
        str: the repeating pattern if found, otherwise an empty string.
    """
    if len(text) < min_len * 2:
        return ""
    pattern_count = 1
    for len_pattern in range(min_len, len(text) // 2 + 1):
        text_pattern = text[-len_pattern:]
        start = len(text) - len_pattern * 2
        text_slice = text[start:start + len_pattern]
        while start >= 0 and text_slice == text_pattern:
            print(f"Found repeating pattern: '{text_pattern}' at position {start} in '{text}'")
            pattern_count += 1
            text_slice = text[start:start + len_pattern]
            if pattern_count >= max_repeats:
                return text_pattern
            start -= len_pattern
        pattern_count = 1
    return ""


def remove_repeating_pattern(model: Small_LLM_Model,response: list[int], pattern: str) -> list[int]:
    """Remove the repeating pattern from the end of the token list."""
    if not pattern:
        return response
    enc0 = model.encode(pattern)[0]
    if hasattr(enc0, "tolist"):
        pattern_tokens = enc0.tolist()
    else:
        pattern_tokens = list(enc0)
    pattern_len = len(pattern_tokens)
    if response[-pattern_len:] == pattern_tokens:
        print(f"Removing repeating pattern tokens: {pattern_tokens} from response: {response}")
        response = response[:-pattern_len]
    return response
