def get_repeating_pattern(text: str,
                          min_len: int = 3,
                          max_repeats: int = 2) -> str:
    """Find a suffix repeated enough times to indicate looping generation.

    Args:
        text: Text whose suffix should be inspected.
        min_len: Minimum repeated-pattern length.
        max_repeats: Number of occurrences required for a match.

    Returns:
        The repeated suffix, or an empty string when none is found.
    """
    if len(text) < min_len * 2:
        return ""
    pattern_count = 1
    for len_pattern in range(min_len, len(text) // 2 + 1):
        text_pattern = text[-len_pattern:]
        start = len(text) - len_pattern * 2
        text_slice = text[start:start + len_pattern]
        while start >= 0 and text_slice == text_pattern:
            pattern_count += 1
            text_slice = text[start:start + len_pattern]
            if pattern_count >= max_repeats:
                return text_pattern
            start -= len_pattern
        pattern_count = 1
    return ""
