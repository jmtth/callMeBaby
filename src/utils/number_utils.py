import re


def extract_decimal_counts(string: str) -> list[int]:
    """Extract the number of decimal places for each number in a string.
    For integers, count as 1 (to allow at least one decimal place if needed).
    For floats, count the number of digits after the decimal point.
    if the number has no decimal part, count as 1 to allow
    for a decimal point and at least one digit after it.
    """
    counts = []
    for match in re.finditer(r"-?\d+(?:\.(\d+))?", string):
        frac = match.group(1)
        counts.append(len(frac) if frac is not None else 1)
    return counts


def is_valid_number_fragment(text: str) -> bool:
    """Check if the text is a valid fragment of a number.
    Valid fragments can be empty, contain digits, a single decimal point,
    a single 'e' for scientific notation, and a '-' for negative numbers.
    The '-' can only be at the start or immediately after an 'e'."""
    if text == "":
        return True

    chars = set("0123456789-.e")
    if any(ch not in chars for ch in text):
        return False

    if text.count("e") > 1:
        return False
    if text.count(".") > 1:
        return False

    e_pos = text.find("e")
    if e_pos != -1 and text.find(".", e_pos) != -1:
        return False

    if "-" in text:
        for i, ch in enumerate(text):
            if ch != "-":
                continue
            if i == 0:
                continue
            if i > 0 and text[i - 1] == "e":
                continue
            return False

    return True
