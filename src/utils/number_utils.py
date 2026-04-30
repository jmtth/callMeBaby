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
    The '-' can only be at the start or immediately after an 'e'.

    args:
        str: the string fragment of a number

    returns:
        bool: true if it is a valid number fragment
    """
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


def is_complete_number(text: str) -> bool:
    """Check if the text is a complete and valid number.
    A complete number can be an integer or a float,
    and may include an optional leading '-' for negatives
    and an optional 'e' for scientific notation.
    It must have at least one digit, and if it has a decimal point,
    it must have digits after it.
    like "123", "123.45", "-123.45", "1e10", "-1e-10", "1.5e+10" are valid,
    but "123.", "-123.", "1e", "1e-", "1e+", ".e10", "-.e10" are not valid.

    args:
        str: a number in string

    returns:
        bool: true if it is a valid number

    """
    if text == "":
        return False

    if text[-1] in {"-", ".", "e"}:
        return False

    if "e" in text:
        left, right = text.split("e", 1)
        if left in {"", "-", ".", "-."}:
            return False
        if right in {"", "+", "-"}:
            return False
        if right[0] in {"+", "-"}:
            right = right[1:]
        return right.isdigit()

    if text in {"-", ".", "-."}:
        return False

    return any(ch.isdigit() for ch in text)