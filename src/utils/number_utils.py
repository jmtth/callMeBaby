import re
from decimal import Decimal, InvalidOperation


NUMBER_PATTERN = re.compile(
    r"(?<![\w.])-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
    r"(?![\w.])"
)


def extract_numbers(string: str) -> list[Decimal]:
    """Extract numeric literals without losing their decimal value.

    Args:
        string: Natural-language text that may contain numeric literals.

    Returns:
        Numeric literals in their order of appearance.
    """
    numbers: list[Decimal] = []
    for match in NUMBER_PATTERN.finditer(string):
        try:
            numbers.append(Decimal(match.group(0)))
        except InvalidOperation:
            continue
    return numbers


def extract_decimal_counts(string: str) -> list[int]:
    """Return each numeric literal's decimal count, using one for integers."""
    counts = []
    for match in re.finditer(r"-?\d+(?:\.(\d+))?", string):
        frac = match.group(1)
        counts.append(len(frac) if frac is not None else 1)
    return counts


def is_valid_number_fragment(text: str) -> bool:
    """Return whether text can be a prefix of a supported number.

    Supported fragments use digits, at most one decimal point, an optional
    lowercase exponent, and minus signs only at the start or after ``e``.

    Args:
        text: Candidate numeric prefix.

    Returns:
        Whether further characters could turn the prefix into a valid number.
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

    if "e" in text:
        left, _ = text.split("e", 1)
        if left in {"", "-", ".", "-."} and text not in {"e", "e-"}:
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
    """Return whether text is a complete supported number.

    Complete values require at least one digit and cannot end with a sign,
    decimal point, or exponent marker.

    Args:
        text: Candidate numeric value.

    Returns:
        Whether the complete string represents a supported number.
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

    return any(ch.isdigit() for ch in text)
