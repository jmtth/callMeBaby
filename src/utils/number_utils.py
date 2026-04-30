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
