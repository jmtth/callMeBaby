def is_number_terminator_token(token_text: str) -> bool:
    """Return whether token text is a valid number terminator."""
    return token_text in {" ", "  ", ",", ", ", " ,", "}"}
