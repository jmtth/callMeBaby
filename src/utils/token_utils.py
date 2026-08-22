def is_number_terminator_token(token_text: str) -> bool:
    """Check if the token text is a valid number terminator
    (whitespace, commas, braces).

    args:
        str : a token string.

    returns:
        bool: true if the token is a valid terminator.
    """
    return token_text in {" ", "  ", ",", ", ", " ,", "}"}
