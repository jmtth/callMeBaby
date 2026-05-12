from src import utils


def test_is_number_terminator_token():
    """Test the is_number_terminator_token function with various inputs."""
    assert utils.is_number_terminator_token(" ") is True
    assert utils.is_number_terminator_token("  ") is True
    assert utils.is_number_terminator_token(",") is True
    assert utils.is_number_terminator_token(", ") is True
    assert utils.is_number_terminator_token(" ,") is True
    assert utils.is_number_terminator_token("}") is True
    assert utils.is_number_terminator_token(".") is False
    assert utils.is_number_terminator_token("a") is False


def test_get_number_token_ids():
    """Test the get_number_token_ids function with
    a sample token_to_id mapping.
    """
    token_to_id = {
        "1": 0,
        "2": 1,
        ".": 2,
        "e": 3,
        "-": 4,
        "Ġ ": 5,   # space token
        "a": 6,
        "Ġa": 7,   # space + a
        "ĠĠ": 8,   # multiple spaces
        "Ġ.": 9,   # space + dot
        "Ġe": 10,  # space + e
        "Ġ-": 11,  # space + dash
        "Ġ1": 12,  # space + 1
        "": 13,    # empty token
    }
    number_token_ids = utils.get_number_token_ids(token_to_id)
    assert number_token_ids == {0, 1, 2, 3, 4}
