from src import utils


def test_is_number_terminator_token():
    """Is number terminator token."""
    assert utils.is_number_terminator_token(" ") is True
    assert utils.is_number_terminator_token("  ") is True
    assert utils.is_number_terminator_token(",") is True
    assert utils.is_number_terminator_token(", ") is True
    assert utils.is_number_terminator_token(" ,") is True
    assert utils.is_number_terminator_token("}") is True
    assert utils.is_number_terminator_token(".") is False
    assert utils.is_number_terminator_token("a") is False
