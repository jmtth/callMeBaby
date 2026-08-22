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
