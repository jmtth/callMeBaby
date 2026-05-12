from src import utils


def test_extract_decimal_counts():
    """Test the extract_decimal_counts function with various inputs."""
    assert utils.extract_decimal_counts("123.45") == [2]
    assert utils.extract_decimal_counts("123.45 123.3") == [2, 1]
    assert utils.extract_decimal_counts("0.001") == [3]
    assert utils.extract_decimal_counts("100") == [1]
    assert utils.extract_decimal_counts(".5") == [1]
    assert utils.extract_decimal_counts("123.") == [1]
    assert utils.extract_decimal_counts("abc") == []
    assert utils.extract_decimal_counts("-12.34e-5") == [2, 1]
    assert utils.extract_decimal_counts("-.5") == [1]
    assert utils.extract_decimal_counts("-.") == []
    assert utils.extract_decimal_counts("-.e-5") == [1]
    assert utils.extract_decimal_counts("-.5e-5") == [1, 1]
    assert utils.extract_decimal_counts("fjhd.fj") == []


def test_is_valid_number_fragment():
    """Test the is_valid_number_fragment function with various inputs."""
    assert utils.is_valid_number_fragment("") is True
    assert utils.is_valid_number_fragment("123") is True
    assert utils.is_valid_number_fragment("-123") is True
    assert utils.is_valid_number_fragment("1e1") is True
    assert utils.is_valid_number_fragment("-1e-") is True
    assert utils.is_valid_number_fragment("3.") is True
    assert utils.is_valid_number_fragment(".5") is True
    assert utils.is_valid_number_fragment("-3.") is True
    assert utils.is_valid_number_fragment("-.5") is True
    assert utils.is_valid_number_fragment("-123.") is True
    assert utils.is_valid_number_fragment("1e") is True
    assert utils.is_valid_number_fragment("1e-") is True
    assert utils.is_valid_number_fragment("1e+") is False
    assert utils.is_valid_number_fragment(".e10") is False
    assert utils.is_valid_number_fragment("-.e10") is False
    assert utils.is_valid_number_fragment("abc") is False
    assert utils.is_valid_number_fragment("1.2.3") is False
    assert utils.is_valid_number_fragment("1e2.3") is False
    assert utils.is_valid_number_fragment("1e-2.3") is False
    assert utils.is_valid_number_fragment("1e-2e3") is False
    assert utils.is_valid_number_fragment("--1") is False
    assert utils.is_valid_number_fragment("1-2") is False
    assert utils.is_valid_number_fragment("1e-2-3") is False


def test_is_complete_number():
    """Test the is_complete_number function with various inputs."""
    assert utils.is_complete_number("123") is True
    assert utils.is_complete_number("123.45") is True
    assert utils.is_complete_number("-123.45") is True
    assert utils.is_complete_number("1e10") is True
    assert utils.is_complete_number("-1e-10") is True
    assert utils.is_complete_number("1.5e+10") is True
    assert utils.is_complete_number("123.") is False
    assert utils.is_complete_number("-123.") is False
    assert utils.is_complete_number("1e") is False
    assert utils.is_complete_number("1e-") is False
    assert utils.is_complete_number(".e10") is False
    assert utils.is_complete_number("-.e10") is False
    assert utils.is_complete_number("-") is False
    assert utils.is_complete_number(".") is False
    assert utils.is_complete_number("-.") is False
    assert utils.is_complete_number("1e+") is False
    assert utils.is_complete_number("") is False
