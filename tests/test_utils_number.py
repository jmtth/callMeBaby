from src import utils

def test_extract_decimal_counts():
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

