from src import utils


def test_get_repeating_pattern():
    """Get repeating pattern."""
    assert utils.get_repeating_pattern(
        "catcatcat", min_len=3, max_repeats=3) == "cat"
    assert utils.get_repeating_pattern(
        "abcabc", min_len=3, max_repeats=2) == "abc"
    assert utils.get_repeating_pattern(
        "abcab", min_len=3, max_repeats=2) == ""
    assert utils.get_repeating_pattern(
        "aaaaaa", min_len=1, max_repeats=6) == "a"
    assert utils.get_repeating_pattern(
        "ababab", min_len=2, max_repeats=3) == "ab"
    assert utils.get_repeating_pattern("") == ""
    assert utils.get_repeating_pattern(
        "qwertyu", min_len=3, max_repeats=2) == ""
