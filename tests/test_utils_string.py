from src import utils
from llm_sdk import Small_LLM_Model
from torch import Tensor
from typing import cast


class FakeModel(Small_LLM_Model):
    def encode(self, text) -> Tensor:
        # return a value compatible with the base class return type for typing
        return cast(Tensor, [[ord(c) for c in text]])


def test_get_repeating_pattern():
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


def test_remove_repeating_pattern():
    model = FakeModel()
    response = [ord(c) for c in "catcatcat"]
    pattern = "cat"
    new_response = utils.remove_repeating_pattern(model, response, pattern)
    assert new_response == [ord(c) for c in "catcat"]

    response = [ord(c) for c in "abcabc"]
    pattern = "abc"
    new_response = utils.remove_repeating_pattern(model, response, pattern)
    assert new_response == [ord(c) for c in "abc"]

    response = [ord(c) for c in "abcab"]
    pattern = "abc"
    new_response = utils.remove_repeating_pattern(model, response, pattern)
    assert new_response == [ord(c) for c in "abcab"]

    response = [ord(c) for c in "aaaaaa"]
    pattern = "a"
    new_response = utils.remove_repeating_pattern(model, response, pattern)
    assert new_response == [ord(c) for c in "aaaaa"]
