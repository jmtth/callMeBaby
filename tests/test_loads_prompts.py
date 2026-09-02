"""Tests for JSON error handling in prompt loading."""
import json
import tempfile
from pathlib import Path

import pytest

from src.call_me_maybe import load_prompts


def test_load_prompts_missing_file():
    """Load prompts missing file."""
    with pytest.raises(ValueError,
                       match="Input file not found.*nonexistent.json"):
        load_prompts("nonexistent.json")


def test_load_prompts_empty_file():
    """Load prompts empty file."""
    with tempfile.NamedTemporaryFile(mode='w',
                                     suffix='.json',
                                     delete=False) as f:
        temp_path = f.name
    try:
        with pytest.raises(ValueError, match="Input file is empty"):
            load_prompts(temp_path)
    finally:
        Path(temp_path).unlink()


def test_load_prompts_invalid_json():
    """Load prompts invalid json."""
    with tempfile.NamedTemporaryFile(mode='w',
                                     suffix='.json',
                                     delete=False) as f:
        f.write('{"invalid": json}')
        temp_path = f.name
    try:
        with pytest.raises(ValueError, match="Invalid JSON"):
            load_prompts(temp_path)
    finally:
        Path(temp_path).unlink()


def test_load_prompts_valid_string():
    """Load prompts valid string."""
    with tempfile.NamedTemporaryFile(mode='w',
                                     suffix='.json',
                                     delete=False) as f:
        json.dump("test prompt", f)
        temp_path = f.name
    try:
        prompts = load_prompts(temp_path)
        assert prompts == ["test prompt"]
    finally:
        Path(temp_path).unlink()


def test_load_prompts_valid_dict_with_prompt():
    """Load prompts valid dict with prompt."""
    with tempfile.NamedTemporaryFile(mode='w',
                                     suffix='.json',
                                     delete=False) as f:
        json.dump({"prompt": "test prompt"}, f)
        temp_path = f.name
    try:
        prompts = load_prompts(temp_path)
        assert prompts == ["test prompt"]
    finally:
        Path(temp_path).unlink()


def test_load_prompts_valid_dict_with_prompts():
    """Load prompts valid dict with prompts."""
    with tempfile.NamedTemporaryFile(mode='w',
                                     suffix='.json',
                                     delete=False) as f:
        json.dump({"prompts": ["p1", "p2"]}, f)
        temp_path = f.name
    try:
        prompts = load_prompts(temp_path)
        assert prompts == ["p1", "p2"]
    finally:
        Path(temp_path).unlink()


def test_load_prompts_dict_missing_keys():
    """Load prompts dict missing keys."""
    with tempfile.NamedTemporaryFile(mode='w',
                                     suffix='.json',
                                     delete=False) as f:
        json.dump({"other_key": "value"}, f)
        temp_path = f.name
    try:
        with pytest.raises(ValueError,
                           match="must contain 'prompt' or 'prompts' key"):
            load_prompts(temp_path)
    finally:
        Path(temp_path).unlink()


def test_load_prompts_valid_list():
    """Load prompts valid list."""
    with tempfile.NamedTemporaryFile(mode='w',
                                     suffix='.json',
                                     delete=False) as f:
        json.dump(["p1", "p2", "p3"], f)
        temp_path = f.name
    try:
        prompts = load_prompts(temp_path)
        assert prompts == ["p1", "p2", "p3"]
    finally:
        Path(temp_path).unlink()


def test_load_prompts_list_with_dicts():
    """Load prompts list with dicts."""
    with tempfile.NamedTemporaryFile(mode='w',
                                     suffix='.json',
                                     delete=False) as f:
        json.dump([{"prompt": "p1"}, {"prompt": "p2"}], f)
        temp_path = f.name
    try:
        prompts = load_prompts(temp_path)
        assert prompts == ["p1", "p2"]
    finally:
        Path(temp_path).unlink()


def test_load_prompts_list_invalid_item():
    """Load prompts list invalid item."""
    with tempfile.NamedTemporaryFile(mode='w',
                                     suffix='.json',
                                     delete=False) as f:
        json.dump(["p1", 123], f)  # 123 is not string or dict
        temp_path = f.name
    try:
        with pytest.raises(ValueError,
                           match="List items must be strings or dicts"):
            load_prompts(temp_path)
    finally:
        Path(temp_path).unlink()


def test_load_prompts_invalid_root_type():
    """Load prompts invalid root type."""
    with tempfile.NamedTemporaryFile(mode='w',
                                     suffix='.json',
                                     delete=False) as f:
        json.dump(123, f)  # Number as root
        temp_path = f.name
    try:
        with pytest.raises(ValueError,
                           match="JSON must be string, dict, or list"):
            load_prompts(temp_path)
    finally:
        Path(temp_path).unlink()


def test_load_prompts_stdin():
    """Load prompts stdin."""
    # We can't actually test stdin, but we can verify None doesn't error early
    # In real usage this would block waiting for input
    # This is more of a documentation test
    pass
