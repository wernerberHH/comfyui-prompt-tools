"""Unit tests for comfyui_prompt_tools.random_pools."""

import pytest

from comfyui_prompt_tools.random_pools import (
    DEFAULT_AGE_POOL,
    DEFAULT_ETHNICITY_POOL,
    pick_age,
    pick_from_pool,
)


@pytest.mark.unit
def test_pick_from_pool_empty_returns_none():
    assert pick_from_pool("") is None
    assert pick_from_pool("   ") is None
    assert pick_from_pool(None) is None


@pytest.mark.unit
def test_pick_from_pool_single_item():
    assert pick_from_pool("only-one") == "only-one"


@pytest.mark.unit
def test_pick_from_pool_comma_split():
    """Commas split items, not group them."""
    result = pick_from_pool("a, b, c")
    assert result in {"a", "b", "c"}


@pytest.mark.unit
def test_pick_from_pool_newline_split():
    result = pick_from_pool("a\nb\nc")
    assert result in {"a", "b", "c"}


@pytest.mark.unit
def test_pick_from_pool_strips_whitespace():
    result = pick_from_pool("  spaces  ")
    assert result == "spaces"


@pytest.mark.unit
def test_pick_from_pool_default_ethnicity_returns_known_value():
    """The default pool yields one of its known entries."""
    result = pick_from_pool(DEFAULT_ETHNICITY_POOL)
    assert result in {
        "European", "East Asian", "South Asian", "Southeast Asian",
        "Latin American", "African", "Middle Eastern", "Mixed heritage",
    }


@pytest.mark.unit
def test_pick_age_range_format():
    """`min-max` produces an int in [min, max]."""
    for _ in range(20):
        age = pick_age("18-22")
        assert isinstance(age, int)
        assert 18 <= age <= 22


@pytest.mark.unit
def test_pick_age_default_is_in_range():
    age = pick_age(DEFAULT_AGE_POOL)  # "22-45"
    assert isinstance(age, int)
    assert 22 <= age <= 45


@pytest.mark.unit
def test_pick_age_pool_format():
    """Comma-separated age pool — picks one of the values as string."""
    age = pick_age("18, 21, 35")
    # Returned as string because the function only converts ranges
    assert age in {"18", "21", "35"}


@pytest.mark.unit
def test_pick_age_empty_returns_none():
    assert pick_age("") is None
    assert pick_age("   ") is None
