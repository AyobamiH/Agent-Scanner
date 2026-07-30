"""Guard test for empty string_utils module."""


def test_string_utils_exports():
    """string_utils should expose empty __all__ by default."""

    import src.utils.string_utils as su

    assert su.__all__ == []
