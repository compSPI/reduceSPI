"""Test offline models."""

from reduceSPI.offline import placeholder


class TestOffline:
    """Test functions from the offline.py module."""

    @staticmethod
    def test_placeholder():
        """Test placeholder."""
        result = placeholder()
        expected = 3
        assert result == expected
