"""
Unit and regression test for the melange package.
"""

# Import package, test suite, and other packages as needed
import melange
import pytest
import sys

def test_melange_imported():
    """Sample test, will always pass so long as import statement worked"""
    assert "melange" in sys.modules
