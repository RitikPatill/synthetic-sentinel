import pytest
from pathlib import Path

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def real_jpeg():
    return FIXTURES / "real.jpg"


@pytest.fixture
def synthetic_jpeg():
    return FIXTURES / "synthetic.jpg"
