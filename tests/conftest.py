import json
from pathlib import Path

import pystac
import pytest

FIXTURES = Path(__file__).parent / "fixtures"


def pytest_configure(config):
    config.addinivalue_line("markers", "network: needs internet / Earthdata")


@pytest.fixture
def mtd_xml():
    return (FIXTURES / "MTD_TL_S2B_14TQL_20220708.xml").read_bytes()


@pytest.fixture
def eco_item():
    return pystac.Item.from_dict(
        json.loads((FIXTURES / "eco_l1ct_item.json").read_text()))
