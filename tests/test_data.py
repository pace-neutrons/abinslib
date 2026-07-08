from pathlib import Path
import sys

import pytest

import abinslib.data


def test_get_data():
    """If pooch is available, check that reference data is loaded correctly"""
    pytest.importorskip("pooch")

    nah = abinslib.data.get_data("NaH.phonon")
    assert isinstance(nah, Path)
    assert nah.is_file()
    assert nah.name == "NaH.phonon"

    with nah.open() as fd:
        header = next(fd).strip()
        assert header == "BEGIN header"


def test_pooch_import_handler(monkeypatch):
    """Check _EUPHONIC_TEST_DATA would be set to None if no pooch available"""
    monkeypatch.setitem(sys.modules, "pooch", None)

    assert abinslib.data._ref_data_or_none() is None


def test_missing_pooch_error(monkeypatch):
    """Check for correct error message if no pooch available"""
    monkeypatch.setattr(abinslib.data, "_EUPHONIC_TEST_DATA", None)

    with pytest.raises(ImportError, match="pooch"):
        abinslib.data.get_data("NaH.phonon")
