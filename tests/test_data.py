from pathlib import Path

import pytest

import abinslib.data


def test_get_data():
    pytest.importorskip("pooch")

    nah = abinslib.data.get_data("NaH.phonon")
    assert isinstance(nah, Path)
    assert nah.is_file()
    assert nah.name == "NaH.phonon"

    with nah.open() as fd:
        header = next(fd).strip()
        assert header == "BEGIN header"


def test_missing_pooch_error(monkeypatch):
    monkeypatch.setattr(abinslib.data, "_EUPHONIC_TEST_DATA", None)

    with pytest.raises(ImportError, match="pooch"):
        abinslib.data.get_data("NaH.phonon")
