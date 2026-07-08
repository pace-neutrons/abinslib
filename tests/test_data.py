from pathlib import Path

from abinslib.data import get_data

def test_get_data():
    nah = get_data("NaH.phonon")
    assert isinstance(nah, Path)
    assert nah.is_file()
    assert nah.name == "NaH.phonon"

    with nah.open() as fd:
        header = next(fd).strip()
        assert header == "BEGIN header"
