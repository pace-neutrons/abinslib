from pathlib import Path

import numpy as np

import abinslib

ref_data = Path(__file__).parent / "data"


def test_import() -> None:
    assert isinstance(abinslib.__version__, str)


def test_testdata() -> None:
    datafile = ref_data / "GaSb_abins_isotropic_raw.npz"
    data = np.load(datafile)

    assert "energy" in data.keys()
