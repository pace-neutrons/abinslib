"""Wrapper to fetch reference data for tutorials etc."""

from pathlib import Path

import pooch

_EUPHONIC_TEST_DATA = pooch.create(
    path=pooch.os_cache("abinslib"),
    # Currently this location is overridden by the only file in registry.
    # Later it will point to something like github release or Zenodo archive,
    # where most ref data can be kept in flat structure with short names.
    base_url=(
        "https://github.com/pace-neutrons/Euphonic/raw/"
        "master/tests_and_analysis/test/data/"
    ),
    registry={
        "NaH.phonon": (
            "ccb30647b5cc9a2f3ab470dda77bc6f3ccc19cb1b8adaf35f5c40ccdaabccde1"
        )
    },
    urls={
        "NaH.phonon": (
            "https://github.com/pace-neutrons/Euphonic/raw/master/"
            "tests_and_analysis/test/data/castep_files/NaH/NaH.phonon"
        )
    },
)


def get_data(filename: str) -> Path:
    """Get external reference data by filename."""
    return Path(_EUPHONIC_TEST_DATA.fetch(filename))
