from packaging.version import Version

import abinslib


def test_valid_version() -> None:

    assert isinstance(abinslib.__version__, str)

    # Raises InvalidVersion if improperly formatted
    Version(abinslib.__version__)
