from pathlib import Path

from euphonic import QpointPhononModes
import numpy as np
import pytest

test_data = Path(__file__).parent / "data"


@pytest.fixture(scope="session")
def modes(request):
    """Precalculated phonon modes"""
    name = request.param
    return QpointPhononModes.from_json_file(
        str(test_data / f"{name}_qpoint_phonon_modes.json")
    )


@pytest.fixture(scope="session")
def ref_npz(request):
    """Reference data dict from npz file"""
    return np.load(test_data / request.param)
