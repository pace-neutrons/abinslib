from pathlib import Path

from euphonic import QpointPhononModes, Quantity
import numpy as np
from numpy.random import Generator, PCG64
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


@pytest.fixture(scope="session")
def tosca_q2(request):
    """Nominal Q2 values corresponding to a 'modes' for Mantid-like TOSCA"""
    q2 = np.load(test_data / f"{request.param}_modes_q2.npy")
    return Quantity(q2, "angstrom^-2")


@pytest.fixture(scope="function")
def rng(request):
    """A numpy Generator instance

    This fixture allows a seed to be specified as indirect parametrization, but
    for most purposes a fixed default should be fine.

    If a test requires more than one rng, use the spawn() method to create
    children from this one.
    """
    seed = getattr(request, "param", 1234)
    return Generator(PCG64(seed))
