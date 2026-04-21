from typing import NamedTuple
from pathlib import Path

from euphonic import QpointPhononModes, Quantity
import numpy as np
from numpy.random import Generator, PCG64
import pytest

test_data = Path(__file__).parent / "data"


class ToscaModes(NamedTuple):
    modes: QpointPhononModes
    q2: Quantity


def _get_modes(name: str) -> QpointPhononModes:
    return QpointPhononModes.from_json_file(
        str(test_data / f"{name}_qpoint_phonon_modes.json")
    )

def _get_q2(name: str) -> Quantity:
    q2 = np.load(test_data / f"{name}_modes_q2.npy")
    return Quantity(q2, "angstrom^-2")


@pytest.fixture(scope="session")
def modes(request):
    """Precalculated phonon modes"""
    return _get_modes(request.param)


@pytest.fixture(scope="session")
def ref_npz(request):
    """Reference data dict from npz file"""
    return np.load(test_data / request.param)


@pytest.fixture(scope="session")
def tosca_modes(request):
    """Phonon modes with nominal Q2 values for Mantid-like TOSCA"""
    return ToscaModes(
        _get_modes(request.param),
        _get_q2(request.param),
    )


@pytest.fixture
def rng(request):
    """A numpy Generator instance

    This fixture allows a seed to be specified as indirect parametrization, but
    for most purposes a fixed default should be fine.

    If a test requires more than one rng, use the spawn() method to create
    children from this one.
    """
    seed = getattr(request, "param", 1234)
    return Generator(PCG64(seed))
