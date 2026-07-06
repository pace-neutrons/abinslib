from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

from euphonic import QpointPhononModes, Quantity
import numpy as np
from numpy.random import PCG64, Generator
import pytest

if TYPE_CHECKING:
    from abinslib.displacements import Displacements

test_data = Path(__file__).parent / "data"


def _regression_data(request) -> Path:
    """Get a regression test data directory for the current module

    e.g. "regression_data/test_isotropic_incoherent"

    Strip the leading "test." but allow further nesting.

    """
    name = ".".join(request.module.__name__.split(".")[1:])
    return Path(__file__).parent / f"regression_data/{name}"


@pytest.fixture(scope="module")
def lazy_datadir(request) -> Path:
    return _regression_data(request)


@pytest.fixture(scope="module")
def original_datadir(request) -> Path:
    return _regression_data(request)


@dataclass
class ToscaModes:
    modes: QpointPhononModes
    q2: Quantity

    def __post_init__(self) -> None:
        # lru_cache set up at instance level so it can be cleaned up properly
        self.b = lru_cache()(self._b)
        self.a = lru_cache()(self._a)

    def _b(self, temperature_k: int) -> Displacements:
        from abinslib.displacements import Displacements

        temperature = Quantity(temperature_k, "kelvin")
        return Displacements.from_modes(self.modes, temperature)

    def _a(self, temperature_k: int) -> Quantity:
        return self.b(temperature_k).to_atomic_displacements()

    def ab(self, temperature_k: int) -> tuple[Quantity, Displacements]:
        return self.a(temperature_k), self.b(temperature_k)


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
