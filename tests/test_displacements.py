"""Unit tests for abinslib.displacements module"""

from pathlib import Path

from euphonic import QpointPhononModes, Quantity
import numpy as np
from numpy.testing import assert_allclose
import pytest

from abinslib.bose import BoseOccupation
from abinslib.isotropic_incoherent import (
    calculate_atomic_displacements,
    calculate_mode_displacements,
)

test_data = Path(__file__).parent / "data"


@pytest.fixture(scope="module")
def ref_modes() -> dict[str, QpointPhononModes]:
    """Precalculated phonon modes, by name"""
    return {
        name: QpointPhononModes.from_json_file(
            str(test_data / f"{name}_qpoint_phonon_modes.json")
        )
        for name in ["GaSb", "ethanol"]
    }


def test_calculate_adp(ref_modes):
    """Check ADP agrees with Euphonic and Abins implementations

    Euphonic reference is calculated on-the-fly

    Abins reference average_a_traces are from Abins calculate_isotropic_dw
    method, which takes weighted sum over traces at each k-point. Note that
    there seems to be a difference in scale convention: Abins value is twice
    as large.

    i.e. in Euphonic coherent Debye-Waller term exp(-W) = exp(-Q^2.<dw>),
    in Mantid Abins incoherent term exp(-2W) = exp(-Q^2 tr(A)/3)
    - the 2 in exponent has been absorbed into A

    """
    gasb_modes = ref_modes["GaSb"]

    dw = calculate_atomic_displacements(gasb_modes, temperature=Quantity(100, "K"))

    euphonic_dw = gasb_modes.calculate_debye_waller(
        temperature=Quantity(100, "K"),
        frequency_min=Quantity(0.01, "meV"),
        symmetrise=False,
    )

    assert dw.temperature == euphonic_dw.temperature
    assert dw.debye_waller.units == euphonic_dw.debye_waller.units
    assert_allclose(dw.debye_waller.magnitude, euphonic_dw.debye_waller.magnitude)

    # Values from abins isotropic DW calculation
    abins_average_a_traces = np.array([0.01321831, 0.01127088])

    assert_allclose(
        np.trace(dw.debye_waller.to("angstrom^2").magnitude, axis1=1, axis2=2),
        abins_average_a_traces / 2,
        atol=1e-8,
    )


def test_a_abins_ref(ref_modes) -> None:
    """Check calculated A against Abins isotropic calculation

    The reference average_a_traces are from Abins calculate_isotropic_dw method
    which takes weighted sum over traces at each k-point.

    """
    gasb_modes = ref_modes["GaSb"]
    ref_a_traces = np.load(test_data / "GaSb_abins_isotropic_dw.npz")["a_traces"]

    dw = calculate_atomic_displacements(gasb_modes, temperature=Quantity(100, "K"))
    assert_allclose(
        np.trace(dw.debye_waller.to("angstrom^2").magnitude, axis1=1, axis2=2),
        ref_a_traces / 2,
        atol=1e-8,
    )


@pytest.mark.parametrize("temperature_k", [0, 100])
def test_displacements_abins_ref(temperature_k, ref_modes) -> None:
    """Check calculated displacements against Mantid-Abins reference

    Note that as in ADP there seems to be a factor two difference as Mantid
    implementation has absorbed the "2" to construct 2W when summing over B

    """
    gasb_modes = ref_modes["GaSb"]
    b = calculate_mode_displacements(
        gasb_modes,
        temperature=Quantity(temperature_k, "kelvin"),
        occupation=BoseOccupation.N_PLUS_ONE,
    )

    ref_b = np.load(test_data / f"GaSb_abins_{temperature_k}k_B.npz")

    assert_allclose(
        b.to("angstrom^2").magnitude[1],
        np.swapaxes(ref_b["qpt-1"], 0, 1),
    )
