from itertools import product
from pathlib import Path

from euphonic import Crystal, QpointPhononModes, Quantity
import numpy as np
from numpy.testing import assert_allclose
import pytest

import abinslib.isotropic_incoherent
from abinslib.bose import BoseOccupation
from abinslib.displacements import (
    calculate_atomic_displacements,
    calculate_mode_displacements,
)
from abinslib.isotropic_incoherent import (
    calculate_isotropic_dw_factor,
    calculate_isotropic_incoherent_spectra,
    q_scaling_isotropic_incoherent_spectra,
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


@pytest.fixture
def patch_cross_sections(monkeypatch):
    """Replace Euphonic cross-section lookup with Mantid values"""

    def _get_mantid_total_cross_sections(crystal: Crystal) -> Quantity:
        mantid_data = {
            "Ga": 6.83,
            "Sb": 3.9,
            "C": 5.551,
            "H": 82.02,
            "O": 4.232,
        }

        return Quantity([mantid_data[symbol] for symbol in crystal.atom_type], "barn")

    monkeypatch.setattr(
        abinslib.isotropic_incoherent,
        "_get_total_cross_sections",
        _get_mantid_total_cross_sections,
    )


def test_isotropic_dw(ref_modes):
    """Check isotropic DW on Q2 mesh agrees with Mantid-Abins"""
    gasb_modes = ref_modes["GaSb"]
    dw_data = np.load(test_data / "GaSb_abins_isotropic_dw.npz")
    q2 = Quantity(dw_data["q2"], "angstrom^-2")

    a = calculate_atomic_displacements(gasb_modes, temperature=Quantity(100, "kelvin"))

    binned_dw_factor = calculate_isotropic_dw_factor(a, q2[None, :])
    assert_allclose(binned_dw_factor[0], dw_data["iso_dw"].transpose(), rtol=1e-6)


@pytest.mark.parametrize(
    "temperature_k,system", product([10, 100], ["GaSb", "ethanol"])
)
def test_calculate_isotropic_incoherent_spectrum(
    temperature_k, ref_modes, system, patch_cross_sections
):
    """Test reference method for fully-isotropic calculation

    Note that there is some difference from Mantid-Abins reference because that
    implementation applies DW and Q2 scaling after initial energy binning
    """
    modes = ref_modes[system]

    temperature = Quantity(temperature_k, "K")
    ref_data = np.load(test_data / f"{system}_abins_{temperature_k}k_isotropic_raw.npz")

    bins = Quantity(ref_data["energy"], str(ref_data["energy_unit"]))
    bin_width = bins[1] - bins[0]

    ref_intensity = ref_data["intensity"]

    b = calculate_mode_displacements(
        modes, temperature=temperature, occupation=BoseOccupation.N_PLUS_ONE
    )
    a = calculate_atomic_displacements(
        modes,
        temperature=temperature,
    )

    # Q2 calculated at exact Mantid-Abins TOSCA backscattering angle
    q2 = Quantity(np.load(test_data / f"{system}_modes_q2.npy"), "angstrom^-2")

    spectra = calculate_isotropic_incoherent_spectra(modes, b, a, q2, bins)
    spectrum = spectra.sum()

    assert_allclose(
        spectrum.y_data.magnitude,
        ref_intensity[0] / bin_width.magnitude,
        rtol=1e-2,
    )


@pytest.mark.parametrize(
    "temperature_k,system", product([10, 100], ["GaSb", "ethanol"])
)
def test_q_scaling_isotropic_incoherent_spectrum(
    temperature_k, ref_modes, system, patch_cross_sections
):
    """Validate fully-isotropic calculation against Mantid-Abins data

    This implementation follows the scheme of rescaling bins for DW/Q2 terms

    The difference with Mantid-Abins reference is smaller than exact
    calculation, if still larger than expected
    """

    modes = ref_modes[system]

    temperature = Quantity(temperature_k, "K")
    ref_data = np.load(test_data / f"{system}_abins_{temperature_k}k_isotropic_raw.npz")

    bins = Quantity(ref_data["energy"], str(ref_data["energy_unit"]))
    bin_width = bins[1] - bins[0]

    ref_intensity = ref_data["intensity"]

    b = calculate_mode_displacements(
        modes, temperature=temperature, occupation=BoseOccupation.N_PLUS_ONE
    )
    a = calculate_atomic_displacements(
        modes,
        temperature=temperature,
    )
    q2 = Quantity(np.load(test_data / "abins-q2-1_4-dump.npy"), "Å^-2")

    spectra = q_scaling_isotropic_incoherent_spectra(modes, b, a, q2, bins)
    spectrum = spectra.sum()

    assert_allclose(
        spectrum.y_data.magnitude,
        ref_intensity[0] / bin_width.magnitude,
        rtol=1e-8,
    )
