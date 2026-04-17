from itertools import product
from pathlib import Path

from euphonic import QpointPhononModes, Quantity
import numpy as np
import pytest


from abinslib.displacements import Displacements
from abinslib.almost_isotropic_incoherent import (
    calculate_almost_isotropic_incoherent_fundamentals,
    calculate_almost_isotropic_incoherent_spectra,
)
from abinslib.util import calculate_indirect_q2

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


def test_calculate_almost_isotropic_incoherent_fundamentals(ref_modes):
    temperature = Quantity(100, "kelvin")
    modes = ref_modes["GaSb"]
    b = Displacements.from_modes(modes, temperature=temperature)
    a = b.to_atomic_displacements()

    q2 = calculate_indirect_q2(
        modes.frequencies,
        angle=(134.98885653282196 * np.pi / 180),
        final_energy=Quantity(32.0, "cm_1").to("hartree"),
    )

    _ = calculate_almost_isotropic_incoherent_fundamentals(
        mode_displacements=b, atomic_displacements=a, nominal_q2=q2
    )


@pytest.mark.parametrize(
    ("temperature_k", "system"), product([10, 100], ["GaSb", "ethanol"])
)
def test_calculate_isotropic_incoherent_spectrum(
    temperature_k,
    ref_modes,
    system,
):
    """Test almost-isotropic fundamentals"""
    modes = ref_modes[system]

    temperature = Quantity(temperature_k, "K")
    ref_data = np.load(test_data / f"{system}_abins_{temperature_k}k_isotropic_raw.npz")

    bins = Quantity(ref_data["energy"], str(ref_data["energy_unit"]))

    b = Displacements.from_modes(modes=modes, temperature=temperature)
    a = b.to_atomic_displacements(crystal=modes.crystal)

    # Q2 calculated at exact Mantid-Abins TOSCA backscattering angle
    q2 = Quantity(np.load(test_data / f"{system}_modes_q2.npy"), "angstrom^-2")

    _ = calculate_almost_isotropic_incoherent_spectra(modes, b, a, q2, bins)
