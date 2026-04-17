from euphonic import Quantity
import numpy as np
import pytest


from abinslib.displacements import Displacements
from abinslib.almost_isotropic_incoherent import (
    calculate_almost_isotropic_incoherent_fundamentals,
    calculate_almost_isotropic_incoherent_spectra,
)
from abinslib.util import calculate_indirect_q2


@pytest.mark.parametrize("modes", ["GaSb"], indirect=True)
def test_calculate_almost_isotropic_incoherent_fundamentals(modes):
    temperature = Quantity(100, "kelvin")
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
    ("temperature_k", "modes"),
    [(10, "GaSb"), (100, "GaSb"), (10, "ethanol"), (100, "ethanol")],
    indirect=["modes"],
)
def test_calculate_isotropic_incoherent_spectrum(temperature_k, modes):
    """Test almost-isotropic fundamentals"""

    temperature = Quantity(temperature_k, "K")
    bins = Quantity(np.arange(0, 8000, 1), "cm_1")

    b = Displacements.from_modes(modes=modes, temperature=temperature)
    a = b.to_atomic_displacements(crystal=modes.crystal)

    q2 = calculate_indirect_q2(
        modes.frequencies,
        angle=(134.98885653282196 * np.pi / 180),
        final_energy=Quantity(32.0, "cm_1").to("hartree"),
    )

    _ = calculate_almost_isotropic_incoherent_spectra(modes, b, a, q2, bins)
