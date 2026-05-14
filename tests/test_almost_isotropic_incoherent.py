from euphonic import Quantity
import numpy as np
import pytest

from abinslib.almost_isotropic_incoherent import (
    calculate_almost_isotropic_incoherent_combinations,
    calculate_almost_isotropic_incoherent_fundamentals,
    calculate_almost_isotropic_incoherent_spectra,
    q_scaling_almost_isotropic_incoherent_combination_spectra,
)
from abinslib.displacements import Displacements


@pytest.mark.parametrize("tosca_modes", ["GaSb"], indirect=True)
def test_calculate_almost_isotropic_incoherent_fundamentals(
    tosca_modes, ndarrays_regression
):
    temperature = Quantity(100, "kelvin")
    b = Displacements.from_modes(tosca_modes.modes, temperature=temperature)
    a = b.to_atomic_displacements()

    intensities = calculate_almost_isotropic_incoherent_fundamentals(
        mode_displacements=b, atomic_displacements=a, nominal_q2=tosca_modes.q2
    )
    ndarrays_regression.check({"intensities": intensities})


@pytest.mark.parametrize(
    ("temperature_k", "tosca_modes"),
    [
        (10, "GaSb"),
        (100, "GaSb"),
        (10, "ethanol"),
        (100, "ethanol"),
    ],
    indirect=["tosca_modes"],
)
def test_calculate_isotropic_incoherent_spectrum(
    temperature_k, tosca_modes, ndarrays_regression
):
    """Test almost-isotropic fundamentals"""
    modes, q2 = tosca_modes

    temperature = Quantity(temperature_k, "K")
    bins = Quantity(np.arange(0, 8000, 1), "cm_1")

    b = Displacements.from_modes(modes=modes, temperature=temperature)
    a = b.to_atomic_displacements()

    spectra = calculate_almost_isotropic_incoherent_spectra(modes, b, a, q2, bins)

    ndarrays_regression.check(
        {
            "x_data": spectra.x_data.magnitude,
            "y_data": spectra.y_data.magnitude,
            "x_data_unit": spectra.x_data_unit,
            "y_data_unit": spectra.y_data_unit,
        }
    )


@pytest.mark.parametrize("modes", ["GaSb"], indirect=True)
def test_almost_isotropic_incoherent_combinations(modes):
    from abinslib.util import calculate_indirect_q2

    combination_frequencies = (
        modes.frequencies[:, :, None, None] + modes.frequencies[None, None, :, :]
    )

    q2 = calculate_indirect_q2(
        combination_frequencies,
        angle=(134.98885653282196 * np.pi / 180),
        final_energy=Quantity(32.0, "cm_1").to("hartree"),
    )

    temperature = Quantity(100, "kelvin")
    b = Displacements.from_modes(modes=modes, temperature=temperature)
    a = b.to_atomic_displacements()

    _ = calculate_almost_isotropic_incoherent_combinations(
        b, a, q2
    )

@pytest.mark.parametrize("modes", ["GaSb"], indirect=True)
def test_q_scaling_combination_spectra(modes):
    from abinslib.util import calculate_indirect_q2

    bins = Quantity(np.arange(0, 8000, 1), "cm_1")
    bin_centres = (bins[1:] + bins[:-1]) * 0.5
    nominal_q2 = calculate_indirect_q2(
        bin_centres,
        angle=(134.98885653282196 * np.pi / 180),
        final_energy=Quantity(32.0, "cm_1").to("hartree"),
    )

    temperature = Quantity(100, "kelvin")
    b = Displacements.from_modes(modes=modes, temperature=temperature)
    a = b.to_atomic_displacements()

    _ = q_scaling_almost_isotropic_incoherent_combination_spectra(
        modes,
        b,
        a,
        nominal_q2,
        bins,
        apply_cross_section=True
    )
