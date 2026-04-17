from euphonic import Quantity
import numpy as np
import pytest


from abinslib.displacements import Displacements
from abinslib.almost_isotropic_incoherent import (
    calculate_almost_isotropic_incoherent_fundamentals,
    calculate_almost_isotropic_incoherent_spectra,
)


@pytest.mark.parametrize(("modes", "tosca_q2"), [("GaSb", "GaSb")], indirect=True)
def test_calculate_almost_isotropic_incoherent_fundamentals(
    modes, tosca_q2, ndarrays_regression
):
    temperature = Quantity(100, "kelvin")
    b = Displacements.from_modes(modes, temperature=temperature)
    a = b.to_atomic_displacements()

    intensities = calculate_almost_isotropic_incoherent_fundamentals(
        mode_displacements=b, atomic_displacements=a, nominal_q2=tosca_q2
    )
    ndarrays_regression.check({"intensities": intensities})


@pytest.mark.parametrize(
    ("temperature_k", "modes", "tosca_q2"),
    [
        (10, "GaSb", "GaSb"),
        (100, "GaSb", "GaSb"),
        (10, "ethanol", "ethanol"),
        (100, "ethanol", "ethanol"),
    ],
    indirect=["modes", "tosca_q2"],
)
def test_calculate_isotropic_incoherent_spectrum(
    temperature_k, modes, tosca_q2, ndarrays_regression
):
    """Test almost-isotropic fundamentals"""

    temperature = Quantity(temperature_k, "K")
    bins = Quantity(np.arange(0, 8000, 1), "cm_1")

    b = Displacements.from_modes(modes=modes, temperature=temperature)
    a = b.to_atomic_displacements(crystal=modes.crystal)

    spectra = calculate_almost_isotropic_incoherent_spectra(modes, b, a, tosca_q2, bins)

    ndarrays_regression.check(
        {
            "x_data": spectra.x_data.magnitude,
            "y_data": spectra.y_data.magnitude,
            "x_data_unit": spectra.x_data_unit,
            "y_data_unit": spectra.y_data_unit,
        }
    )
