from pathlib import Path
import warnings

from euphonic import QpointPhononModes, Quantity
import numpy as np
from numpy.testing import assert_allclose

from abinslib.isotropic_incoherent import (
    BoseOccupation,
    calculate_atomic_displacements,
    calculate_bose_factor,
    calculate_mode_displacements,
    calculate_isotropic_incoherent_fundamentals,
    calculate_isotropic_incoherent_spectra,
)
from abinslib.util import calculate_indirect_q2


test_data = Path(__file__).parent / "data"


def test_calculate_adp():
    """Check ADP agrees with Euphonic implementation"""

    modes = QpointPhononModes.from_json_file(
        test_data / "GaSb_qpoint_phonon_modes.json"
    )

    dw = calculate_atomic_displacements(modes, temperature=Quantity(100, "K"))

    ref_dw = modes.calculate_debye_waller(
        temperature=Quantity(100, "K"),
        frequency_min=Quantity(0.01, "meV"),
        symmetrise=False,
    )

    assert dw.temperature == ref_dw.temperature
    assert dw.debye_waller.units == ref_dw.debye_waller.units
    assert_allclose(dw.debye_waller.magnitude, ref_dw.debye_waller.magnitude)


def test_calculate_isotropic_incoherent_fundamentals():
    modes = QpointPhononModes.from_json_file(
        test_data / "GaSb_qpoint_phonon_modes.json"
    )
    b = calculate_mode_displacements(modes, temperature=Quantity(100, "K"))
    a = calculate_atomic_displacements(
        modes, temperature=Quantity(100, "K"), mode_displacements=b
    )

    calculate_isotropic_incoherent_fundamentals(
        modes, b, a, Quantity(np.ones_like(modes.frequencies), "angstrom^-2")
    )
    # No reference data yet, just check it doesn't raise an error!


def test_calculate_isotropic_incoherent_spectrum():
    temperature = Quantity(10, "K")
    ref_data = np.load(test_data / "GaSb_abins_isotropic_raw.npz")

    bins = Quantity(ref_data["energy"], str(ref_data["energy_unit"]))
    ref_intensity = ref_data["intensity"]

    modes = QpointPhononModes.from_json_file(
        test_data / "GaSb_qpoint_phonon_modes.json"
    )
    b = calculate_mode_displacements(modes, temperature=temperature)
    a = calculate_atomic_displacements(
        modes, temperature=temperature, mode_displacements=b
    )

    # Those mode displacements are with 2n+1 statistics, but we want n+1
    two_n_plus_one = calculate_bose_factor(
        modes.frequencies,
        temperature,
        BoseOccupation.TWO_N_PLUS_ONE,
    )
    n_plus_one = two_n_plus_one * 0.5 + 0.5
    b = b * (n_plus_one / two_n_plus_one)[:, :, None, None, None]

    q2 = calculate_indirect_q2(
        modes.frequencies,
        angle=(134.98885653282196 * np.pi / 180),
        final_energy=Quantity(32.0, "1/cm").to("meV"),
    )

    spectra = calculate_isotropic_incoherent_spectra(modes, b, a, q2, bins)

    spectrum = spectra.sum()

    scale_offset = (
        (spectrum.y_data.sum().magnitude / ref_intensity[0].sum())
    )
    if not (0.95 < scale_offset < 1.05):
        msg = f"Overall magnitude different from AbINS: {scale_offset * 100:.1f}%"
        warnings.warn(msg)

    # Compare normalized spectra for now
    assert_allclose(
        spectrum.y_data.magnitude,
        ref_intensity[0] * scale_offset,
        rtol=0.01,
    )
