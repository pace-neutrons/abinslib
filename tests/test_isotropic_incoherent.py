from pathlib import Path
import warnings

from euphonic import QpointPhononModes, Quantity
import numpy as np
from numpy.testing import assert_allclose
import pytest

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

    modes = QpointPhononModes.from_json_file(
        str(test_data / "GaSb_qpoint_phonon_modes.json")
    )

    dw = calculate_atomic_displacements(modes, temperature=Quantity(100, "K"))

    euphonic_dw = modes.calculate_debye_waller(
        temperature=Quantity(100, "K"),
        frequency_min=Quantity(0.01, "meV"),
        symmetrise=False,
    )

    assert dw.temperature == euphonic_dw.temperature
    assert dw.debye_waller.units == euphonic_dw.debye_waller.units
    assert_allclose(dw.debye_waller.magnitude, euphonic_dw.debye_waller.magnitude)

    # Values from abins isotropic DW calculation
    abins_average_a_traces = np.array([0.01321831, 0.01127088])

    assert_allclose(np.trace(dw.debye_waller.to("angstrom^2").magnitude, axis1=1, axis2=2),
                    abins_average_a_traces / 2,
                    atol=1e-8)


def test_calculate_isotropic_incoherent_fundamentals():
    modes = QpointPhononModes.from_json_file(
        str(test_data / "GaSb_qpoint_phonon_modes.json")
    )
    b = calculate_mode_displacements(modes, temperature=Quantity(100, "K"))
    a = calculate_atomic_displacements(
        modes, temperature=Quantity(100, "K"), mode_displacements=b
    )

    calculate_isotropic_incoherent_fundamentals(
        modes, b, a, Quantity(np.ones_like(modes.frequencies), "angstrom^-2")
    )
    # No reference data yet, just check it doesn't raise an error!


@pytest.mark.parametrize('temperature_k', [10, 100])
def test_calculate_isotropic_incoherent_spectrum(temperature_k):
    temperature = Quantity(temperature_k, "K")
    ref_data = np.load(test_data / f"GaSb_abins_{temperature_k}k_isotropic_raw.npz")

    bins = Quantity(ref_data["energy"], str(ref_data["energy_unit"]))
    ref_intensity = ref_data["intensity"]

    modes = QpointPhononModes.from_json_file(
        str(test_data / "GaSb_qpoint_phonon_modes.json")
    )

    n_plus_one_b = calculate_mode_displacements(
        modes, temperature=temperature, occupation=BoseOccupation.N_PLUS_ONE)
    two_n_plus_one_b = calculate_mode_displacements(
        modes, temperature=temperature, occupation=BoseOccupation.TWO_N_PLUS_ONE)
    a = calculate_atomic_displacements(
        modes, temperature=temperature, mode_displacements=two_n_plus_one_b
    )

    q2 = calculate_indirect_q2(
        modes.frequencies,
        angle=(134.98885653282196 * np.pi / 180),
        final_energy=Quantity(32.0, "1/cm").to("meV"),
    )

    spectra = calculate_isotropic_incoherent_spectra(modes, n_plus_one_b, a, q2, bins)

    spectrum = spectra.sum()

    # TODO This is still a little mysterious, investigate further.
    # We seem to differ by a factor ~4?
    scale_offset = (
        spectrum.y_data.sum().magnitude / ref_intensity[0].sum()
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


def test_a_abins_ref() -> None:
    """Check calculated A against Abins isotropic calculation

    The reference average_a_traces are from Abins calculate_isotropic_dw method
    which takes weighted sum over traces at each k-point.

    """

    modes = QpointPhononModes.from_json_file(
        str(test_data / "GaSb_qpoint_phonon_modes.json")
    )

    dw = calculate_atomic_displacements(modes, temperature=Quantity(100, "K"))
    abins_average_a_traces = np.array([0.01321831, 0.01127088])

    assert_allclose(np.trace(dw.debye_waller.to("angstrom^2").magnitude, axis1=1, axis2=2),
                    abins_average_a_traces / 2,
                    atol=1e-8)


def test_displacements_abins_ref() -> None:
    """Check calculated displacements against Mantid-Abins reference

    Note that as in ADP there seems to be a factor two difference as Mantid
    implementation has absorbed the "2" to construct 2W when summing over B

    """

    modes = QpointPhononModes.from_json_file(
        str(test_data / "GaSb_qpoint_phonon_modes.json")
    )
    b = calculate_mode_displacements(modes, temperature=Quantity(0, 'kelvin'))

    # From Abins, GaSb
    # b-tensors on qpt 2 before bose weighting:
    ref_b_gasb_qpt_2 = np.array(
        [[[[ 1.48587656e-03,  7.50849933e-04,  7.48360116e-04],
           [ 7.50849933e-04,  3.79424206e-04,  3.78164567e-04],
           [ 7.48360116e-04,  3.78164567e-04,  3.76910788e-04]],

          [[ 1.38937833e-03,  7.10432646e-04,  7.08110593e-04],
           [ 7.10432646e-04,  3.63272737e-04,  3.62088584e-04],
           [ 7.08110593e-04,  3.62088584e-04,  3.60909927e-04]]],

         [[[ 1.89313182e-09, -1.22087873e-06,  1.22283305e-06],
           [-1.22087873e-06,  1.11538946e-03, -1.11666500e-03],
           [ 1.22283305e-06, -1.11666500e-03,  1.11794255e-03]],

          [[ 1.80713405e-09, -1.15304297e-06,  1.15487878e-06],
           [-1.15304297e-06,  1.04835577e-03, -1.04953272e-03],
           [ 1.15487878e-06, -1.04953272e-03,  1.05071154e-03]]],


         [[[ 1.71992043e-04, -1.68918173e-04, -1.68917047e-04],
           [-1.68918173e-04,  1.65899251e-04,  1.65898144e-04],
           [-1.68917047e-04,  1.65898144e-04,  1.65897038e-04]],

          [[ 2.73952065e-04, -2.68765808e-04, -2.68763937e-04],
           [-2.68765808e-04,  2.63699831e-04,  2.63697918e-04],
           [-2.68763937e-04,  2.63697918e-04,  2.63696004e-04]]],

         [[[ 2.59028780e-04, -2.70461782e-04, -2.70538263e-04],
           [-2.70461782e-04,  2.82612314e-04,  2.82692237e-04],
           [-2.70538263e-04,  2.82692237e-04,  2.82772183e-04]],

          [[ 5.43028601e-05, -5.59343902e-05, -5.59404480e-05],
           [-5.59343902e-05,  5.78273828e-05,  5.78349764e-05],
           [-5.59404480e-05,  5.78349764e-05,  5.78425793e-05]]],

         [[[ 5.12162113e-13, -2.09565362e-09,  2.09541534e-09],
           [-2.09565362e-09,  3.40594774e-04, -3.40476541e-04],
           [ 2.09541534e-09, -3.40476541e-04,  3.40358351e-04]],

          [[ 1.11038411e-12,  1.03777160e-08, -1.03769150e-08],
           [ 1.03777160e-08,  1.18803506e-04, -1.18783735e-04],
           [-1.03769150e-08, -1.18783735e-04,  1.18763971e-04]]],

        [[[ 4.67513677e-04,  2.22335826e-04,  2.22411360e-04],
          [ 2.22335826e-04,  1.05844601e-04,  1.05880162e-04],
          [ 2.22411360e-04,  1.05880162e-04,  1.05915737e-04]],

         [[ 1.59504121e-04,  7.81237807e-05,  7.81494256e-05],
          [ 7.81237807e-05,  3.83667651e-05,  3.83795488e-05],
          [ 7.81494256e-05,  3.83795488e-05,  3.83923371e-05]]]])

    assert_allclose(
        b.to("angstrom^2").magnitude[1],
        ref_b_gasb_qpt_2,
    )

# Raw intensity values from Abins _calculate_order_one (pre-DW)
# [[[0., 0., 0., 0.00589447, 0.00589447, 0.00586518],
#   [0., 0., 0., 0.00193254, 0.00193254, 0.00192238]]
#  [[0.0174369,  0.01732897, 0.00349493, 0.00698129, 0.00583648, 0.00583332],
#   [0.01643643, 0.01628717, 0.0055592,  0.00143936, 0.0020362,  0.00202893]]]
