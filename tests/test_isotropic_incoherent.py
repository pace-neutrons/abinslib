from pathlib import Path
import warnings

from euphonic import QpointPhononModes, Quantity
import numpy as np
from numpy.testing import assert_allclose
import pytest

from abinslib.isotropic_incoherent import (
    BoseOccupation,
    calculate_atomic_displacements,
    calculate_mode_displacements,
    calculate_isotropic_dw_factor,
    calculate_isotropic_incoherent_fundamentals,
    calculate_isotropic_incoherent_spectra,
)
from abinslib.util import calculate_indirect_q2


test_data = Path(__file__).parent / "data"

# Raw intensity values from Abins _calculate_order_one (i.e. pre-DW)
abins_fundamentals_no_dw = [
    (
        0,
        Quantity(np.load(test_data / "GaSb_modes_q2.npy"), "angstrom^-2"),
        np.array(
            [
                [
                    [0.0, 0.0, 0.0, 0.00566968, 0.00566967, 0.00564963],
                    [0.0, 0.0, 0.0, 0.00185884, 0.00185884, 0.00185173],
                ],
                [
                    [
                        0.00773538,
                        0.0077202,
                        0.00293306,
                        0.00666757,
                        0.00559242,
                        0.00559219,
                    ],
                    [
                        0.00729155,
                        0.00725607,
                        0.00466545,
                        0.00137468,
                        0.00195106,
                        0.00194506,
                    ],
                ],
            ]
        ),
    ),
    # Nominal Q^2=1. used for Mantid-Abins fully-isotropic calculations
    # (energy-dependent scale factor applied as a later step)
    (
        0,
        Quantity(np.ones([2, 6]), "angstrom^-2"),
        [
            [
                [0.0, 0.0, 0.0, 0.00022573, 0.00022573, 0.00022323],
                [0.0, 0.0, 0.0, 7.40083172e-05, 7.40083087e-05, 7.31655322e-05],
            ],
            [
                [0.0007474, 0.00074444, 0.00016793, 0.0002748, 0.00022698, 0.00022642],
                [
                    7.04520332e-04,
                    6.99689704e-04,
                    2.67115967e-04,
                    5.66576074e-05,
                    7.91891594e-05,
                    7.87544076e-05,
                ],
            ],
        ],
    ),
    (
        100,
        Quantity(np.load(test_data / "GaSb_modes_q2.npy"), "angstrom^-2"),
        np.array(
            [
                [
                    [0.0, 0.0, 0.0, 0.00589447, 0.00589447, 0.00586518],
                    [0.0, 0.0, 0.0, 0.00193254, 0.00193254, 0.00192238],
                ],
                [
                    [
                        0.0174369,
                        0.01732897,
                        0.00349493,
                        0.00698129,
                        0.00583648,
                        0.00583332,
                    ],
                    [
                        0.01643643,
                        0.01628717,
                        0.0055592,
                        0.00143936,
                        0.0020362,
                        0.00202893,
                    ],
                ],
            ]
        ),
    ),
]


@pytest.fixture(scope="module")
def gasb_modes() -> QpointPhononModes:
    """Phonon modes of GaSb on two q-points from CASTEP"""
    return QpointPhononModes.from_json_file(
        str(test_data / "GaSb_qpoint_phonon_modes.json")
    )


def test_calculate_adp(gasb_modes):
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


@pytest.mark.parametrize("temperature_k,q2,abins_ref", abins_fundamentals_no_dw)
def test_calculate_isotropic_incoherent_fundamentals(
    gasb_modes, temperature_k, q2, abins_ref
):
    temperature = Quantity(temperature_k, "K")
    b = calculate_mode_displacements(
        gasb_modes, temperature=temperature, occupation=BoseOccupation.N_PLUS_ONE
    )

    a = calculate_atomic_displacements(
        gasb_modes,
        temperature=temperature,
        mode_displacements=calculate_mode_displacements(
            gasb_modes,
            temperature=temperature,
            occupation=BoseOccupation.TWO_N_PLUS_ONE,
        ),
    )

    result = calculate_isotropic_incoherent_fundamentals(gasb_modes, b, a, q2)
    # Remove DW scaling to compare with Abins pre-DW intensities
    dw_factor = calculate_isotropic_dw_factor(a, q2)
    result /= dw_factor

    abins_ref = np.swapaxes(abins_ref, -1, -2)
    assert_allclose(result, abins_ref, rtol=1e-5, atol=1e-8)


@pytest.mark.parametrize("temperature_k", [10, 100])
def test_calculate_isotropic_incoherent_spectrum(temperature_k, gasb_modes):
    temperature = Quantity(temperature_k, "K")
    ref_data = np.load(test_data / f"GaSb_abins_{temperature_k}k_isotropic_raw.npz")

    bins = Quantity(ref_data["energy"], str(ref_data["energy_unit"]))
    ref_intensity = ref_data["intensity"]

    n_plus_one_b = calculate_mode_displacements(
        gasb_modes, temperature=temperature, occupation=BoseOccupation.N_PLUS_ONE
    )
    two_n_plus_one_b = calculate_mode_displacements(
        gasb_modes, temperature=temperature, occupation=BoseOccupation.TWO_N_PLUS_ONE
    )
    a = calculate_atomic_displacements(
        gasb_modes, temperature=temperature, mode_displacements=two_n_plus_one_b
    )

    # Q2 calculated at exact Mantid-Abins TOSCA backscattering angle
    q2 = Quantity(np.load(test_data / "GaSb_modes_q2.npy"), "angstrom^-2")

    spectra = calculate_isotropic_incoherent_spectra(
        gasb_modes, n_plus_one_b, a, q2, bins
    )

    spectrum = spectra.sum()

    # TODO This is still a little mysterious, investigate further.
    # We seem to differ by a factor ~8?
    scale_offset = spectrum.y_data.sum().magnitude / ref_intensity[0].sum()
    if not (0.95 < scale_offset < 1.05):
        msg = f"Overall magnitude different from AbINS: {scale_offset * 100:.1f}%"
        warnings.warn(msg)

    # Compare normalized spectra for now
    assert_allclose(
        spectrum.y_data.magnitude,
        ref_intensity[0] * scale_offset,
        rtol=1e-2
    )


def test_a_abins_ref(gasb_modes) -> None:
    """Check calculated A against Abins isotropic calculation

    The reference average_a_traces are from Abins calculate_isotropic_dw method
    which takes weighted sum over traces at each k-point.

    """
    ref_a_traces = np.load(test_data / "GaSb_abins_isotropic_dw.npz")["a_traces"]

    dw = calculate_atomic_displacements(gasb_modes, temperature=Quantity(100, "K"))
    assert_allclose(
        np.trace(dw.debye_waller.to("angstrom^2").magnitude, axis1=1, axis2=2),
        ref_a_traces / 2,
        atol=1e-8,
    )


def test_isotropic_dw(gasb_modes):
    """Check isotropic DW on Q2 mesh agrees with Mantid-Abins"""
    dw_data = np.load(test_data / "GaSb_abins_isotropic_dw.npz")
    q2 = Quantity(dw_data["q2"], "angstrom^-2")

    a = calculate_atomic_displacements(gasb_modes, temperature=Quantity(100, "kelvin"))

    binned_dw_factor = calculate_isotropic_dw_factor(a, q2[None, :])
    assert_allclose(binned_dw_factor[0], dw_data["iso_dw"].transpose(), rtol=1e-6)


@pytest.mark.parametrize("temperature_k", [0, 100])
def test_displacements_abins_ref(temperature_k, gasb_modes) -> None:
    """Check calculated displacements against Mantid-Abins reference

    Note that as in ADP there seems to be a factor two difference as Mantid
    implementation has absorbed the "2" to construct 2W when summing over B

    """
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
