from itertools import product
from pathlib import Path

from euphonic import Crystal, QpointPhononModes, Quantity
from euphonic.spectra import Spectrum1DCollection
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
    q_scaling_isotropic_incoherent_spectra,
)


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


@pytest.fixture
def patch_cross_sections(mocker):
    """Replace Euphonic cross-section lookup with Mantid values"""

    def _get_mantid_total_cross_sections(crystal: Crystal) -> Quantity:
        mantid_data = {
            "Ga": 6.83,
            "Sb": 3.9,
            "C": 5.551,
            "H": 82.02,
            "O": 4.232,
        }

        return Quantity(
            [mantid_data[symbol] for symbol in crystal.atom_type],
            "barn"
        )

    mocker.patch("abinslib.isotropic_incoherent._get_total_cross_sections",
                 wraps=_get_mantid_total_cross_sections)


@pytest.mark.parametrize("temperature_k,q2,abins_ref", abins_fundamentals_no_dw)
def test_calculate_isotropic_incoherent_fundamentals(
        ref_modes, temperature_k, q2, abins_ref,
):
    gasb_modes = ref_modes["GaSb"]
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

    result = calculate_isotropic_incoherent_fundamentals(
        gasb_modes, b, a, q2, include_dw=False
    )

    abins_ref = np.swapaxes(abins_ref, -1, -2)
    assert_allclose(result, abins_ref, rtol=1e-5, atol=1e-8)


@pytest.mark.parametrize(
    "temperature_k,system", product([10, 100], ["GaSb", "ethanol"])
)
def test_calculate_isotropic_incoherent_spectrum(temperature_k, ref_modes, system, patch_cross_sections):
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
def test_q_scaling_isotropic_incoherent_spectrum(temperature_k, ref_modes, system, patch_cross_sections):
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


def test_isotropic_dw(ref_modes):
    """Check isotropic DW on Q2 mesh agrees with Mantid-Abins"""
    gasb_modes = ref_modes["GaSb"]
    dw_data = np.load(test_data / "GaSb_abins_isotropic_dw.npz")
    q2 = Quantity(dw_data["q2"], "angstrom^-2")

    a = calculate_atomic_displacements(gasb_modes, temperature=Quantity(100, "kelvin"))

    binned_dw_factor = calculate_isotropic_dw_factor(a, q2[None, :])
    assert_allclose(binned_dw_factor[0], dw_data["iso_dw"].transpose(), rtol=1e-6)


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


def test_binning(ref_modes):
    """Check we can reproduce Mantid-Abins histogram binning

    Abins loops over q-points and atoms separately, which isn't how the
    abins-lib function works. So check that we can reproduce the results with
    a simple call to np.histogram from abins-lib mode intensities.
    """
    gasb_modes = ref_modes["GaSb"]
    ref_data = np.load(test_data / "GaSb_isotropic_binning.npz")

    temperature = Quantity(0, "K")
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
    q2 = Quantity(np.ones_like(gasb_modes.frequencies), "angstrom^-2")

    intensities = calculate_isotropic_incoherent_fundamentals(
        gasb_modes, b, a, q2, include_dw=False
    )

    bins = np.arange(0.0, 4100.001, 1.0)
    for atom_index, q_index in product((0, 1), (0, 1)):
        ref_spec = ref_data[f"atom_{atom_index}_k_{q_index}"]

        hist, _ = np.histogram(
            gasb_modes.frequencies[q_index].to("1/cm").magnitude,
            bins=bins,
            weights=intensities[q_index, :, atom_index],
            density=False,
        )
        hist *= gasb_modes.weights[q_index]

        assert_allclose(ref_spec, hist, rtol=1e-10)


def test_calculate_isotropic_incoherent_spectra_q1_no_dw(ref_modes):
    """Check spectrum collection against intermediate Abins data

    We are checking against the spectrum calculated at Q=1 before corrections
    are applied (i.e. from the middle of _calculate_s_isotropic), so

    - (input) q-point weights are included
    - cross sections are not included
    - nominal Q = 1

    Note that before DW is applied we are getting excellent agreement with 0K
    Abins calculations.

    """
    gasb_modes = ref_modes["GaSb"]
    temperature = Quantity(0.0, "K")

    bins = Quantity(np.arange(0.0, 4100.001, 1.0), "1/cm")

    n_plus_one_b = calculate_mode_displacements(
        gasb_modes, temperature=temperature, occupation=BoseOccupation.N_PLUS_ONE
    )
    two_n_plus_one_b = calculate_mode_displacements(
        gasb_modes, temperature=temperature, occupation=BoseOccupation.TWO_N_PLUS_ONE
    )
    a = calculate_atomic_displacements(
        gasb_modes, temperature=temperature, mode_displacements=two_n_plus_one_b
    )

    # Intensities at Q=1 before DW, Q2 scaling
    q2 = Quantity(np.ones_like(gasb_modes.frequencies), "angstrom^-2")
    spectra = calculate_isotropic_incoherent_spectra(
        gasb_modes,
        n_plus_one_b,
        a,
        q2,
        bins,
        apply_cross_section=False,
        include_dw=False,
    )

    ref_spectra = Spectrum1DCollection.from_json_file(
        test_data / "abins-spectra-unit-q.json"
    )

    for atom_index in {item["atom_index"] for item in spectra.iter_metadata()}:
        assert_allclose(
            ref_spectra.select(atom_index=atom_index).y_data.magnitude,
            spectra.select(atom_index=atom_index).y_data.magnitude,
        )

    # Repeat comparison after Q2 scaling applied to binned results
    ref_spectra = Spectrum1DCollection.from_json_file(
        test_data / "abins-spectra-no-dw.json"
    )
    spec_q2_scale = Quantity(np.load(test_data / "abins-q2-dump.npy"), "Å^-2 Å^2")

    for atom_index in {item["atom_index"] for item in spectra.iter_metadata()}:
        assert_allclose(
            ref_spectra.select(atom_index=atom_index).y_data.magnitude,
            (spectra.select(atom_index=atom_index).y_data * spec_q2_scale)
            .to("barn cm")
            .magnitude,
            rtol=1e-8,
        )

    # With Q2 scaling and DW
    ref_spectra = Spectrum1DCollection.from_json_file(
        test_data / "abins-spectra-dw.json"
    )

    dw_q2 = spec_q2_scale / Quantity(1, "Å^2")
    binned_dw_factor = np.swapaxes(calculate_isotropic_dw_factor(a, dw_q2), -2, -1)[0]

    for atom_index in {item["atom_index"] for item in spectra.iter_metadata()}:
        dw = binned_dw_factor[atom_index]

        assert_allclose(
            ref_spectra.select(atom_index=atom_index).y_data.magnitude,
            (spectra.select(atom_index=atom_index).y_data * spec_q2_scale * dw)
            .to("barn cm")
            .magnitude,
            rtol=1e-8,
        )
