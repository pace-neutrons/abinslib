"""Unit tests for abinslib.displacements module"""

from itertools import product

from euphonic import Quantity
import numpy as np
from numpy.testing import assert_allclose
import pytest

from abinslib.displacements import Displacements


def test_displacements(rng):
    """Self-consistency check of displacements properties"""

    displacements = Displacements(
        displacements=rng.random((2, 4, 5, 3, 3)),
        weights=np.array((0.2, 0.8)),
        bose_n=rng.random((2, 4)),
        temperature=Quantity(10, "K"),
    )

    assert_allclose(
        (displacements.n / displacements.one)[:, :, 0, 0, 0], displacements.bose_n
    )

    assert_allclose(
        displacements.two_n_plus_one - displacements.one, displacements.n * 2.0
    )


@pytest.mark.parametrize(
    ("modes", "abins_average_a_traces"),
    [("GaSb", (0.01321831, 0.01127088))],
    indirect=("modes",),
)
def test_calculate_adp(modes, abins_average_a_traces):
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

    atomic_displacements = Displacements.from_modes(
        modes, temperature=Quantity(100, "K")
    ).to_atomic_displacements()

    euphonic_dw = modes.calculate_debye_waller(
        temperature=Quantity(100, "K"),
        frequency_min=Quantity(0.01, "meV"),
        symmetrise=False,
    ).debye_waller

    assert atomic_displacements.units == euphonic_dw.units
    assert_allclose(atomic_displacements.magnitude, euphonic_dw.magnitude)

    assert_allclose(
        np.trace(atomic_displacements.to("angstrom^2").magnitude, axis1=1, axis2=2),
        np.array(abins_average_a_traces) / 2,
        atol=1e-8,
    )


@pytest.mark.parametrize(
    ("modes", "temperature_k"), product(["GaSb"], [0, 100]), indirect=("modes",)
)
def test_dw_regression(modes, temperature_k, ndarrays_regression):
    dw = Displacements.from_modes(
        modes, temperature=Quantity(temperature_k, "K")
    ).to_atomic_displacements()
    ndarrays_regression.check({"dw": dw.to("angstrom^2").magnitude})


@pytest.mark.parametrize(
    ("modes", "ref_npz"), [("GaSb", "GaSb_abins_isotropic_dw.npz")], indirect=True
)
def test_a_abins_ref(modes, ref_npz) -> None:
    """Check calculated A against Abins isotropic calculation

    The reference average_a_traces are from Abins calculate_isotropic_dw method
    which takes weighted sum over traces at each k-point.

    """
    ref_a_traces = ref_npz["a_traces"]

    dw = Displacements.from_modes(
        modes, temperature=Quantity(100, "K")
    ).to_atomic_displacements()
    assert_allclose(
        np.trace(dw.to("angstrom^2").magnitude, axis1=1, axis2=2),
        ref_a_traces / 2,
        atol=1e-8,
    )


@pytest.mark.parametrize(
    ("modes", "temperature_k", "ref_npz"),
    [
        ("GaSb", 0, "GaSb_abins_0k_B.npz"),
        ("GaSb", 100, "GaSb_abins_100k_B.npz"),
    ],
    indirect=("modes", "ref_npz"),
)
def test_displacements_abins_ref(modes, temperature_k, ref_npz) -> None:
    """Check calculated displacements against Mantid-Abins reference

    Note that as in ADP there seems to be a factor two difference as Mantid
    implementation has absorbed the "2" to construct 2W when summing over B

    """
    b = Displacements.from_modes(
        modes,
        temperature=Quantity(temperature_k, "kelvin"),
    )

    assert_allclose(
        b.n_plus_one.to("angstrom^2").magnitude[1],
        np.swapaxes(ref_npz["qpt-1"], 0, 1),
    )
