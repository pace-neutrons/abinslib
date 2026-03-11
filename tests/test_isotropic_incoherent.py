from pathlib import Path

from euphonic import QpointPhononModes, Quantity
import numpy as np
from numpy.testing import assert_allclose

from abinslib.isotropic_incoherent import (
    calculate_atomic_displacements,
    calculate_mode_displacements,
    calculate_isotropic_incoherent_fundamentals,
)

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
