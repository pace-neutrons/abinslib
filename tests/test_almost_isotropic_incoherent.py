from pathlib import Path

from euphonic import QpointPhononModes, Quantity
import numpy as np
import pytest


from abinslib.displacements import Displacements
from abinslib.almost_isotropic_incoherent import (
    calculate_isotropic_incoherent_fundamentals,
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


def test_calculate_isotropic_incoherent_fundamentals(ref_modes):
    temperature = Quantity(100, "kelvin")
    modes = ref_modes["GaSb"]
    b = Displacements.from_modes(modes, temperature=temperature)
    a = b.to_atomic_displacements()

    q2 = calculate_indirect_q2(
        modes.frequencies,
        angle=(134.98885653282196 * np.pi / 180),
        final_energy=Quantity(32.0, "cm_1").to("hartree"),
    )

    _ = calculate_isotropic_incoherent_fundamentals(
        mode_displacements=b,
        atomic_displacements=a,
        nominal_q2=q2
    )
