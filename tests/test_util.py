from euphonic import Quantity
from numpy import pi
from numpy.testing import assert_allclose

from abinslib.util import calculate_indirect_q2


def test_calculate_indirect_q2() -> None:
    """Check indirect-geometry kinematic-constraint calculation

    Reference values and TOSCA parameters are from the Mantid 6.15 QECoverage
    interface

    (Note that the nominal 38.92 scattering angle is quite a large deviation
     from the 45 degrees currently used for intensity calculations.)
    """

    energy_transfer = Quantity([11.496160804020064, 984.8698391959799], "meV")
    expected_q = Quantity([1.8674137643168969, 20.827728202858143], "1/angstrom")

    q2 = calculate_indirect_q2(
        energy_transfer=energy_transfer,
        angle=(38.92 * pi / 180),
        final_energy=Quantity(3.634, "meV"),
    )

    assert_allclose(
        q2.to("angstrom^-2").magnitude, (expected_q**2).to("angstrom^-2").magnitude
    )
