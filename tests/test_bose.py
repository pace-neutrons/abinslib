from euphonic import Quantity
from numpy.testing import assert_allclose
import pytest

from abinslib.bose import (
    NegativeTemperatureError,
    BoseOccupation,
    calculate_bose_factor,
)


@pytest.fixture
def frequencies() -> Quantity:
    return Quantity([1.0, 2.0, 3.0], "1/cm")


@pytest.fixture(scope="module", params=["kelvin", "celsius"])
def temperature(request) -> Quantity:
    return Quantity(100, request.param)


def test_negative_temperature(frequencies):
    with pytest.raises(NegativeTemperatureError):
        calculate_bose_factor(
            frequencies, Quantity(-500, "celsius"), BoseOccupation.N_PLUS_ONE
        )


def test_relative_factors(frequencies, temperature):
    one = calculate_bose_factor(frequencies, temperature, BoseOccupation.ONE)
    n = calculate_bose_factor(frequencies, temperature, BoseOccupation.N)
    n_plus_one = calculate_bose_factor(
        frequencies, temperature, BoseOccupation.N_PLUS_ONE
    )
    two_n_plus_one = calculate_bose_factor(
        frequencies, temperature, BoseOccupation.TWO_N_PLUS_ONE
    )

    assert_allclose(one, 1.0)
    assert_allclose(n_plus_one, n + 1.0)
    assert_allclose(two_n_plus_one, 2 * n + 1.0)


@pytest.mark.parametrize(
    ("occupation", "expected"),
    [
        (BoseOccupation.ONE, 1.0),
        (BoseOccupation.N, 0.0),
        (BoseOccupation.N_PLUS_ONE, 1.0),
        (BoseOccupation.TWO_N_PLUS_ONE, 1.0),
    ],
)
def test_zero_t_factors(frequencies, occupation, expected):
    assert_allclose(
        calculate_bose_factor(frequencies, Quantity(0, "kelvin"), occupation),
        expected,
    )


@pytest.mark.parametrize("occupation", [1, "N", None, True])
def test_bad_occupation_type(frequencies, temperature, occupation):
    with pytest.raises(TypeError, match="Not a valid occupation number"):
        calculate_bose_factor(frequencies, temperature, occupation)


def test_factors_regression(frequencies, temperature, num_regression):
    """Check values from calculate_bose_factor have not changed"""
    factor = calculate_bose_factor(
        frequencies, temperature, occupation=BoseOccupation.N
    )
    num_regression.check({"bose_factor": factor})
