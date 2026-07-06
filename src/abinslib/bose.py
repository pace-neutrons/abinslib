"""Bose occupation enum and calculations."""

from enum import Enum, auto

from euphonic import Quantity, ureg
import numpy as np


class NegativeTemperatureError(ValueError):
    """Bose factor is not meaningful for negative temperature."""


class BoseOccupation(Enum):
    """Occupation number for Bose-Einstein statistics.

    Typically we use 2N+1 in Debye-Waller factor (i.e. atomic displacements),
    N+1 for energy transfer to the sample and N for energy transfer from the
    sample.

    ONE is simply 1: i.e. unscaled temperature-insensitive occupation
    """

    ONE = auto()
    N = auto()
    N_PLUS_ONE = auto()
    TWO_N_PLUS_ONE = auto()


def calculate_bose_factor(
    frequencies: Quantity,
    temperature: Quantity,
    occupation: BoseOccupation,
) -> np.array:
    """Get Bose factors corresponding to an array of frequency or energy.

    Args:
        frequencies: phonon frequencies, usually indexed (qpt, mode)
        temperature: determines magnitude of Bose factors
        occupation: typically N_PLUS_ONE is used for phonon excitations

    Returns:
        Bose factor array corresponding to input frequencies (i.e. usually
        indexed (qpt, mode))
        
    """
    if temperature == Quantity(0.0, "kelvin"):
        return _zero_t_bose_factor(frequencies, occupation)
    if temperature < Quantity(0.0, "kelvin"):
        raise NegativeTemperatureError("Temperature must not be negative")

    frequencies = frequencies.to("hartree").magnitude
    # Cast T to Kelvin first in case of non-multiplicative unit (celsius)
    kT = (ureg.k * temperature.to("kelvin")).to("hartree").magnitude

    two_n_plus_one = 1 / (np.tanh(frequencies / (2 * kT)))

    match occupation:
        case BoseOccupation.TWO_N_PLUS_ONE:
            return two_n_plus_one
        case BoseOccupation.N_PLUS_ONE:
            return two_n_plus_one * 0.5 + 0.5
        case BoseOccupation.N:
            return two_n_plus_one * 0.5 - 0.5
        case BoseOccupation.ONE:
            return np.ones_like(two_n_plus_one)
        case other:
            raise TypeError(f"Not a valid occupation number: {other}")


def _zero_t_bose_factor(
    frequencies: Quantity, occupation: BoseOccupation
) -> np.ndarray:
    """Get ideal occupation values if T=0, avoiding divide-by-zero."""
    match occupation:
        case BoseOccupation.N_PLUS_ONE | BoseOccupation.TWO_N_PLUS_ONE:
            return np.ones_like(frequencies.magnitude)
        case BoseOccupation.ONE:
            return np.ones_like(frequencies.magnitude)
        case BoseOccupation.N:
            return np.zeros_like(frequencies.magnitude)
        case other:
            raise TypeError(f"Not a valid occupation number: {other}")
