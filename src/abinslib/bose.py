"""Bose occupation calculations"""

from enum import Enum, auto

from euphonic import Quantity, ureg
import numpy as np


class BoseOccupation(Enum):
    """Occupation number for Bose-Einstein statistics

    Typically we use 2N+1 in Debye-Waller factor (i.e. atomic displacements),
    N+1 for energy transfer to the sample and N for energy transfer from the
    sample.
    """

    N = auto()
    N_PLUS_ONE = auto()
    TWO_N_PLUS_ONE = auto()


def calculate_bose_factor(
    frequencies: Quantity,
    temperature: Quantity,
    occupation: BoseOccupation,
) -> np.array:
    """Get Bose factors corresponding to an array of frequency or energy"""

    frequencies = frequencies.to("hartree").magnitude
    kT = (ureg.k * temperature).to("hartree").magnitude

    two_n_plus_one = 1 / (np.tanh(frequencies / (2 * kT)))

    match occupation:
        case BoseOccupation.TWO_N_PLUS_ONE:
            return two_n_plus_one
        case BoseOccupation.N_PLUS_ONE:
            return two_n_plus_one * 0.5 + 0.5
        case BoseOccupation.N:
            return two_n_plus_one * 0.5 - 0.5
        case other:
            raise ValueError(f"Not a valid occupation number: {other}")
