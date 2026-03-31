"""Functions for calculation of atomic displacement tensors

The phonon mode displacements are described by 3x3 matrices related to the
covariance matrices of atomic positions associated with each mode. In the
CLIMAX/Abins notation this is 'B' and may include a Bose occupation factor.

The sum over modes (include Brillouin Zone integration for periodic systems)
with appropriate Bose occupation gives a physical distribution known in various
forms as 'A', 'Atomic Displacement Parameters', 'U', ... and sometimes even
'B'.

In abins-lib these are referred to respectively as 'mode displacements' and
'atomic displacements' where space permits, or 'B' and 'A' in compact notation,
downcased to 'b' and 'a' in python variable names.

The preferred data type for A is the DebyeWaller class from Euphonic.

"""

from __future__ import annotations
from dataclasses import dataclass
from functools import cached_property
from typing import Self, TYPE_CHECKING

from euphonic import DebyeWaller, Quantity
import numpy as np

from .bose import BoseOccupation, calculate_bose_factor


if TYPE_CHECKING:
    from euphonic import QpointPhononModes


@dataclass(frozen=True)
class Displacements:
    """Atomic displacement dataset

    Current features:
    - access Bose-weighted displacements through cached properties

    Soon:
    - compute other weights on-the-fly

    Maybe:
    - Allow slicing by qpt, atom and/or mode index
      - return a View object that automatically slices from cached parent data?

    Parameters
    ----------
    mode_displacements:
      Atomic displacement tensor Quantity with dimensions (qpts, modes, atoms, 3, 3)
      and units length^2
    weights:
      Normalised q-point weights corresponding to axis 0 of mode_displacements
    temperature:
      Temperature at which Bose occupations were calculated
    occupation:
      Bose Occupation type (N, N+1 or 2N+1)

    """

    mode_displacements: Quantity
    weights: np.array
    temperature: Quantity
    occupation: BoseOccupation

    @classmethod
    def from_modes(
        cls: Self,
        modes: QpointPhononModes,
        temperature: Quantity,
        frequency_min: Quantity = Quantity(10, "cm_1"),
        occupation: BoseOccupation = BoseOccupation.N_PLUS_ONE,
    ) -> Self:
        return cls(
            mode_displacements=calculate_mode_displacements(
                modes, temperature, frequency_min, occupation
            ),
            weights=modes.weights,
            temperature=temperature,
            occupation=occupation,
        )

    def _get_occupied_displacements(self, occupation: BoseOccupation) -> Quantity:
        if self.occupation != occupation:
            raise ValueError()
        return self.mode_displacements

    @cached_property
    def n(self) -> Quantity:
        return self._get_occupied_displacements(BoseOccupation.N)

    @cached_property
    def n_plus_one(self) -> Quantity:
        return self._get_occupied_displacements(BoseOccupation.N_PLUS_ONE)

    @cached_property
    def two_n_plus_one(self) -> Quantity:
        return self._get_occupied_displacements(BoseOccupation.TWO_N_PLUS_ONE)


def calculate_mode_displacements(
    modes: QpointPhononModes,
    temperature: Quantity,
    frequency_min: Quantity = Quantity(10, "cm_1"),
    occupation: BoseOccupation = BoseOccupation.N_PLUS_ONE,
) -> Quantity:
    """Get the 3x3 displacement tensor (B) for each atom and phonon mode

    Output array indices: (qpt, mode, atom, direction, direction)

    Implementation is heavily based on the Euphonic
    QpointPhononModes.calculate_debye_waller

    """
    frequencies = modes.frequencies.to("hartree").magnitude

    # Boolean mask of modes to include (i.e. frequency over threshold)
    mask = frequencies > frequency_min.to("hartree").magnitude

    bose_factor = calculate_bose_factor(
        modes.frequencies,
        temperature,
        occupation=occupation,
    )

    mode_displacements = np.zeros(
        [*modes.frequencies.shape, modes.crystal.n_atoms, 3, 3]
    )

    # Euphonic DW does chunking here and works on multiple q-points at once.
    # For now we do something simpler and iterate over q-points

    for q_index, q_eigenvectors in enumerate(modes.eigenvectors):
        evec_term = np.real(
            np.einsum("ijk,ijl->ijkl", q_eigenvectors, np.conj(q_eigenvectors))
        )

        mode_displacements[q_index] = np.einsum(
            "j,i,i,ijkl->ijkl",
            1 / (2 * modes.crystal.atom_mass.to("m_e").magnitude),
            bose_factor[q_index] / frequencies[q_index],
            mask[q_index],
            evec_term,
        )

    mode_displacements = Quantity(mode_displacements, "bohr**2").to(
        modes.crystal.cell_vectors_unit + "**2"
    )
    return mode_displacements


def calculate_atomic_displacements(
    modes: QpointPhononModes,
    temperature: Quantity,
    mode_displacements: Quantity | None = None,
) -> DebyeWaller:
    """Calculate atomic displacement tensor (A) for each atom

    The return type is a Euphonic DebyeWaller object: "DebyeWaller" in Euphonic
    terminology is identical to A in CLIMAX terminology.
    In Euphonic coherent scattering intensity calculations, the Debye—Waller
    intensity factor appears as exp(-W_k) inside a square of sums.

    In incoherent intensity calculations the Debye—Waller factor typically
    appears as a pure factor exp(-2W_k).

    In both cases W_k is the displacement tensor of an atom summed over all
    phonon modes - i.e. "A".

    """
    if mode_displacements is None:
        mode_displacements = calculate_mode_displacements(
            modes=modes,
            temperature=temperature,
            occupation=BoseOccupation.TWO_N_PLUS_ONE,
        )

    dw = np.einsum(
        "ijklm,i->klm",
        mode_displacements.magnitude,
        modes.weights * 0.5,  # q-point symm weights, /2 scale convention for W
    )

    return DebyeWaller(
        modes.crystal, Quantity(dw, mode_displacements.units), temperature
    )
