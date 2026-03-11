from __future__ import annotations
from typing import TYPE_CHECKING

from euphonic import ureg, DebyeWaller, Quantity
import numpy as np

if TYPE_CHECKING:
    from euphonic import QpointPhononModes


def calculate_mode_displacements(
    modes: QpointPhononModes,
    temperature: Quantity,
    frequency_min: Quantity = Quantity(0.01, "meV"),
) -> Quantity:
    """Get the 3x3 displacement tensor (B) for each atom and phonon mode

    Output array indices: (qpt, mode, atom, direction, direction)

    Implementation is heavily based on the Euphonic
    QpointPhononModes.calculate_debye_waller

    """
    k_B = (1 * ureg.k).to("hartree/K").magnitude
    frequencies = modes.frequencies.to("hartree").magnitude
    temperature_k = temperature.to("K").magnitude

    # Boolean mask of modes to include (i.e. frequency over threshold)
    mask = frequencies > frequency_min.to("hartree").magnitude

    if temperature > Quantity(0.0, "kelvin"):
        x = frequencies / (2 * k_B * temperature_k)
        freq_term = 1 / (frequencies * np.tanh(x))
    else:
        freq_term = 1 / frequencies

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
            1 / (4 * modes.crystal.atom_mass.to("m_e").magnitude),
            freq_term[q_index],
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
            modes=modes, temperature=temperature
        )

    dw = np.einsum("ijklm,i->klm", mode_displacements.magnitude, modes.weights)

    return DebyeWaller(
        modes.crystal, Quantity(dw, mode_displacements.units), temperature
    )


def calculate_isotropic_incoherent_fundamentals(
        modes: QpointPhononModes,
        mode_displacements: Quantity,
        atomic_displacements: DebyeWaller,
        nominal_q2: Quantity,
) -> Quantity:
    """Calculate mode intensities in fully-isotropic approximation

    S = exp(-(Q^2 tr(A)/3)) Q^2 tr(B) / 3

    - Fundamentals only
    - Atomic cross sections not applied
    - Ignore actual q-points and use nominal Q^2 instead

    Return array indices (qpt, mode, atom)

    """

    intensities = (nominal_q2.to("bohr^-2").magnitude[:, :, None]
                   * np.trace(mode_displacements.to("bohr^2").magnitude,
                              axis1=-2, axis2=-1)
                   / 3)

    dw = atomic_displacements.debye_waller.to("bohr^2").magnitude

    dw_factor = np.exp(-nominal_q2.to("bohr^-2").magnitude[:, :, None]
                       * np.trace(dw, axis1=-2, axis2=-1)[None, None, :]
                       / 3)

    return intensities * dw_factor
