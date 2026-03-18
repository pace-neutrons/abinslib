from __future__ import annotations
from typing import TYPE_CHECKING

from euphonic import ureg, DebyeWaller, Quantity
from euphonic.spectra import Spectrum1DCollection
import numpy as np

from .bose import BoseOccupation, calculate_bose_factor

if TYPE_CHECKING:
    from euphonic import QpointPhononModes


def calculate_mode_displacements(
    modes: QpointPhononModes,
    temperature: Quantity,
    frequency_min: Quantity = Quantity(0.01, "meV"),
    occupation: BoseOccupation = BoseOccupation.TWO_N_PLUS_ONE,
) -> Quantity:
    """Get the 3x3 displacement tensor (B) for each atom and phonon mode

    Output array indices: (qpt, mode, atom, direction, direction)

    Implementation is heavily based on the Euphonic
    QpointPhononModes.calculate_debye_waller

    """
    frequencies = modes.frequencies.to("hartree").magnitude

    # Boolean mask of modes to include (i.e. frequency over threshold)
    mask = frequencies > frequency_min.to("hartree").magnitude

    if temperature > Quantity(0.0, "kelvin"):
        bose_factor = calculate_bose_factor(
            modes.frequencies,
            temperature,
            occupation=occupation,
        )
    else:
        # Actually this is incorrect for N (phonon emission) statistics
        bose_factor = 1.0

    freq_term = bose_factor / frequencies

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

    mode_displacements = mode_displacements / 2

    dw = np.einsum("ijklm,i->klm", mode_displacements.magnitude, modes.weights)
    # dw /= 2  # B values have same scaling as Mantid Abins, but this dw = A/2

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
    intensities = (
        nominal_q2.to("bohr^-2").magnitude[:, :, None]
        * np.trace(mode_displacements.to("bohr^2").magnitude, axis1=-2, axis2=-1)
        / 3
    )
    import warnings
    warnings.warn(str(intensities))

    dw = atomic_displacements.debye_waller.to("bohr^2").magnitude

    dw_factor = np.exp(
        -2  # Mantid incorporates this factor 2 into A
        * nominal_q2.to("bohr^-2").magnitude[:, :, None]
        * np.trace(dw, axis1=-2, axis2=-1)[None, None, :]
        / 3
    )

    return intensities * dw_factor


def calculate_isotropic_incoherent_spectra(
    modes: QpointPhononModes,
    mode_displacements: Quantity,
    atomic_displacements: DebyeWaller,
    nominal_q2: Quantity,
    bins: Quantity,
) -> Spectrum1DCollection:
    """Calculate INS intensities in fully-isotropic incoherent approximation

    Note that to give expected results, mode_displacements should have N+1 Bose
    occupation and atomic_displacements should have 2N+1 occupation.

    Actual q-points of phonon modes will be disregarded; instead each mode
    intensity will be based on a separate array of nominal Q^2 values
    corresponding to modes. This is intended to approximate powder-averaging
    with kinematic constraints: for indirect geometry the energy-Q^2
    relationship can be determined using abinslib.utils.calculate_indirect_q2.

    """

    intensities = calculate_isotropic_incoherent_fundamentals(
        modes=modes,
        mode_displacements=mode_displacements,
        atomic_displacements=atomic_displacements,
        nominal_q2=nominal_q2,
    )

    bin_width = bins[1] - bins[0]

    from euphonic.util import get_reference_data

    xs_coh_data = get_reference_data(
        collection="BlueBook", physical_property="coherent_cross_section"
    )
    xs_inc_data = get_reference_data(
        collection="BlueBook", physical_property="incoherent_cross_section"
    )
    cross_sections = [
        xs_coh_data[symbol].to("barn").magnitude
        + xs_inc_data[symbol].to("barn").magnitude
        for symbol in modes.crystal.atom_type
    ]

    q_weights = modes.weights / modes.weights.sum()

    weighted_intensities = np.einsum(
        "i,k,ijk->ijk", q_weights, np.array(cross_sections), intensities
    )

    frequencies = modes.frequencies.to(bins.units).magnitude
    y_data = np.zeros([modes.crystal.n_atoms, len(bins) - 1])

    # Swap atom and freq axes for clean iteration
    intensities_view = np.swapaxes(weighted_intensities, 1, 2)
    for q_frequencies, q_data in zip(frequencies, intensities_view):
        for atom_index, atom_data in enumerate(q_data):
            y_q_atom, _ = np.histogram(
                q_frequencies,
                bins=bins.magnitude,
                weights=atom_data,
                density=False,
            )
            y_data[atom_index] += y_q_atom

    # Apply correct spectral scaling / units
    y_data = y_data * ureg("barn") / bin_width

    metadata = {
        "method": "isotropic incoherent",
        "cross sections": "incoherent + coherent",
        "line_data": [
            {"atom_index": i, "atom_symbol": symbol}
            for i, symbol in enumerate(modes.crystal.atom_type)
        ],
    }
    return Spectrum1DCollection(x_data=bins, y_data=y_data, metadata=metadata)
