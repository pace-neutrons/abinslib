from __future__ import annotations
from typing import TYPE_CHECKING

from euphonic import ureg, DebyeWaller, Quantity
from euphonic.crystal import Crystal
from euphonic.spectra import Spectrum1DCollection
import numpy as np

from .bose import BoseOccupation, calculate_bose_factor

if TYPE_CHECKING:
    from euphonic import QpointPhononModes


def _get_total_cross_sections(crystal: Crystal) -> Quantity:
    from euphonic.util import get_reference_data

    xs_coh_data = get_reference_data(
        collection="BlueBook", physical_property="coherent_cross_section"
    )
    xs_inc_data = get_reference_data(
        collection="BlueBook", physical_property="incoherent_cross_section"
    )
    return Quantity(
        [
            xs_coh_data[symbol].to("barn").magnitude
            + xs_inc_data[symbol].to("barn").magnitude
            for symbol in crystal.atom_type
        ],
        "barn",
    )


def calculate_mode_displacements(
    modes: QpointPhononModes,
    temperature: Quantity,
    frequency_min: Quantity = Quantity(10, "cm_1"),
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


def calculate_isotropic_incoherent_fundamentals(
    modes: QpointPhononModes,
    mode_displacements: Quantity,
    atomic_displacements: DebyeWaller,
    nominal_q2: Quantity,
    include_dw: bool = True,
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

    if include_dw:
        dw_factor = calculate_isotropic_dw_factor(atomic_displacements, nominal_q2)
    else:
        dw_factor = 1.0

    return intensities * dw_factor


def calculate_isotropic_dw_factor(
    atomic_displacements: DebyeWaller,
    q2: Quantity,
) -> np.ndarray:
    dw = atomic_displacements.debye_waller.to("bohr^2").magnitude

    return np.exp(
        -2  # Mantid incorporates this factor 2 into A
        * np.expand_dims(q2.to("bohr^-2").magnitude, -1)
        * np.trace(dw, axis1=-2, axis2=-1)[None, None, :]
        / 3
    )


def calculate_isotropic_incoherent_spectra(
    modes: QpointPhononModes,
    mode_displacements: Quantity,
    atomic_displacements: DebyeWaller,
    nominal_q2: Quantity,
    bins: Quantity,
    apply_cross_section: bool = True,
    include_dw: bool = True,
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
        include_dw=include_dw,
    )

    bin_width = bins[1] - bins[0]
    q_weights = modes.weights / modes.weights.sum()

    if apply_cross_section:
        atom_weights = _get_total_cross_sections(modes.crystal).to("barn").magnitude
    else:
        atom_weights = np.ones_like(modes.crystal.atom_mass)

    weighted_intensities = np.einsum(
        "i,k,ijk->ijk", q_weights, atom_weights, intensities
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
        "cross sections": ("incoherent + coherent" if apply_cross_section else "none"),
        "line_data": [
            {"atom_index": i, "atom_symbol": symbol, "quantum_order": 1}
            for i, symbol in enumerate(modes.crystal.atom_type)
        ],
    }
    return Spectrum1DCollection(x_data=bins, y_data=y_data, metadata=metadata)


def q_scaling_isotropic_incoherent_spectra(
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
    intensity will be calculated at Q=1/Å then rescaled to nominal Q^2 values
    corresponding to energy bins. This is intended to approximate
    powder-averaging with kinematic constraints: for indirect geometry the
    energy-Q^2 relationship can be determined using
    abinslib.utils.calculate_indirect_q2.

    """

    # Input should have a Q^2 per output bin center
    assert nominal_q2.shape == bins[:-1].shape

    spectra = calculate_isotropic_incoherent_spectra(
        modes=modes,
        mode_displacements=mode_displacements,
        atomic_displacements=atomic_displacements,
        nominal_q2=Quantity(np.ones_like(modes.frequencies.magnitude), "Å^-2"),
        bins=bins,
        apply_cross_section=True,
        include_dw=False,
    )

    if not isinstance(spectra, Spectrum1DCollection):
        raise TypeError("Only 1D spectra supported at this point")

    if any(
        map((lambda metadata: metadata["quantum_order"] != 1), spectra.iter_metadata())
    ):
        raise ValueError("Only order 1 supported at this point")

    # More generally this factor is Q^2N / N!
    q2_scale = nominal_q2 / Quantity(1, "Å^-2")

    dw = calculate_isotropic_dw_factor(
        atomic_displacements=atomic_displacements,
        q2=nominal_q2,
    )

    spectra.y_data = spectra.y_data * q2_scale * np.swapaxes(dw, -1, -2)[0]

    return spectra
