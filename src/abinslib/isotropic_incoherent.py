from __future__ import annotations

from typing import TYPE_CHECKING

from euphonic import Quantity, ureg
from euphonic.crystal import Crystal
from euphonic.spectra import Spectrum1DCollection
import numpy as np

if TYPE_CHECKING:
    from euphonic import QpointPhononModes

    from . import Displacements


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


def calculate_isotropic_incoherent_fundamentals(
    modes: QpointPhononModes,
    mode_displacements: Quantity,
    atomic_displacements: Quantity,
    nominal_q2: Quantity,
    include_dw: bool = True,
) -> np.ndarray:
    """Calculate mode intensities in fully-isotropic approximation

    S = exp(-(Q^2 tr(A)/3)) Q^2 tr(B) / 3

    - Fundamentals only
    - Atomic cross sections not applied
    - Ignore actual q-points and use nominal Q^2 instead

    Return array indices (qpt, mode, atom)

    """
    intensities = (
        np.einsum(
            "ij,ijkll->ijk",
            nominal_q2.to("bohr^-2").magnitude,
            mode_displacements.to("bohr^2").magnitude,
        )
        / 3
    )

    if include_dw:
        dw_factor = calculate_isotropic_dw_factor(atomic_displacements, nominal_q2)
    else:
        dw_factor = 1.0

    return intensities * dw_factor


def calculate_isotropic_dw_factor(
    atomic_displacements: Quantity,
    q2: Quantity,
) -> np.ndarray:
    dw = atomic_displacements.to("bohr^2").magnitude

    return np.exp(
        -2  # Mantid incorporates this factor 2 into A
        * np.expand_dims(q2.to("bohr^-2").magnitude, -1)
        * np.trace(dw, axis1=-2, axis2=-1)[None, None, :]
        / 3
    )


def _bin_mode_intensities(
    modes: QpointPhononModes,
    intensities: np.ndarray,
    bins: Quantity,
    apply_cross_section: bool = True,
) -> Quantity:
    """Bin intensities corresponding to QpointPhononModes to 1D spectra

    Sum over q-point and mode indices, using q-point weights from modes

    Output array has shape (atom_indices, bin_indices)
    """
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
    return y_data


def calculate_isotropic_incoherent_spectra(
    modes: QpointPhononModes,
    mode_displacements: Displacements,
    atomic_displacements: Quantity,
    nominal_q2: Quantity,
    bins: Quantity,
    apply_cross_section: bool = True,
    include_dw: bool = True,
) -> Spectrum1DCollection:
    """Calculate INS intensities in fully-isotropic incoherent approximation

    Actual q-points of phonon modes will be disregarded; instead each mode
    intensity will be based on a separate array of nominal Q^2 values
    corresponding to modes. This is intended to approximate powder-averaging
    with kinematic constraints: for indirect geometry the energy-Q^2
    relationship can be determined using abinslib.utils.calculate_indirect_q2.

    """

    intensities = calculate_isotropic_incoherent_fundamentals(
        modes=modes,
        mode_displacements=mode_displacements.n_plus_one,
        atomic_displacements=atomic_displacements,
        nominal_q2=nominal_q2,
        include_dw=include_dw,
    )
    y_data = _bin_mode_intensities(
        modes=modes,
        intensities=intensities,
        bins=bins,
        apply_cross_section=apply_cross_section,
    )

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
    mode_displacements: Displacements,
    atomic_displacements: Quantity,
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

    # More generally this factor is Q^2N / N!
    q2_scale = nominal_q2 / Quantity(1, "Å^-2")

    dw = calculate_isotropic_dw_factor(
        atomic_displacements=atomic_displacements,
        q2=nominal_q2,
    )

    spectra.y_data = spectra.y_data * q2_scale * np.swapaxes(dw, -1, -2)[0]

    return spectra
