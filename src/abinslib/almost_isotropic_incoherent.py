"""Semi-analytic powder averaging approximations in CLIMAX/AbINS lineage"""

from __future__ import annotations
from typing import TYPE_CHECKING

from euphonic.spectra import Spectrum1DCollection
import numpy as np

from .displacements import Displacements
from .isotropic_incoherent import _bin_mode_intensities

if TYPE_CHECKING:
    from euphonic import DebyeWaller, QpointPhononModes, Quantity


def calculate_almost_isotropic_incoherent_fundamentals(
    mode_displacements: Displacements,
    atomic_displacements: DebyeWaller,
    nominal_q2: Quantity,
) -> np.ndarray:
    """Calculate fundamental mode intensities in almost-isotropic approximation

    S =  exp(-(Q^2 tr(A + 2 tr(A:B)/tr(B))/5)) Q^2 tr(B) / 3

    - Fundamentals only
    - Atomic cross sections not applied
    - Ignore actual q-points and use nominal Q^2 instead

    Return array indices (qpt, mode, atom)

    """
    q2_term = (
        np.einsum(
            "ij,ijkll->ijk",
            nominal_q2.to("bohr^-2").magnitude,
            mode_displacements.n_plus_one.to("bohr^2").magnitude,
        )
        / 3
    )

    a = atomic_displacements.debye_waller.to("bohr^2").magnitude * 2
    b = mode_displacements.n_plus_one.to("bohr^2").magnitude
    a_trace = np.einsum("...ii", a)
    b_trace = np.einsum("...ii", b)
    ba_trace = np.einsum("ijklm, kml->ijk", b, a)
    inv_b_trace = np.divide(
        1.0,
        b_trace,
        out=np.zeros_like(b_trace),
        where=(b_trace != 0.0),
    )

    exp_term = np.exp(
        -nominal_q2[:, :, None].to("bohr^-2").magnitude
        * (a_trace[None, None, :] + 2.0 * ba_trace * inv_b_trace)
        / 5.0
    )
    return q2_term * exp_term


def calculate_almost_isotropic_incoherent_spectra(
    modes: QpointPhononModes,
    mode_displacements: Displacements,
    atomic_displacements: DebyeWaller,
    nominal_q2: Quantity,
    bins: Quantity,
    apply_cross_section: bool = True,
) -> Spectrum1DCollection:
    """Calculate INS intensities in almost-isotropic incoherent approximation

    Actual q-points of phonon modes will be disregarded; instead each mode
    intensity will be based on a separate array of nominal Q^2 values
    corresponding to modes. This is intended to approximate powder-averaging
    with kinematic constraints: for indirect geometry the energy-Q^2
    relationship can be determined using abinslib.utils.calculate_indirect_q2.

    """
    intensities = calculate_almost_isotropic_incoherent_fundamentals(
        mode_displacements=mode_displacements,
        atomic_displacements=atomic_displacements,
        nominal_q2=nominal_q2,
    )
    y_data = _bin_mode_intensities(
        modes=modes,
        intensities=intensities,
        bins=bins,
        apply_cross_section=apply_cross_section,
    )

    metadata = {
        "method": "almost-isotropic incoherent approximation",
        "cross sections": ("incoherent + coherent" if apply_cross_section else "none"),
        "line_data": [
            {"atom_index": i, "atom_symbol": symbol, "quantum_order": 1}
            for i, symbol in enumerate(modes.crystal.atom_type)
        ],
    }
    return Spectrum1DCollection(x_data=bins, y_data=y_data, metadata=metadata)
