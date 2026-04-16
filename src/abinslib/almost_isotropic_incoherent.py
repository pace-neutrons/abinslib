"""Semi-analytic powder averaging approximations in CLIMAX/AbINS lineage"""

from euphonic import DebyeWaller, Quantity
import numpy as np

from .displacements import Displacements

def calculate_isotropic_incoherent_fundamentals(
    mode_displacements: Displacements,
    atomic_displacements: DebyeWaller,
    nominal_q2: Quantity,
) -> Quantity:
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

    a_trace = np.einsum("...ii", atomic_displacements.debye_waller)
    b_trace = np.einsum("...ii", mode_displacements.n_plus_one)
    ba_trace = np.einsum(
        "ijklm, kml->ijk",
        mode_displacements.n_plus_one,
        atomic_displacements.debye_waller)
    inv_b_trace = np.divide(
        1.0,
        b_trace,
        out=np.zeros_like(b_trace),
        where=(b_trace != 0.0),
    )

    exp_term = np.exp(
        -nominal_q2[:, :, None] * (a_trace[None, None, :] + 2.0 * ba_trace * inv_b_trace) / 5.0
    )
    return q2_term * exp_term
