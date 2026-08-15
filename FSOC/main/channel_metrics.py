"""
Channel metrics extraction: maps FSOC simulation outputs to quantum channel parameters.

Physics mapping:
    - Transmittance (eta):   PIB_after_AO / total_power_sent
    - Strehl ratio:          peak(corrected_psf) / peak(diffraction_limited_psf)
    - Phase variance:        var(residual_phase) over aperture  [rad^2]
    - Depolarizing prob:     1 - exp(-phase_variance / 2)  (Marechal approximation)
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class FSOCChannelMetrics:
    eta: float              # transmittance = PIB / total power sent
    strehl_ratio: float     # peak PSF ratio (corrected / diffraction-limited)
    phase_variance: float   # residual phase variance [rad^2]
    depolarizing_prob: float  # derived from phase variance
    background_counts: float  # photons per detection slot from background


def _safe_float(value: float, default: float = 0.0) -> float:
    """Return a finite float or a default."""
    try:
        value = float(value)
    except Exception:
        return default
    return value if np.isfinite(value) else default


def _finite_array(arr: np.ndarray) -> np.ndarray:
    """Return a finite array with NaN/inf replaced by zeros."""
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def extract_metrics(
    wf_focal_corrected,
    psf_diffraction_limited,
    wf_pupil_residual,
    aperture_mask,
    focal_grid,
    fiber_radius,
    total_power_sent,
    zero_magnitude_flux=3.9e10,
    stellar_magnitude=5,
    fiber_solid_angle=1e-10,
    dark_count_rate=1e-6,
):
    """Extract quantum channel metrics from FSOC simulation wavefronts.

    Parameters
    ----------
    wf_focal_corrected : hcipy.Wavefront
        AO-corrected wavefront at the focal plane.
    psf_diffraction_limited : array
        Diffraction-limited PSF power array (from unaberrated wavefront).
    wf_pupil_residual : hcipy.Wavefront
        Residual wavefront at the pupil plane after AO correction.
    aperture_mask : array
        Boolean mask for the telescope aperture on the pupil grid.
    focal_grid : hcipy.Grid
        The focal-plane grid.
    fiber_radius : float
        Fiber core radius in the focal plane [m].
    total_power_sent : float
        Total optical power sent by Alice.
    zero_magnitude_flux : float
        Photon flux for magnitude-0 star [photons/s].
    stellar_magnitude : float
        Apparent magnitude of the background star.
    fiber_solid_angle : float
        Solid angle subtended by the fiber [sr].
    dark_count_rate : float
        Detector dark count probability per time slot.

    Returns
    -------
    FSOCChannelMetrics
    """
    # --- Transmittance (eta) ---
    focal_r = np.sqrt(focal_grid.x**2 + focal_grid.y**2)
    fiber_mask = focal_r <= fiber_radius
    weights = focal_grid.weights
    power_map = _finite_array(wf_focal_corrected.power)
    # hcipy weights can be a scalar (uniform) or per-pixel array
    if np.ndim(weights) == 0:
        # Use unweighted sum to stay consistent with normalized power usage.
        pib = float(np.sum(power_map[fiber_mask]))
    else:
        weights = _finite_array(weights)
        pib = float(np.sum(power_map[fiber_mask] * weights[fiber_mask]))
    eta = _safe_float(pib / total_power_sent, 0.0) if total_power_sent > 0 else 0.0
    eta = min(eta, 1.0)

    # --- Strehl ratio ---
    psf_corrected = power_map
    psf_diffraction_limited = _finite_array(psf_diffraction_limited)
    max_dl = float(np.max(psf_diffraction_limited)) if psf_diffraction_limited.size else 0.0
    max_corr = float(np.max(psf_corrected)) if psf_corrected.size else 0.0
    strehl = _safe_float(max_corr / max_dl, 0.0) if max_dl > 0 else 0.0
    strehl = min(strehl, 1.0)

    # --- Phase variance ---
    phase = _finite_array(wf_pupil_residual.phase)
    mask = aperture_mask > 0.5
    if np.sum(mask) > 0:
        phase_masked = phase[mask]
        phase_variance = _safe_float(np.var(phase_masked), 0.0)
    else:
        phase_variance = 0.0

    # --- Depolarizing probability (Marechal approximation) ---
    depolarizing_prob = _safe_float(1.0 - np.exp(-phase_variance / 2.0), 0.0)

    # --- Background counts ---
    stellar_flux = zero_magnitude_flux * 10 ** (-stellar_magnitude / 2.5)
    background_counts = _safe_float(stellar_flux * fiber_solid_angle + dark_count_rate, 0.0)

    return FSOCChannelMetrics(
        eta=eta,
        strehl_ratio=strehl,
        phase_variance=phase_variance,
        depolarizing_prob=depolarizing_prob,
        background_counts=background_counts,
    )
