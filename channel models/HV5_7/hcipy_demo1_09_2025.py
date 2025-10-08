# -*- coding: utf-8 -*-
"""
HCIPy: Reproduce Fig. 3 (PSF & pupil images) numerically
--------------------------------------------------------
Generates 8 PNGs:
(a) PSF, PS1@P1   (b) PSF, PS2@P2   (c) PSF, PS3@P3   (d) PSF, all
(e) pupil, PS1@P1 (f) pupil, PS2@P2 (g) pupil, PS3@P3 (h) pupil, all
Plus a 4x2 mosaic in correct order.
"""

import numpy as np
import matplotlib.pyplot as plt
import hcipy as hci
from pathlib import Path

# -----------------------------
# 0) CONFIG (edit here)
# -----------------------------
D_sky      = 0.40            # [m] representative on-sky diameter
D_bench    = 2.0e-3          # [m] bench entrance pupil diameter
lam        = 0.5e-6         # [m] wavelength
# lam        = 1.55e-6         # [m] wavelength
Dr0_total  = 11.8            # D/r0 (target, strong turbulence)
r0_sky     = D_sky / Dr0_total
r0_bench   = D_bench / Dr0_total
sigma_I2_target = 2.138      # temporal scintillation index (for reference)
turbulance = True 

# Use measured L0/D (Table 4) and scale to bench
L0_over_D_measured = {"PS1": 3.945, "PS2": 8.585, "PS3": 16.592}
L0_bench = {k: v * D_bench for k, v in L0_over_D_measured.items()}  # [m]

# Bench propagation distances (Zi)
Z_bench = {"PS1": 0.0015, "PS2": 0.072, "PS3": 1.4}  # [m]

# Temporal scaling (tau = 100): 5 kHz on-sky ↔ 50 Hz bench
f_samp_sky   = 5_000.0  # [Hz]
f_samp_bench = 50.0     # [Hz]
tau          = f_samp_sky / f_samp_bench
t_exp        = 50e-6    # [s] negligible on bench
n_exp_subsamples = 1    # 1 = instantaneous

# Winds on bench (tau=100); directions arbitrary (paper doesn’t fix them)
winds = {
    "PS1": {"speed": 5.0e-4,  "theta": 0.0},
    "PS2": {"speed": 1.3e-3,  "theta": np.pi/3},
    "PS3": {"speed": 1.2e-2,  "theta": 2*np.pi/3},
}

# Pupil sampling (high to reduce ringing)
N_pupil = 256
supersample = 4
# N_pupil = 1024
# supersample = 6

# Focal plane sampling (in λ/D units)
q = 10                         # pixels per λ/D
fov_lambda_over_D = 30         # radius in λ/D

# Short-exposure frames per config (Fig.3 shows exemplars)
n_frames = 1

# Measured Cn^2 * dz (on-sky) → weights in -5/3 domain for r0 split
Cn2dz_sky = {"PS1": 2.563e-11, "PS2": 1.396e-11, "PS3": 1.015e-12}
Cn2dz_sum = sum(Cn2dz_sky.values())
r0_layer_weights = {k: v / Cn2dz_sum for k, v in Cn2dz_sky.items()}

# Output directory
outdir = Path("./fig3_outputs")
outdir.mkdir(parents=True, exist_ok=True)

# Colormaps / plotting
cmap_pupil = "plasma"   # colored pupil
cmap_psf   = "magma"
vmin_log_psf = -8.0     # fixed log10 range for all PSF panels
vmax_log_psf =  0.0

# -----------------------------
# 1) HELPERS
# -----------------------------
def split_r0_layers(r0_total, active_labels, weights=None):
    """Return dict r0_i with sum r0_i^{-5/3} = r0_total^{-5/3}.
       weights are fractions in the -5/3 domain keyed by labels."""
    labels = list(active_labels)
    if weights is None:
        w = np.ones(len(labels)) / len(labels)
    else:
        w = np.array([weights.get(lbl, 0.0) for lbl in labels], dtype=float)
        w = w / (w.sum() + 1e-16)
    inv_r0_total_53 = (1.0 / r0_total) ** (5.0 / 3.0)
    inv_r0_i_53 = w * inv_r0_total_53
    return {lbl: (1.0 / (inv_r0_i_53[k] ** (3.0 / 5.0))) for k, lbl in enumerate(labels)}

def make_layers_and_atmosphere(pupil_grid, active_labels, r0_i_dict):
    """Create HCIPy atmospheric layers and a MultiLayerAtmosphere (with scintillation)."""
    layers = []
    for lbl in active_labels:
        L0 = L0_bench[lbl]
        r0 = r0_i_dict[lbl]
        Cn2 = hci.Cn_squared_from_fried_parameter(r0, lam)
        vx = winds[lbl]["speed"] * np.cos(winds[lbl]["theta"])
        vy = winds[lbl]["speed"] * np.sin(winds[lbl]["theta"])
        layer = hci.InfiniteAtmosphericLayer(
            pupil_grid,
            Cn_squared=Cn2,
            L0=L0,
            velocity=(vx, vy),
            height=Z_bench[lbl]
        )
        layers.append(layer)
    layers_sorted = sorted(layers, key=lambda L: getattr(L, "height", 0.0))
    atmos = hci.MultiLayerAtmosphere(layers_sorted, scintillation=True)
    return layers_sorted, atmos

def apply_atmosphere(E0, atmos, lam):
    """Apply atmosphere and return complex field on exit plane."""
    wf_in  = hci.Wavefront(E0.copy(), lam)
    wf_out = atmos(wf_in)
    return wf_out.electric_field

def evolve_atmosphere(atmos, dt):
    """Frozen flow step."""
    atmos.evolve_until(atmos.t + dt)

def instantaneous_frame(aperture_field, atmos, lam, propagator_fraunhofer):
    """One instantaneous pupil & PSF frame (with scintillation)."""
    E_last = apply_atmosphere(aperture_field, atmos, lam)
    I_pupil = np.abs(E_last)**2
    wf_last = hci.Wavefront(E_last, lam)
    wf_foc  = propagator_fraunhofer(wf_last)
    I_psf   = wf_foc.intensity
    return I_pupil, I_psf

def save_field_png(field, grid, fname, title=None, log10=False,
                   vmin=None, vmax=None, cmap="gray",
                   circular_mask_radius=None):
    """Save a field on 'grid' as PNG; can mask outside a circle (for pupil)."""
    plt.figure(figsize=(5,4))
    arr = np.array(field, dtype=float)
    if log10:
        arr = np.log10(arr + 1e-16)

    # Optional circular mask (for pupil display)
    if circular_mask_radius is not None:
        rr = np.sqrt(grid.x**2 + grid.y**2)
        mask = (rr <= circular_mask_radius)
        arr = np.where(mask, arr, np.nan)

    hci.imshow_field(arr, grid=grid, cmap=cmap, interpolation="nearest",
                    vmin=vmin, vmax=vmax)
    if title:
        plt.title(title)
    plt.colorbar(shrink=0.8)
    plt.tight_layout()
    plt.savefig(fname, dpi=300)
    plt.close()

# -----------------------------
# 2) GRIDS & APERTURE
# -----------------------------
pupil_grid = hci.make_pupil_grid(N_pupil, D_bench)
aperture   = hci.circular_aperture(D_bench)
A          = hci.evaluate_supersampled(aperture, pupil_grid, supersample)  # amplitude

E_clean = A.astype(complex)

# Focal grid derived from the pupil grid (prevents "+" artifacts)
focal_grid = hci.make_focal_grid_from_pupil_grid(pupil_grid, q=q, num_airy=fov_lambda_over_D)
fraunhofer = hci.FraunhoferPropagator(pupil_grid, focal_grid)

# -----------------------------
# 3) CONFIGS
# -----------------------------
configs = {
    "PS1@P1": ["PS1"],
    "PS2@P2": ["PS2"],
    "PS3@P3": ["PS3"],
    "all":    ["PS1", "PS2", "PS3"],
}

# -----------------------------
# 4) RUN
# -----------------------------
results = {}
dt = 1.0 / f_samp_bench

print("r0 layer weights (-5/3 domain) from measured Cn^2*dz:", r0_layer_weights)

for cfg_name, active in configs.items():
    # split r0 across the active layers using measured weights
    r0_i = split_r0_layers(r0_bench, active, r0_layer_weights)

    # build atmosphere (scintillation=True)
    layers_sorted, atmos = make_layers_and_atmosphere(pupil_grid, active, r0_i)

    # warm-up (optional)
    evolve_atmosphere(atmos, dt)

    pupil_frames, psf_frames = [], []
    for _ in range(n_frames):
        I_pupil_acc = 0.0
        I_psf_acc   = 0.0
        for _s in range(n_exp_subsamples):
            evolve_atmosphere(atmos, dt / max(1, n_exp_subsamples))
            I_pupil, I_psf = instantaneous_frame(E_clean, atmos, lam, fraunhofer)
            I_pupil_acc += I_pupil
            I_psf_acc   += I_psf

        # exposure average (if subsamples>1)
        I_pupil = I_pupil_acc / n_exp_subsamples
        I_psf   = I_psf_acc   / n_exp_subsamples

        # normalize per-frame for visual comparability (peak=1)
        I_pupil = I_pupil / (I_pupil.max() + 1e-16)
        I_psf   = I_psf   / (I_psf.max()   + 1e-16)

        pupil_frames.append(I_pupil)
        psf_frames.append(I_psf)

    # aggregate SE frames (mean)
    I_pupil_mean = np.mean(pupil_frames, axis=0)
    I_psf_mean   = np.mean(psf_frames, axis=0)

    results[cfg_name] = {"I_pupil": I_pupil_mean, "I_psf": I_psf_mean, "r0_layers": r0_i}

    # Save PNGs (PSF in fixed log range; pupil masked as a colored disk)
    save_field_png(
        results[cfg_name]["I_psf"], focal_grid,
        outdir / f"{cfg_name}_PSF.png",
        title=f"{cfg_name} — short-exposure PSF",
        log10=True, vmin=vmin_log_psf, vmax=vmax_log_psf, cmap=cmap_psf
    )
    save_field_png(
        results[cfg_name]["I_pupil"], pupil_grid,
        outdir / f"{cfg_name}_pupil.png",
        title=f"{cfg_name} — pupil intensity",
        log10=False, cmap=cmap_pupil,
        circular_mask_radius=D_bench/2.0
    )

# -----------------------------
# 5) MOSAIC 4x2
# -----------------------------
mosaic_files = [
    # ("PS1@P1_PSF.png",   "PSF, PS1 at P1"),
    # ("PS2@P2_PSF.png",   "PSF, PS2 at P2"),
    # ("PS3@P3_PSF.png",   "PSF, PS3 at P3"),
    # ("all_PSF.png",      "PSF, all"),
    ("PS1@P1_pupil.png", "pupil, PS1 at P1"),
    ("PS2@P2_pupil.png", "pupil, PS2 at P2"),
    ("PS3@P3_pupil.png", "pupil, PS3 at P3"),
    ("all_pupil.png",    "pupil, all"),
]

fig, axes = plt.subplots(1, 4, figsize=(14, 7))
# fig, axes = plt.subplots(2, 4, figsize=(14, 7))
for idx, ax in enumerate(axes.ravel()):
    fname, ttl = mosaic_files[idx]
    img = plt.imread(outdir / fname)
    ax.imshow(img)
    ax.set_title(ttl, fontsize=10)
    ax.axis("off")
fig.suptitle(f"wavelength : {lam} , turbulance : {turbulance}")
plt.tight_layout()
plt.savefig(outdir / "Fig3_mosaic_4x2.png", dpi=200)
plt.close()

print(f"Done. PNGs written to: {outdir.resolve()}")
print("Per-layer r0 (bench) used per config:")
for k, v in results.items():
    print(k, v["r0_layers"])