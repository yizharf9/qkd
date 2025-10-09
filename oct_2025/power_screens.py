# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import hcipy as hci

# ========== 1) Pupil / Telescope ==========
telescope_diameter = 8.0       # [m]
central_obscuration = 1.2      # [m]
central_obscuration_ratio = central_obscuration / telescope_diameter
spider_width = 0.05            # [m]

oversizing_factor = 16.0 / 15.0
num_pupil_pixels_base = 240
num_pupil_pixels = int(num_pupil_pixels_base * oversizing_factor)
pupil_grid_diameter = telescope_diameter * oversizing_factor
pupil_grid = hci.make_pupil_grid(num_pupil_pixels, pupil_grid_diameter)

VLT_ap_gen = hci.make_obstructed_circular_aperture(
    telescope_diameter,
    central_obscuration_ratio,
    num_spiders=4,
    spider_width=spider_width
)
VLT_aperture = hci.evaluate_supersampled(VLT_ap_gen, pupil_grid, 4)

# ========== 2) Wavefront & Focal Grid ==========
wavelength_sci = 2.2e-6  # [m]
wf = hci.Wavefront(VLT_aperture.copy(), wavelength_sci)
wf.total_power = 1.0  # normalize pupil power

spatial_resolution = wavelength_sci / telescope_diameter  # [rad] ~ λ/D
focal_grid = hci.make_focal_grid(q=4, num_airy=30, spatial_resolution=spatial_resolution)

propagator = hci.FraunhoferPropagator(pupil_grid, focal_grid)
E_img = propagator.forward(wf)   # complex field at focal plane

# ⬇️ Use the 'power' PROPERTY (Field), not a method
I_img = E_img.power              # intensity on focal_grid (PSF), same as |E|^2

# ========== 3) 3×3 image-plane bucket (centered) ==========
I2 = I_img.shaped
ny, nx = focal_grid.shape
assert ny >= 3 and nx >= 3

cy, cx = ny // 2, nx // 2
rows = slice(cy - 1, cy + 2)
cols = slice(cx - 1, cx + 2)

w = focal_grid.weights
if np.ndim(w) == 0 or (hasattr(w, "size") and w.size == 1):
    # uniform pixel area (scalar weight)
    wpix = float(w)
    power_total  = wpix * I2.sum()
    power_bucket = wpix * I2[rows, cols].sum()
else:
    # per-pixel weights
    w2 = w.reshape(focal_grid.shape)
    power_total  = float((I2 * w2).sum())
    power_bucket = float((I2[rows, cols] * w2[rows, cols]).sum())

frac = power_bucket / power_total if power_total > 0 else np.nan
print(f"[IMAGE 3×3] captured = {power_bucket:.6e}  ({frac:.2%} of total {power_total:.6e})")

# ========== 4) Plots: pupil + PSF with bucket ==========
mask2 = np.zeros_like(I2, dtype=float)
mask2[rows, cols] = 1.0

fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))

# Left: pupil
hci.imshow_field(VLT_aperture, cmap='gray', ax=axes[0])
axes[0].set_title('Pupil (VLT)')
axes[0].set_xlabel('pupil x [m]')
axes[0].set_ylabel('pupil y [m]')

# Right: PSF (log10) + 3×3 bucket overlay
im1 = axes[1].imshow(np.log10(I2 / I2.max()), origin='lower', cmap='inferno', vmin=-6)
axes[1].contour(mask2, levels=[0.5], linewidths=1.8)
axes[1].set_title(f'PSF (log10) with 3×3 bucket\nCaptured: {power_bucket:.3e} ({frac:.2%})')
axes[1].set_xlabel('focal x [rad]')
axes[1].set_ylabel('focal y [rad]')
plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

plt.tight_layout()
plt.show()
