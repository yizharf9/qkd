# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import hcipy as hci

# =================== 1) PUPIL / TELESCOPE ===================
telescope_diameter = 8.0       # [m]
central_obscuration = 1.2      # [m]
central_obscuration_ratio = central_obscuration / telescope_diameter
spider_width = 0.05            # [m]

# Oversizing for alias control in FFT pupil->focal propagation
oversizing_factor = 16.0 / 15.0
num_pupil_pixels_base = 240
num_pupil_pixels = int(num_pupil_pixels_base * oversizing_factor)  # must be int
pupil_grid_diameter = telescope_diameter * oversizing_factor
pupil_grid = hci.make_pupil_grid(num_pupil_pixels, pupil_grid_diameter)

# VLT-like aperture (circular with central obstruction and 4 spiders)
VLT_aperture_generator = hci.make_obstructed_circular_aperture(
    telescope_diameter,
    central_obscuration_ratio,
    num_spiders=4,
    spider_width=spider_width
)
# Supersample to reduce edge aliasing
VLT_aperture = hci.evaluate_supersampled(VLT_aperture_generator, pupil_grid, 4)

# =================== 2) WAVEFRONT AND FOCAL GRID ===================
wavelength_wfs = 0.7e-6   # [m] (not used below, kept for context)
wavelength_sci = 2.2e-6   # [m]

# Complex wavefront across the pupil; amplitude = aperture, flat phase
wf = hci.Wavefront(VLT_aperture.copy(), wavelength_sci)
wf.total_power = 1.0  # normalize total power at pupil plane

# Focal grid in angular coordinates (radians on-sky)
# spatial_resolution should be ~ λ/D (radians), q = pixels per λ/D, num_airy = field half-width in Airy radii
spatial_resolution = wavelength_sci / telescope_diameter
focal_grid = hci.make_focal_grid(q=4, num_airy=30, spatial_resolution=spatial_resolution)

# Fraunhofer propagation to the focal plane
propagator = hci.FraunhoferPropagator(pupil_grid, focal_grid)
E_img = propagator.forward(wf)          # complex field
I_img = E_img.power                     # intensity Field on focal_grid (PSF)

# =================== 3) 3×3 IMAGE-PLANE BUCKET (CENTERED) ===================
# Work in 2D for pixel selection
I2 = I_img.shaped
ny, nx = focal_grid.shape
assert ny >= 3 and nx >= 3, "Focal grid must be at least 3x3 for a 3x3 bucket."

# Center indices
cy, cx = ny // 2, nx // 2
rows = slice(cy - 1, cy + 2)
cols = slice(cx - 1, cx + 2)

# Build a 3x3 mask as a Field on focal_grid
mask2 = np.zeros_like(I2, dtype=float)
mask2[rows, cols] = 1.0
bucket_mask = hci.Field(mask2.ravel(), focal_grid)

# Energy integrals (area-correct), robust to scalar/1D weights



# =================== 4) FIGURES: PUPIL + PSF WITH BUCKET ===================
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))

# Left: pupil
hci.imshow_field(VLT_aperture, cmap='gray', ax=axes[0])
axes[0].set_title('Pupil (VLT)')
axes[0].set_xlabel('pupil x [m]')
axes[0].set_ylabel('pupil y [m]')

# Right: PSF (log10) + 3×3 bucket outline
im1 = axes[1].imshow(np.log10(I2 / I2.max()), origin='lower',
                     cmap='inferno', vmin=-6)
# Overlay the 3×3 boundary as a contour
axes[1].contour(mask2, levels=[0.5], linewidths=1.8)
axes[1].set_xlabel('focal x [rad]')
axes[1].set_ylabel('focal y [rad]')
plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

plt.tight_layout()

plt.show()

# =================== 5) (OPTIONAL) OFF-AXIS BUCKET EXAMPLE ===================
# If later you want the 3×3 bucket around (x0, y0) in radians:
# x, y = focal_grid.as_('cartesian').coords
# i_near = np.argmin((x - x0)**2 + (y - y0)**2)
# cy2, cx2 = np.unravel_index(i_near, (ny, nx))
# cy2 = np.clip(cy2, 1, ny-2); cx2 = np.clip(cx2, 1, nx-2)
# rows2, cols2 = slice(cy2-1, cy2+2), slice(cx2-1, cx2+2)
# mask2_off = np.zeros_like(I2); mask2_off[rows2, cols2] = 1.0
# bucket_mask_off = hci.Field(mask2_off.ravel(), focal_grid)
# power_bucket_off = float(hci.integrate_field(I_img * bucket_mask_off))
# print("Off-axis 3×3 captured:", power_bucket_off, "fraction:", power_bucket_off/power_total)
