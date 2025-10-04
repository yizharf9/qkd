import numpy as np
import matplotlib.pyplot as plt
import hcipy as hci
# Parameters (edit freely)
# -----------------------------
telescope_diameter = 8.0       # [m]
central_obscuration = 1.2      # [m]
central_obscuration_ratio = central_obscuration / telescope_diameter
spider_width = 0.05            # [m]
oversizing_factor = 16.0 / 15.0

# Sampling like SPHERE WFS (240 px) with oversizing -> 256 px
num_pupil_pixels = int(240 * oversizing_factor)
pupil_grid_diameter = telescope_diameter * oversizing_factor
pupil_grid = hci.make_pupil_grid(num_pupil_pixels, pupil_grid_diameter)

# Wavelengths
wavelength_sci = 500e-9

# Turbulence settings
r0_at_500nm = 0.10             # [m] Fried parameter at 500 nm (smaller => stronger turb)
L0 = 25.0                      # [m] outer scale (np.inf for pure Kolmogorov)
wind_speed = 10.0              # [m/s]
layer_height = 0.0             # [m] single phase layer for illustration
rng_seed = 1234

# Focal sampling
q = 4                          # pixels per lambda/D
num_airy = 30                  # field of view radius in lambda/D


# -----------------------------
# Build pupil (VLT-like)
# -----------------------------
VLT_aperture_generator = hci.make_obstructed_circular_aperture(
    telescope_diameter,
    central_obscuration_ratio,
    num_spiders=4,
    spider_width=spider_width
)
# supersampling factor 4 like in the tutorial
VLT_aperture = hci.evaluate_supersampled(VLT_aperture_generator, pupil_grid, 4)

# Incoming wavefront at science wavelength
wf_clean = hci.Wavefront(VLT_aperture.copy(), wavelength_sci)
wf_clean.total_power = 1.0

# Focal grid and Fraunhofer propagator
spatial_resolution = wavelength_sci / telescope_diameter
focal_grid = hci.make_focal_grid(q=q, num_airy=num_airy, spatial_resolution=spatial_resolution)
propagator = hci.FraunhoferPropagator(pupil_grid, focal_grid)

# Diffraction-limited PSF
img_clean = propagator(wf_clean)
I_clean = img_clean.intensity
#I_clean /= I_clean.max()

# -----------------------------
# Turbulence layer (version-robust API)
# -----------------------------

# Convert to Cn^2 at 500 nm as required by HCIPy helper
Cn2 = hci.Cn_squared_from_fried_parameter(r0_at_500nm, 500e-9)
layer = hci.InfiniteAtmosphericLayer(
    pupil_grid, Cn2, L0, wind_speed, height=layer_height)

# Apply phase to wavefront (single snapshot)
wf_turb = layer(wf_clean.copy())
phase = layer.phase_for(wavelength_sci)

# PSF with turbulence
img_turb = propagator(wf_turb)
I_turb = img_turb.intensity
#I_turb /= I_turb.max()

# -----------------------------
# Plots (match the style you showed: log10 PSF)
# -----------------------------
plt.figure(figsize=(12,10))

plt.subplot(2,2,1)
plt.title('VLT-like pupil amplitude')
hci.imshow_field(wf_clean.amplitude, grid=pupil_grid, cmap='gray', interpolation='nearest')
plt.xlabel('x [m]'); plt.ylabel('y [m]')
plt.colorbar(shrink=0.8)

plt.subplot(2,2,2)
plt.title('Turbulence phase on pupil [rad]')
hci.imshow_field(phase, grid=pupil_grid, cmap='twilight', interpolation='nearest')
plt.xlabel('x [m]'); plt.ylabel('y [m]')
plt.colorbar(shrink=0.8)

plt.subplot(2,2,3)
plt.title('log10 PSF (no turbulence)')
hci.imshow_field(np.log10(I_clean + 1e-16), grid=focal_grid, cmap='inferno', interpolation='nearest')
print("log10 PSF (no turbulence):"+"\n"+str(np.log10(I_clean + 1e-16)))
plt.colorbar(shrink=0.8)

plt.subplot(2,2,4)
plt.title('log10 PSF (with turbulence)')
hci.imshow_field(np.log10(I_turb + 1e-16), grid=focal_grid, cmap='inferno', interpolation='nearest')
print("log10 PSF (with turbulence):"+"\n"+str(np.log10(I_clean + 1e-16)))

plt.colorbar(shrink=0.8)

plt.tight_layout()
plt.show()

