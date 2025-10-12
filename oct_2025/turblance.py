"""
FSOC channel simulation using HCIPy
- Layer 1: atmospheric turbulence (InfiniteAtmosphericLayer)
- Layer 2: circular aperture with 4 spider legs
- Layer 3: optic (step-index) fiber coupling

This script creates each layer separately, shows a quick diagnostic plot, and then
waits for the user to confirm (via terminal input) before proceeding to the next layer.

Requirements:
  pip install hcipy numpy matplotlib

Run:
  python fsoc_hcipy_simulation.py

"""

import numpy as np
import matplotlib.pyplot as plt

try:
    from hcipy import *
except Exception as e:
    raise ImportError("HCIPy is required. Install it with: pip install hcipy") from e

# ----------------------------- Parameters -----------------------------
wavelength = 1e-6  # 1 micron
num_pupil_pixels = 256
D_pupil = 1.0       # normalized pupil diameter (meters in absolute units if you like)

# Turbulence parameters (layer 1)
r0 = 0.15          # Fried parameter (m) -- example value
L0 = 25.0          # outer scale (m)
Cn2_layer = Cn_squared_from_fried_parameter(r0, wavelength) if 'Cn_squared_from_fried_parameter' in globals() else None
velocity = (10.0, 0.0)  # m/s horizontal wind

# Aperture / spiders parameters (layer 2)
central_obscuration_ratio = 0.0
num_spiders = 4
spider_width = 0.02  # fraction of pupil diameter

# Fiber parameters (layer 3)
multimode_fiber_core_radius = 25e-6  # 25 micron
singlemode_fiber_core_radius = 2e-6 # 2 micron
fiber_NA = 0.13
fiber_length = 10.0

# ----------------------------- Helper plotting -----------------------------
def show_field(field, title=None):
    plt.figure(figsize=(5,5))
    imshow_field(field)
    if title:
        plt.title(title)
    plt.colorbar()
    plt.tight_layout()
    plt.show()

# ----------------------------- Create grids & initial wavefront -----------------------------
print('Setting up pupil grid and initial wavefront...')
pupil_grid = make_pupil_grid(num_pupil_pixels, D_pupil)

# initial plane wave over full pupil
initial_aperture = circular_aperture(D_pupil)(pupil_grid)
wf0 = Wavefront(initial_aperture, wavelength)
wf0.total_power = 1.0

print('Initial wavefront and pupil ready.')
show_field(wf0.aberration if hasattr(wf0, 'aberration') else initial_aperture, 'Initial pupil (amplitude)')

input('Layer 1: Press Enter to generate atmospheric turbulence layer and apply it to the wavefront...')

# ----------------------------- Layer 1: Atmospheric turbulence -----------------------------
print('\n--- Layer 1: Atmospheric turbulence ---')
# Convert r0 to Cn^2 where possible using helper from HCIPy doc if available
try:
    if Cn2_layer is None:
        Cn_squared = Cn_squared_from_fried_parameter(r0, wavelength)
    else:
        Cn_squared = Cn2_layer
except Exception:
    # fallback: pick a reasonable Cn_squared for demo
    Cn_squared = 1e-15

# Create an infinite atmospheric layer on the pupil grid
atm_layer = InfiniteAtmosphericLayer(pupil_grid, Cn_squared, L0, velocity)
# evolve it to a time so that phase is generated
atm_layer.evolve_until(0.0)
phase_screen = atm_layer.phase_for(wavelength)

# apply turbulence as a multiplicative phase to wavefront
wf_turb = Wavefront(initial_aperture * np.exp(1j * phase_screen), wavelength)
wf_turb.total_power = 1.0

print('Generated atmospheric phase screen (radians).')
show_field(phase_screen, 'Turbulence phase screen (radians)')

input('Turbulence generated. Press Enter to continue to Layer 2 (aperture with spiders)...')

# ----------------------------- Layer 2: aperture with 4 spider legs -----------------------------
print('\n--- Layer 2: Circular aperture + 4 spider legs ---')
# Use helper in HCIPy to create an obstructed circular aperture
try:
    aperture_with_spiders = make_obstructed_circular_aperture(D_pupil, central_obscuration_ratio, num_spiders=num_spiders, spider_width=spider_width)(pupil_grid)
except Exception:
    # fallback: build aperture from circular aperture and add spiders manually
    aperture_with_spiders = circular_aperture(D_pupil)(pupil_grid)
    # create four spiders by using make_spider
    angles = np.linspace(0, 2*np.pi, num_spiders, endpoint=False)
    for ang in angles:
        p1 = np.array([-0.5*np.cos(ang), -0.5*np.sin(ang)]) * D_pupil
        p2 = np.array([ 0.5*np.cos(ang),  0.5*np.sin(ang)]) * D_pupil
        spider = make_spider(p1, p2, spider_width * D_pupil)(pupil_grid)
        aperture_with_spiders = aperture_with_spiders * (1 - spider)

# Apply the aperture to the turbulent wavefront
wf_apertured = Wavefront(aperture_with_spiders * np.exp(1j * phase_screen), wavelength)
wf_apertured.total_power = 1.0

print('Aperture with spiders created and applied to wavefront.')
show_field(aperture_with_spiders, 'Aperture with 4 spiders (amplitude mask)')

input('Aperture generated. Press Enter to continue to Layer 3 (optic fiber coupling)...')

# ----------------------------- Layer 3: optic fiber coupling -----------------------------
print('\n--- Layer 3: Optic fiber (step-index) ---')
# Setup propagator from pupil to focus to match fiber NA
# Choose focal grid size based on fiber core radius
num_focal_pixels = 256
D_focus = 2.1 * multimode_fiber_core_radius
focal_grid = make_pupil_grid(num_focal_pixels, D_focus)

# focal length chosen so the f-number matches fiber_NA (focal_length = D_pupil/(2*NA))
focal_length = D_pupil / (2 * fiber_NA)
propagator = FraunhoferPropagator(pupil_grid, focal_grid, focal_length=focal_length)

wf_foc = propagator(Wavefront(aperture_with_spiders * np.exp(1j*phase_screen), wavelength))
wf_foc.total_power = 1.0

# Create fibers
multi_mode_fiber = StepIndexFiber(multimode_fiber_core_radius, fiber_NA, fiber_length)
single_mode_fiber = StepIndexFiber(singlemode_fiber_core_radius, fiber_NA, fiber_length)

# Propagate into fibers
wf_mmf = multi_mode_fiber.forward(wf_foc)
wf_smf = single_mode_fiber.forward(wf_foc)

print(f'Multi-mode fiber throughput: {wf_mmf.total_power:.6f}')
print(f'Single-mode fiber throughput: {wf_smf.total_power:.6f}')

# Show results in focal plane and fiber outputs
plt.figure(figsize=(12,5))
plt.subplot(1,3,1)
imshow_field(wf_foc.power)
plt.title('Focused intensity (to fiber)')

plt.subplot(1,3,2)
imshow_field(wf_mmf.power)
plt.title('Multi-mode fiber output power')

plt.subplot(1,3,3)
imshow_field(wf_smf.power)
plt.title('Single-mode fiber output power')
plt.tight_layout()
plt.show()

print('\nSimulation complete. You can adjust parameters at the top of the script and re-run.')
