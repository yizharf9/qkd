"""
FSOC channel simulation using HCIPy
- Layer 1: atmospheric turbulence (InfiniteAtmosphericLayer)
- Layer 2: circular aperture with 4 spider legs (like the VLT aperture)
- Layer 3: optic (step-index) fiber coupling

This script creates each layer separately, saves diagnostic plots to a folder, and runs automatically.

Requirements:
  pip install hcipy numpy matplotlib

Run:
  python fsoc_hcipy_simulation.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import datetime

try:
    from hcipy import *
except Exception as e:
    raise ImportError("HCIPy is required. Install it with: pip install hcipy") from e

# ----------------------------- Parameters -----------------------------
try:
    run_number
except NameError:
    run_number= 1
try:
    wavelength
except NameError:
    wavelength = 1.55e-6  # 1 micronx
num_pupil_pixels = 256
telescope_diameter = 8.0  # example for an 8-meter telescope like VLT

# Turbulence parameters (layer 1)
try:
    r0_ref
except NameError:
    r0_ref = 0.2 
lambda_ref = 0.5e-6        # Fried parameter (m)
r0 = r0_ref * (wavelength / lambda_ref) ** (6.0 / 5.0)
L0 = 25.0
          # outer scale (m)
Cn2_layer = Cn_squared_from_fried_parameter(r0, wavelength) if 'Cn_squared_from_fried_parameter' in globals() else None
velocity = (10.0, 0.0)  # m/s horizontal wind

# Aperture / spiders parameters (layer 2)
central_obscuration_ratio = 0.15  # realistic central obstruction for VLT
spider_width = 0.05  # fraction of pupil diameter

# Fiber parameters (layer 3)
multimode_fiber_core_radius = 25e-6  # 25 micron
fiber_NA = 0.13
fiber_length = 10.0

# Keep single-mode fiber truly single-mode across wavelengths by fixing V-number
# V = 2*pi*a*NA/lambda. Choose V_target < 2.405 (cutoff); use 2.0 for margin.
V_target = 2.0
singlemode_fiber_core_radius = (V_target * wavelength) / (2 * np.pi * fiber_NA)

# Output folder
output_dir = 'simulation_output_'+f"wavelength_{wavelength*1e6:.2f}um_"+f"r0_{r0*1e3:.1f}mm_"+f"run_{run_number}"
os.makedirs(output_dir, exist_ok=True)

# ----------------------------- Helper plotting -----------------------------
def save_field(field, title, filename):
    plt.figure(figsize=(5, 5))
    imshow_field(field)
    plt.title(f"{title} (λ={wavelength*1e6:.2f}µm)")
    plt.colorbar()
    plt.tight_layout()
    filepath = os.path.join(output_dir, f"{filename}_lambda_{wavelength*1e6:.2f}um.png")
    plt.savefig(filepath, dpi=300)
    plt.close()

# ----------------------------- Create grids & initial wavefront -----------------------------
print('Setting up pupil grid and initial wavefront...')
pupil_grid = make_pupil_grid(num_pupil_pixels, telescope_diameter)

# Generate the VLT-like aperture with central obscuration and 4 spider legs
vlt_aperture_function = make_obstructed_circular_aperture(
    telescope_diameter,
    central_obscuration_ratio,
    num_spiders=4,
    spider_width=spider_width
)

initial_aperture = vlt_aperture_function(pupil_grid)
wf0 = Wavefront(initial_aperture, wavelength)
wf0.total_power = 1.0

print('Initial wavefront and pupil ready.')
save_field(initial_aperture, 'Pupil (VLT)', 'pupil_vlt')

# ----------------------------- Layer 1: Atmospheric turbulence -----------------------------
print('\n--- Layer 1: Atmospheric turbulence ---')
try:
    if Cn2_layer is None:
        Cn_squared = Cn_squared_from_fried_parameter(r0, wavelength)
    else:
        Cn_squared = Cn2_layer
except Exception:
    Cn_squared = 1e-15

atm_layer = InfiniteAtmosphericLayer(pupil_grid, Cn_squared, L0, velocity)
atm_layer.evolve_until(0.0)
phase_screen = atm_layer.phase_for(wavelength)

wf_turb = Wavefront(initial_aperture * np.exp(1j * phase_screen), wavelength)
print("wf_turb.total_power: ", wf_turb.total_power)
print('Generated atmospheric phase screen (radians).')
save_field(phase_screen, 'Turbulence phase screen (radians)', 'turbulence_phase_screen')

# ----------------------------- Layer 2: aperture with 4 spider legs -----------------------------
print('\n--- Layer 2: Circular aperture + 4 spider legs (VLT-like) ---')
aperture_with_spiders = vlt_aperture_function(pupil_grid)

wf_apertured = Wavefront(aperture_with_spiders * np.exp(1j * phase_screen), wavelength)
print("wf_apertured.total_power: ", wf_apertured.total_power)
print('Aperture with spiders created and applied to wavefront.')
# save_field(aperture_with_spiders, 'VLT-like Aperture (amplitude mask)', 'vlt_aperture_mask')

# ----------------------------- Layer 3: optic fiber coupling -----------------------------
print('\n--- Layer 3: Optic fiber (step-index) ---')
num_focal_pixels = 256
D_focus = 2.1 * multimode_fiber_core_radius
focal_grid = make_pupil_grid(num_focal_pixels, D_focus)

# Normalize focal length vs wavelength so the fiber-plane PSF size stays comparable
lambda_ref = 1.55e-6
focal_length = (telescope_diameter / (2 * fiber_NA)) * (lambda_ref / wavelength)
propagator = FraunhoferPropagator(pupil_grid, focal_grid, focal_length=focal_length)

wf_foc = propagator(Wavefront(aperture_with_spiders * np.exp(1j * phase_screen), wavelength))
#wf_foc.total_power = 1.0
print("wf_foc.total_power: ", wf_foc.total_power)
multi_mode_fiber = StepIndexFiber(multimode_fiber_core_radius, fiber_NA, fiber_length)
single_mode_fiber = StepIndexFiber(singlemode_fiber_core_radius, fiber_NA, fiber_length)

wf_mmf = multi_mode_fiber.forward(wf_foc)
wf_smf = single_mode_fiber.forward(wf_foc)

single_mode_power = wf_smf.total_power
multi_mode_power = wf_mmf.total_power

print(f'Multi-mode fiber throughput: {wf_mmf.total_power:.6f}')
print(f'Single-mode fiber throughput: {wf_smf.total_power:.6f}')

plt.figure(figsize=(12, 5))
plt.subplot(1, 3, 1)
imshow_field(wf_foc.power)
plt.title(f'Focused intensity (to fiber) (λ={wavelength*1e6:.2f}µm)')

plt.subplot(1, 3, 2)
imshow_field(wf_mmf.power)
plt.title(f'Multi-mode fiber output power (λ={wavelength*1e6:.2f}µm)')

plt.subplot(1, 3, 3)
imshow_field(wf_smf.power)
plt.title(f'Single-mode fiber output power (λ={wavelength*1e6:.2f}µm)')
plt.tight_layout()

fiber_output_path = os.path.join(output_dir, f"fiber_outputs_lambda_{wavelength*1e6:.2f}um.png")
plt.savefig(fiber_output_path, dpi=300)
plt.close()

print('\nSimulation complete. All images saved to the simulation_output folder.')





def update_csv(wavelength,r0,run_number,single_mode_power,multi_mode_power):
    import pandas as pd
    import os

    # Define the file path and column names
    file_path = "./massive_output.csv"
    columns = ["wavelength", "r0", "run_number", "single_mode_power", "multi_mode_power","time"]

    # Create a new DataFrame row from your variables
    # Note: The values are placed in a list.
    new_row_df = pd.DataFrame([{
        "wavelength": wavelength,
        "r0": r0,
        "run_number": run_number,
        "single_mode_power": single_mode_power,
        "multi_mode_power": multi_mode_power,
        "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }])

    # Check if the file already exists
    if os.path.exists(file_path):
        print(f"File '{file_path}' found. Appending new data...")
        # Append the new row to the existing file without writing the header
        new_row_df.to_csv(file_path, mode='a', header=False, index=False)
    else:
        print(f"File '{file_path}' not found. Creating a new file...")
        # Create a new file and write the new row, including the header
        new_row_df.to_csv(file_path, mode='w', header=columns, index=False)

    print("Operation complete. Data has been saved.")

    # Optional: You can read and print the file's content to verify
    # print("\nCurrent file content:")
    # print(pd.read_csv(file_path))
    
update_csv(wavelength,r0,run_number,single_mode_power,multi_mode_power)