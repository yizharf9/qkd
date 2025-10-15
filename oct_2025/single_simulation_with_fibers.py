"""
FSOC channel simulation using HCIPy
- Layer 1: atmospheric turbulence (InfiniteAtmosphericLayer)
- Layer 2: circular aperture with 4 spider legs (like the VLT aperture)
- Layer 3: optic (step-index) fiber coupling

This script creates each layer separately, saves diagnostic plots to a folder, and runs automatically.

Requirements:
  pip install hcipy numpy matplotlib pandas

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
    run_number= 0
try:
    wavelength
except NameError:
    wavelength = 1.234e-6
    wavelength = 1.55e-6
try:
    r0_ref
except NameError:
    r0_ref = 0.2
try:
    save_images
except NameError:
    save_images_prompt = input("Save images (y/n)? ")
    if save_images_prompt.lower() == "y":
        save_images = True
    elif save_images_prompt.lower() == "n":
        save_images = False
    else:
        exit("Not a valid input! Please enter 'y' or 'n'.")

num_pupil_pixels = 256
telescope_diameter = 8.0  # example for an 8-meter telescope like VLT

# ----------------------------- NEW: Directory Check -----------------------------
print("Checking current working directory...")
current_directory_name = os.path.basename(os.getcwd())
if current_directory_name == 'oct_2025':
    print(f"✅ Success: Script is running from the correct directory ('{current_directory_name}').")
else:
    print(f"⚠️ Warning: Script is NOT running from the 'oct_2025' directory.")
    print(f"   Current directory is: '{current_directory_name}'")
    exit("Execution stopped. Please run the script from the 'oct_2025' directory.")
print("-" * 60) # Visual separator
# -------------------------------------------------------------------------------

# Turbulence parameters (layer 1)
lambda_ref = 0.5e-6
r0 = r0_ref * (wavelength / lambda_ref) ** (6.0 / 5.0)
L0 = 25.0 # outer scale (m)

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
V_target = 2.0
singlemode_fiber_core_radius = (V_target * wavelength) / (2 * np.pi * fiber_NA)

# ----------------------------- OUTPUT DIR SECTION (CORRECTED) -----------------------------
base_output_dir = 'simulation_output'
# Using r0_ref in the folder name for consistency with logged data
run_specific_subdir = f"wavelength_{wavelength*1e6:.2f}um_r0_ref_{r0_ref*1e3:.1f}mm_run_{run_number}"
output_dir = os.path.join(base_output_dir, run_specific_subdir)
os.makedirs(output_dir, exist_ok=True)
# --------------------------------------------------------------------------

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
if save_images:
    save_field(initial_aperture, 'Pupil (VLT)', 'pupil_vlt')

# ----------------------------- Layer 1: Atmospheric turbulence -----------------------------
print('\n--- Layer 1: Atmospheric turbulence ---')
Cn_squared = Cn_squared_from_fried_parameter(r0, wavelength)
atm_layer = InfiniteAtmosphericLayer(pupil_grid, Cn_squared, L0, velocity)
atm_layer.evolve_until(0.0)
phase_screen = atm_layer.phase_for(wavelength)

print(f'Generated atmospheric phase screen. D/r0 = {telescope_diameter/r0:.2f}')
if save_images:
    save_field(phase_screen, 'Turbulence phase screen (radians)', 'turbulence_phase_screen')

# ----------------------------- Layer 2 & 3: Propagation and Fiber Coupling -----------------------------
print('\n--- Layers 2 & 3: Aperture propagation and Fiber coupling ---')
wf_after_turbulence = Wavefront(initial_aperture * np.exp(1j * phase_screen), wavelength)

num_focal_pixels = 256
D_focus = 2.1 * multimode_fiber_core_radius
focal_grid = make_pupil_grid(num_focal_pixels, D_focus)

lambda_norm_foc = 1.55e-6
focal_length = (telescope_diameter / (2 * fiber_NA)) * (lambda_norm_foc / wavelength)
propagator = FraunhoferPropagator(pupil_grid, focal_grid, focal_length=focal_length)

wf_foc = propagator(wf_after_turbulence)
print(f"Power before fiber coupling: {wf_foc.total_power:.6f}")

multi_mode_fiber = StepIndexFiber(multimode_fiber_core_radius, fiber_NA, fiber_length)
single_mode_fiber = StepIndexFiber(singlemode_fiber_core_radius, fiber_NA, fiber_length)

wf_mmf = multi_mode_fiber.forward(wf_foc)
wf_smf = single_mode_fiber.forward(wf_foc)

single_mode_power = wf_smf.total_power
multi_mode_power = wf_mmf.total_power

print(f'Multi-mode fiber throughput: {multi_mode_power:.6f}')
print(f'Single-mode fiber throughput: {single_mode_power:.6f}')

# ----------------------------- SAVING STAGE -----------------------------
if save_images:
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
    print(f'\nSimulation complete. All images saved to: {output_dir}')
else:
    print('\nSimulation complete. No images saved.')

# ----------------------------- DATA STORAGE STAGE -----------------------------
def update_csv(wavelength, r0_ref_val, run_num, sm_power, mm_power):
    import pandas as pd

    file_path = "./massive_output.csv"
    columns = ["wavelength", "r0_ref", "run_number", "single_mode_power", "multi_mode_power", "time"]

    new_row_df = pd.DataFrame([{
        "wavelength": wavelength,
        "r0_ref": r0_ref_val,
        "run_number": run_num,
        "single_mode_power": sm_power,
        "multi_mode_power": mm_power,
        "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }])

    if os.path.exists(file_path):
        new_row_df.to_csv(file_path, mode='a', header=False, index=False)
    else:
        new_row_df.to_csv(file_path, mode='w', header=columns, index=False)
    print("Operation complete. Data has been saved.")

update_csv(wavelength, r0_ref, run_number, single_mode_power, multi_mode_power)