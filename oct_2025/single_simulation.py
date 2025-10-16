import os
import numpy as np
import matplotlib.pyplot as plt
import datetime
try:
    from hcipy import *
except Exception as e:
    raise ImportError("HCIPy is required. Install it with: pip install hcipy") from e

# ----------------------------- Directory Check -----------------------------
print("Checking current working directory...")
current_directory_name = os.path.basename(os.getcwd())
if current_directory_name == 'oct_2025':
    print(f"✅ Success: Script is running from the correct directory ('{current_directory_name}').")
else:
    print(f"⚠️ Warning: Script is NOT running from the 'oct_2025' directory.")
    print(f"   Current directory is: '{current_directory_name}'")
    exit("Execution stopped. Please run the script from the 'oct_2025' directory.")
print("-" * 60) # Visual separator
# ----------------------------- Parameters -----------------------------
try:
    run_number
except NameError:
    run_number= 26
try:
    wavelength
except NameError:
    wavelength = 1.55e-6
try:
    r0_ref
except NameError:
    r0_ref = 0.1
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

# ---------- 1) Pupil (VLT-like) ----------
D = 8.0                 # [m]
D_obs = 1.2             # [m]
eps = D_obs / D # central obscuration ratio
spider_w = 0.05         # [m]
oversz = 16.0/15.0
N = int(240 * oversz) #256 # number of pixels across pupil diameter
pupil_grid = make_pupil_grid(N, D * oversz)

ap_gen = make_obstructed_circular_aperture(D, eps, num_spiders=4, spider_width=spider_w)
ap = evaluate_supersampled(ap_gen, pupil_grid, 4) #!

# ---------- 2) Wavelength & focal grid ----------
# wavelength = 2.2e-6  # [m] science wavelength
q = 4
num_airy = 30
spatial_res = wavelength / D  # [rad] per λ/D
focal_grid = make_focal_grid(q=q, num_airy=num_airy, spatial_resolution=spatial_res)
prop = FraunhoferPropagator(pupil_grid, focal_grid) # transforms from pupil_grid to focal_grid

# ---------- 3) Wavefront (before turbulence) ----------
wf0 = Wavefront(ap, wavelength)
wf0.total_power = 1.0
psf0 = prop(wf0).power  # unaberrated PSF (relative units)

# ---------- 4) Single-layer turbulence ----------
seeing = 0.6       # [arcsec] @ 500 nm
L0 = 40.0          # [m]
tau0 = 5e-3        # [s]
lam_ref = 500e-9   # [m]

r0 = r0_ref * (wavelength / lam_ref) ** (6.0 / 5.0)    #! overlaps original file
# r0 = seeing_to_fried_parameter(seeing)                # [m] at 500 nm
Cn2 = Cn_squared_from_fried_parameter(r0, lam_ref)    # [m^(-2/3)]
v = 0.314 * r0 / tau0                                     # [m/s] #! try to run with v = 0

# HCIPy 0.7.0 API fallback
try:
    layer = InfiniteAtmosphericLayer(pupil_grid, Cn2, L0, v)
except TypeError:
    layer = InfiniteAtmosphericLayer(Cn2, L0, v)

wf1 = layer(wf0)                 # wavefront AFTER turbulence
psf1 = prop(wf1).power           # instantaneous PSF with turbulence

# ---------- 5) Bucket definition: 9 µm circle in focal plane ----------
Fnum_sci = 50.0                 # adjust to your optics if needed #! understand what this mean...
f_eff = Fnum_sci * D            # [m] effective focal length
diam_phys = 9e-6                # [m]
rad_phys  = diam_phys / 2.0     # [m]
alpha = rad_phys / f_eff        # [rad] angular radius on focal plane

theta = np.sqrt(focal_grid.x**2 + focal_grid.y**2)  # [rad] #! how this happens?
bucket_mask = (theta <= alpha).astype(float)

# ---------- 6) Proper integration with grid weights (HCIPy 0.7.0) ----------
# w = focal_grid.weights #! understand signifficance
w = 1

power0_in_bucket = np.sum(psf0 * bucket_mask * w)
power0_total     = np.sum(psf0 * w)
frac0 = power0_in_bucket / power0_total

power1_in_bucket = np.sum(psf1 * bucket_mask * w)
power1_total     = np.sum(psf1 * w)
frac1 = power1_in_bucket / power1_total

# ---------- 7) Print results ----------
print("=== Bucket (9 µm diameter) @ focal plane ===")
print(f"F/# = {Fnum_sci:.1f},  f_eff = {f_eff:.3f} m")
print(f"Circle radius (phys): {rad_phys:.3e} m")
print(f"Circle radius (ang):  {alpha:.3e} rad  (~{alpha*206265:.3f} arcsec)")
print("\n-- BEFORE turbulence (unaberrated) --")
print(f"Power in bucket:  {power0_in_bucket:.6e} (relative)")
print(f"Total power:      {power0_total:.6e} (relative)")
print(f"Fractional power: {frac0:.6%}")
print("\n-- AFTER turbulence (instantaneous) --")
print(f"Power in bucket:  {power1_in_bucket:.6e} (relative)")
print(f"Total power:      {power1_total:.6e} (relative)")
print(f"Fractional power: {frac1:.6%}")

# ---------- 8) Optional: overlay circle on both PSFs ----------
if save_images:
    import matplotlib.patches as mpatches
    from mpl_toolkits.axes_grid1 import make_axes_locatable
#yassume: psf is an HCIPy Field on a focal grid with angular coords (rad)
fg = psf0.grid                        # same grid for psf1
f_m = f_eff          # <-- set this to your f

# compute image extents in meters using the grid limits (in rad) times f
xmin_m, xmax_m = fg.x.min()*f_m, fg.x.max()*f_m
ymin_m, ymax_m = fg.y.min()*f_m, fg.y.max()*f_m

# plot using plain matplotlib imshow with 'extent' in meters
fig, axes = plt.subplots(1,2, figsize=(9,4), dpi=150)
fig, axes = plt.subplots(1, 2, figsize=(9, 4), dpi=150)

for ax, psf, title in zip(
    axes, [psf0, psf1],
    ["Unaberrated PSF (before)", "Turbulent PSF (after)"]
):
    psf_img = np.log10((psf / psf.max()).shaped + 1e-12)  # key fix
    im = ax.imshow(psf_img,
                   origin='lower',
                   extent=[xmin_m, xmax_m, ymin_m, ymax_m],
                   cmap='inferno', vmin=-6, vmax=0)

    circ = mpatches.Circle((0.0, 0.0), radius=alpha*f_m, fill=False, linewidth=1.5)
    ax.add_patch(circ)
    ax.set_title(title)
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.06)
    cb = fig.colorbar(im, cax=cax)
    cb.set_label(r'$\log_{10}(\mathrm{Intensity}/\max)$')

# run these ONCE after the loop
plt.tight_layout()

base_output_dir = 'simulation_output'
os.makedirs(base_output_dir, exist_ok=True)
output_path = os.path.join(
    base_output_dir,
    f"wavelength_{wavelength*1e6:.2f}um_r0_ref_{r0_ref*1e3:.1f}mm_run_{run_number}.png"
)
plt.savefig(output_path, dpi=300)
plt.close()


def update_csv( wavelength,
                r0_ref_val,
                run_num,
                power_in_bucket_before,
                total_power_before,
                precentage_before,
                power_in_bucket_after,
                total_power_after,
                precentage_after
                ):
    import pandas as pd

    file_path = "./massive_output.csv"
    columns = [
        "wavelength",
        "r0_ref",
        "run_number",
        "power_in_bucket_before_turbulance",
        "total_power_before_turbulance",
        "precentage_before_turbulance",
        "power_in_bucket_after_turbulance",
        "total_power_after_turbulance",
        "precentage_after_turbulance",
        "time"
        ]

    new_row_df = pd.DataFrame([{
        "wavelength": wavelength,
        "r0_ref": r0_ref_val,
        "run_number": run_num,
        
        "power_in_bucket_before_turbulance": power_in_bucket_before,
        "total_power_before_turbulance": total_power_before,
        "precentage_before_turbulance": precentage_before,
        
        "power_in_bucket_after_turbulance": power_in_bucket_after,
        "total_power_after_turbulance": total_power_after,
        "precentage_after_turbulance": precentage_after,
        
        "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }])

    if os.path.exists(file_path):
        new_row_df.to_csv(file_path, mode='a', header=False, index=False)
    else:
        new_row_df.to_csv(file_path, mode='w', header=columns, index=False)
    print("Operation complete. Data has been saved.")

update_csv(wavelength, r0_ref, run_number,power0_in_bucket,power0_total,frac0,power1_in_bucket,power1_total,frac1)