import os
import utils
import numpy as np
import matplotlib.pyplot as plt
try:
    from hcipy import *
except Exception as e:
    raise ImportError("HCIPy is required. Install it with: pip install hcipy") from e

# ----------------------------- Directory Check -----------------------------
utils.check_dir()
# ----------------------------- Parameters -----------------------------
try:
    TurbulencLayer
except NameError:
    TurbulencLayer = True
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

if TurbulencLayer is True:
    wf1 = layer(wf0)      
else:
    wf1=wf0           # wavefront AFTER turbulence
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
#region : prints
print(f"wavelength = {wavelength} , r0 = {r0_ref} , ")
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
#endregion
# ---------- 8) Optional: overlay circle on both PSFs ----------
if save_images:
    phase_screen = layer.phase_for(wavelength)          # Field on the pupil grid (radians)

    # pupil-plane extent (meters -> mm for readability)
    scale_mm = 1e3
    xmin_p, xmax_p = pupil_grid.x.min()*scale_mm, pupil_grid.x.max()*scale_mm
    ymin_p, ymax_p = pupil_grid.y.min()*scale_mm, pupil_grid.y.max()*scale_mm
    extent_pupil_mm = [xmin_p, xmax_p, ymin_p, ymax_p]

    # focal-plane extent (מחשבים מהמוקד)
    f_m = f_eff
    xmin_m = psf0.grid.x.min()*f_m
    xmax_m = psf0.grid.x.max()*f_m
    ymin_m = psf0.grid.y.min()*f_m
    ymax_m = psf0.grid.y.max()*f_m
    extent_focal_mm = [xmin_m*scale_mm, xmax_m*scale_mm, ymin_m*scale_mm, ymax_m*scale_mm]
    # 1) Figure & axes
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), dpi=150)

    # --- (a) Phase screen (pupil plane) ---
    utils.plot_phase_screen(fig,axes[0],phase_screen,extent_pupil_mm)

    # --- (b) Unaberrated PSF ---
    utils.plot_psf_on(fig,axes[1], psf0,alpha,f_m,extent_focal_mm,scale_mm, "Unaberrated PSF (before)")

    # --- (c) Turbulent PSF ---
    utils.plot_psf_on(fig,axes[2], psf1,alpha,f_m,extent_focal_mm,scale_mm, "Turbulent PSF (after)")

    plt.tight_layout()

    # save or show (תמיד נשמור לאותה תיקייה כמו קודם)
    base_output_dir = 'simulation_output'
    os.makedirs(base_output_dir, exist_ok=True)
    out_path = os.path.join(
        base_output_dir,
        f"phase_psf_wl_{wavelength*1e6:.2f}um_r0ref_{r0_ref*1e3:.1f}mm_run_{run_number}.png"
    )
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"✅ Saved combined figure to: {out_path}")
    plt.close(fig)

utils.update_csv(wavelength, r0_ref, run_number,power0_in_bucket,power0_total,frac0,power1_in_bucket,power1_total,frac1)