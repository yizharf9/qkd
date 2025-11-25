import os
from params import *
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
    run_number= 0
try:
    wavelength
except NameError:
    # wavelength = 1.234e-6
    wavelength = 1.55e-6
try:
    r0_ref
except NameError:
    r0_ref = 0.1
try:
    focal_dim
except NameError:
    focal_dim=None
try:
    Run_test_batch
except NameError:
    Run_test_batch = False
try:
    Add_Stellar_Noise
except NameError:
    Add_Stellar_Noise_prompt = input("add stellar noise (y/n) ?  ") 
    if Add_Stellar_Noise_prompt == "y" :
        Add_Stellar_Noise = True
    elif Add_Stellar_Noise_prompt == "n" :
        Add_Stellar_Noise = False
    else : 
        exit("not a valid input!")
try:
    USE_AO
except NameError:
    USE_AO_prompt = input("use adaptive optics (y/n) ?  ") 
    if USE_AO_prompt == "y" :
        USE_AO = True
    elif USE_AO_prompt == "n" :
        USE_AO = False
    else : 
        exit("not a valid input!")
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

# ---------- ignore runtime warning (num_additions = 2 * num_complex_multiplications + 2 * num_complex_additions ...) ----------
import warnings
warnings.filterwarnings("ignore", module="hcipy")
# ---------- 1) Pupil (VLT-like) ----------
pupil_grid = make_pupil_grid(N, D * oversz)

aperture_generator = make_obstructed_circular_aperture(D, eps, num_spiders=4, spider_width=spider_w)
aperture = evaluate_supersampled(aperture_generator, pupil_grid, 4) 

# ---------- 2) Wavelength & focal grid ----------
spatial_res = wavelength / D  # [rad] per λ/D
focal_grid = make_focal_grid(q=q, num_airy=num_airy,spatial_resolution=spatial_res, pupil_diameter=D,focal_length=focal_dim)
propagator = FraunhoferPropagator(pupil_grid, focal_grid) # transforms from pupil_grid to focal_grid

# ---------- 3) Wavefront (before turbulence) ----------
initial_wavefront = Wavefront(aperture,wavelength) #wf_pupil
initial_psf = propagator(initial_wavefront).power  # unaberrated PSF (relative units)
initial_phase = propagator(initial_wavefront).phase  # unaberrated PSF (relative unitxs)

# ---------- 4) Single-layer turbulence ----------
r0 = r0_ref * (wavelength / lam_ref) ** (6.0 / 5.0)    
Cn2 = Cn_squared_from_fried_parameter(r0, lam_ref)    # [m^(-2/3)]
v = 0.314 * r0 / tau0                                     # [m/s] 

# HCIPy 0.7.0 API fallback
try:
    layer = InfiniteAtmosphericLayer(pupil_grid, Cn2, L0, v)
except TypeError:
    layer = InfiniteAtmosphericLayer(Cn2, L0, v)

if TurbulencLayer is True:
    wf1 = layer(initial_wavefront)      
else:
    r0_ref=1
    wf1=initial_wavefront           # wavefront AFTER turbulence

psf1 = propagator(wf1).power           # instantaneous PSF with turbulence
phase1 = propagator(wf1).phase           # instantaneous PSF with turbulence
Wf_in_focal=propagator(wf1)

# ---------- 5) Bucket definition: 9 µm circle in focal plane ----------
theta = np.sqrt(focal_grid.x**2 + focal_grid.y**2)  # [rad] 
bucket_mask = (theta <= alpha).astype(float)

# ---------- 6) Proper integration with grid weights (HCIPy 0.7.0) ----------
# w = focal_grid.weights 
w = 1
power0_in_bucket = np.sum(initial_psf * bucket_mask * w)
power0_total     = np.sum(initial_psf * w)
frac0 = power0_in_bucket / power0_total

if Add_Stellar_Noise :
    noisy_psf1 = abs(utils.add_noise_to_wavefront(Wf_in_focal, D,stellar_magnitude=12))**2
    # ---------- 6) Proper integration with grid weights (HCIPy 0.7.0) ----------
    # w = focal_grid.weights 
    w = 1
    
    psf1 = noisy_psf1
    
    power1_in_bucket = np.sum(noisy_psf1 * bucket_mask * w)
    power1_total     = np.sum(noisy_psf1 * w)
    frac1 = power1_in_bucket / power1_total
else :
    power1_in_bucket = np.sum(psf1 * bucket_mask * w)
    power1_total     = np.sum(psf1 * w)
    frac1 = power1_in_bucket / power1_total

# ---------- 7) Print results ----------
#! all listed in the params file so no need to print really...
#region : prints 
print("\n-- current run params --")
print(f"\nwavelength = {wavelength} , r0 = {r0_ref} ")
# print("=== Bucket (9 µm diameter) @ focal plane ===")
# print(f"F/# = {Fnum_sci:.1f},  f_eff = {f_eff:.3f} m")
# print(f"Circle radius (phys): {rad_phys:.3e} m")
# print(f"Circle radius (ang):  {alpha:.3e} rad  (~{alpha*206265:.3f} arcsec)")

print("\n-- BEFORE turbulence (unaberrated) --")
print(f"Power in bucket:  {power0_in_bucket:.6e} (relative)")
print(f"Total power:      {power0_total:.6e} (relative)")
print(f"Fractional power: {frac0:.6%}")
print("\n-- AFTER turbulence (instantaneous) --")
print(f"Power in bucket:  {power1_in_bucket:.6e} (relative)")
print(f"Total power:      {power1_total:.6e} (relative)")
print(f"Fractional power: {frac1:.6%}")
#check for energy conservation
Energy_conservation = utils.check_energy_conservation(wf1,Wf_in_focal)
#endregion

# ---------- 8) Implement Adaptive Optics ----------
if USE_AO:
    wf_wfs_after_dm_prop = utils.use_adaptive_optics(
        initial_wavefront,
        psf1,
        pupil_grid,
        focal_grid,
        layer,
        propagator,
        D,
        aperture,
    )
else:
    wf_wfs_after_dm_prop=propagator(wf1)

# ---------- 9) Optional: overlay circle on both PSFs ----------
if save_images:
    phase_screen = layer.phase_for(wavelength)          # Field on the pupil grid (radians)
    power_screen = np.abs(np.exp(1j*phase_screen))**2          #! to be changed to psf

    # pupil-plane extent (meters -> mm for readability)
    scale_mm = 1e3
    xmin_p, xmax_p = pupil_grid.x.min()*scale_mm, pupil_grid.x.max()*scale_mm
    ymin_p, ymax_p = pupil_grid.y.min()*scale_mm, pupil_grid.y.max()*scale_mm
    extent_pupil_mm = [xmin_p, xmax_p, ymin_p, ymax_p]

    
    # focal-plane extent (מחשבים מהמוקד)
    f_m = f_eff
    xmin_m = initial_psf.grid.x.min()*f_m
    xmax_m = initial_psf.grid.x.max()*f_m
    ymin_m = initial_psf.grid.y.min()*f_m
    ymax_m = initial_psf.grid.y.max()*f_m
    extent_focal_mm = [xmin_m*scale_mm, xmax_m*scale_mm, ymin_m*scale_mm, ymax_m*scale_mm]
    # 1) Figure & axes
    fig, axes = plt.subplots(2, 4, figsize=(20, 4.5), dpi=150)

    # --- (a) Phase screen (pupil plane) ---
    utils.plot_phase_screen(fig,axes[0,0],phase_screen,extent_pupil_mm,title="A) Atmosphere Phase")
    utils.plot_psf_on(fig,axes[1,0],power_screen,alpha,f_m,extent_focal_mm,scale_mm,title="A) Atmosphere PSF")
    # --- (b) Unaberrated PSF ---
    utils.plot_phase_screen(fig,axes[0,1],phase_screen,extent_pupil_mm,mask = aperture,title="B) Unaberrated Phase (Before)")
    utils.plot_psf_on(fig,axes[1,1], aperture,alpha,f_m,extent_focal_mm,scale_mm, title="B) Unaberrated PSF (before)")
    # --- (c) Turbulent PSF ---
    utils.plot_phase_screen(fig,axes[0,2],phase1,extent_pupil_mm , title="D) Turbulent Phase (after)")
    utils.plot_psf_on(fig,axes[1,2], psf1,alpha,f_m,extent_focal_mm,scale_mm,title="D) Turbulent PSF (after)")
    # --- (d) AO-corrected PSF ---
    utils.plot_phase_screen(fig,axes[0,3],wf_wfs_after_dm_prop.phase,extent_pupil_mm , title="AO correction (phase)")
    utils.plot_psf_on(fig,axes[1,3], wf_wfs_after_dm_prop.power,alpha,f_m,extent_focal_mm,scale_mm,title="AO correction (psf)") 
    plt.tight_layout()

    # save or show 
    base_output_dir = 'simulation_output'
    os.makedirs(base_output_dir, exist_ok=True)
    out_path = os.path.join(
        base_output_dir,
        f"phase_psf_wl_{wavelength*1e6:.2f}um_r0ref_{r0_ref*1e3:.1f}mm_run_{run_number}.png"
    )
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"✅ Saved combined figure to: {out_path}")
    plt.close(fig)

if not Run_test_batch :
    utils.update_csv(wavelength, r0_ref, run_number,focal_dim,power0_in_bucket,power0_total,frac0,power1_in_bucket,power1_total,frac1,conservation_of_energy=Energy_conservation)