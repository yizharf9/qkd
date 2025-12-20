import os
from params import *
import utils
import numpy as np
import matplotlib.pyplot as plt
try:
    from hcipy import *
except Exception as e:
    raise ImportError("HCIPy is required. Install it with: pip install hcipy") from e
import params as params
import pandas as pd
from scipy.ndimage import sum as scipy_sum
from tqdm.notebook import tqdm
import time
import datetime
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
    run_number= 999
try:
    wavelength
except NameError:
    # wavelength = 1.234e-6
    wavelength = 1.55e-6

try:
    r0_ref
except NameError:        
    r0_ref = 0.017

try:
    r0_ref_list
except NameError:
    r0_ref_list=[r0_ref]

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
    USE_AO = True
try:
    num_airy
except NameError:
    num_airy=5

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
if USE_AO:
    try: 
        Multy_use_AO
    except NameError:
        Multy_use_AO_prompt = input("USE multy mode AO (y/n)? ")
        if Multy_use_AO_prompt.lower() == "y":
            Multy_use_AO = True
        elif Multy_use_AO_prompt.lower() == "n":
            Multy_use_AO = False
        else:
            exit("Not a valid input! Please enter 'y' or 'n'.")


# ---------- ignore runtime warning (num_additions = 2 * num_complex_multiplications + 2 * num_complex_additions ...) ----------
import warnings
warnings.filterwarnings("ignore", module="hcipy")
# ---------- 1) Pupil (VLT-like) ----------



# ---------- 1) Pupil (VLT-like) ---------- 
D = 0.3                # [m]
D_obs =0.1             # [m]
eps = D_obs / D # central obscuration ratio
spider_w = 0.005         # [m]
oversz = 16.0/15.0
N = int(240 * oversz) #256 # number of pixels across pupil diameter
N=240
pupil_grid = make_pupil_grid(N*oversz, D * oversz)
ap_gen = make_obstructed_circular_aperture(D, eps, num_spiders=4, spider_width=spider_w)
ap = evaluate_supersampled(ap_gen, pupil_grid, 4) 
aperture_generator = ap_gen
aperture = ap
# ---------- 2) Wavelength & focal grid ----------
# wavelength = 2.2e-6  # [m] science wavelength
q = 4
#num_airy = 60
spatial_res = wavelength / D  # [rad] per λ/D
focal_grid = make_focal_grid(q=q, num_airy=num_airy,spatial_resolution=spatial_res)

prop = FraunhoferPropagator(pupil_grid, focal_grid) # transforms from pupil_grid to focal_grid

# ---------- 3) Wavefront (before turbulence) ----------
wf0=Wavefront(ap,wavelength) #wf_pupil
#wf0.electric_field = Wavefront(ap, wavelength).electric_field+Wavefront(ap,ref_wavelength).electric_field
# wf0.total_power = 1.0
wf0.total_power=1.0
psf0 = prop.forward(wf0).power  # unaberrated PSF (relative units)
phase0 = prop.forward(wf0).phase  # unaberrated PSF (relative unitxs)

# ---------- 4) Single-layer turbulence ----------
                                    # [m/s] 

# HCIPy 0.7.0 API fallback
if TurbulencLayer is True:
    L0 = 40.0          # [m]
    tau0 = 5e-3        # [s]
    lam_ref = 500e-9   # [m]
#r0 = r0_ref * (wavelength / lam_ref) ** (6.0 / 5.0)    
    r0 = r0_ref * (wavelength / lam_ref) ** (6.0 / 5.0)    
    Cn2 = Cn_squared_from_fried_parameter(r0_ref, lam_ref)    # [m^(-2/3)]
    v = 0.314 * r0 / tau0 
    try:
        layer = InfiniteAtmosphericLayer(pupil_grid, Cn2, L0, v)
    except TypeError:
        layer = InfiniteAtmosphericLayer(Cn2, L0, v)
    wf1=layer(wf0)
else:
    r0_ref=1
    try:
        layer = InfiniteAtmosphericLayer(pupil_grid,1e-23, 1, 0)
    except TypeError:
        layer = InfiniteAtmosphericLayer(1e-23, 1, 0)
        r0_ref=1
    wf1=layer(wf0)
    wf1=wf0           # wavefront AFTER turbulence
psf1 = prop(wf1).power           # instantaneous PSF with turbulence
phase1 = prop(wf1).phase           # instantaneous PSF with turbulence
Wf_in_focal=prop(wf1)

# ---------- 5) Bucket definition: 9 µm circle in focal plane ----------
theta = np.sqrt(focal_grid.x**2 + focal_grid.y**2)  # [rad] 
bucket_mask = (theta <= alpha).astype(float)

# ---------- 6) Proper integration with grid weights (HCIPy 0.7.0) ----------
# w = focal_grid.weights 
w = 1

power0_in_bucket = np.sum(psf0)
power0_total     = np.sum(psf0 * w)
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
#------------
if USE_AO:
    s_time=time.time()
    if Multy_use_AO:
        [wf_wfs_after_atmos,deformable_mirror,wf_focal_plane,wfs_image] = utils.AO(num_airy=num_airy,save_images=save_images,run_number=run_number,r0_ref_list=r0_ref_list)        # Show created animation
    if not(Multy_use_AO):
        [wf_wfs_after_atmos,deformable_mirror,wf_focal_plane,wfs_image] = utils.AO(num_airy=num_airy,save_images=save_images,run_number=run_number,single_r0=r0_ref)        # Show created animation
    AO_power=float(np.sum(wf_focal_plane.power))/norm
    print(f"AO correction took {datetime.timedelta(seconds=time.time()-s_time)} (hh:mm:ss)")
else:
    wf_focal_plane=prop(wf1)
    AO_power=0000000.00000
# ---------- 8) Optional: overlay circle on both PSFs ----------
if save_images:
    initial_psf=wf0.power
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
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 4.5), dpi=150)

    # --- (a) Phase screen (pupil plane) ---
    utils.plot_phase_screen(fig,axes[0,0],phase_screen,extent_pupil_mm,title="A) Atmosphere Phase")
    utils.plot_psf_on(fig,axes[1,0],power_screen,alpha,f_m,extent_focal_mm,scale_mm,title="A) Atmosphere PSF")
    # --- (b) Unaberrated PSF ---
    utils.plot_phase_screen(fig,axes[0,1],phase_screen,extent_pupil_mm,mask = aperture,title="B) Unaberrated Phase (Before)")
    utils.plot_psf_on(fig,axes[1,1], aperture,alpha,f_m,extent_focal_mm,scale_mm, title="B) Unaberrated PSF (before)")
    # --- (c) Turbulent PSF ---
    utils.plot_phase_screen(fig,axes[0,2],phase1,extent_pupil_mm , title="D) Turbulent Phase (after)")
    utils.plot_psf_on(fig,axes[1,2], psf1,alpha,f_m,extent_focal_mm,scale_mm,title="D) Turbulent PSF (after)",log_scale=True)
    # --- (d) AO-corrected PSF ---
    utils.plot_phase_screen(fig,axes[0,3],wf_focal_plane.phase,extent_pupil_mm , title="AO correction (phase)")
    utils.plot_psf_on(fig,axes[1,3],wf_focal_plane.power,alpha,f_m,extent_focal_mm,scale_mm,title="AO correction (psf)",log_scale=True) 
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
    """( wavelength,r0_ref_val,run_num,power_in_bucket_before,total_power_before,precentage_before,total_power_after,precentage_after,conservation_of_energy,num_airy,power_after_AO=None,power_in_bucket_after=1111111):"""
    utils.update_csv(wavelength=wavelength, r0_ref_val=r0_ref, run_num=run_number,power_in_bucket_before=power0_in_bucket,total_power_before=power0_total,precentage_before=frac0,total_power_after=power1_in_bucket,precentage_after=frac1,conservation_of_energy=Energy_conservation,num_airy=num_airy,power_after_AO=AO_power,power_in_bucket_after=power1_total)