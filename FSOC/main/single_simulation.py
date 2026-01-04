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
from tqdm.auto import tqdm
import time
import datetime

# ============================================================================
# Helper: Poisson noise (fallback if not defined elsewhere)
# ============================================================================
def large_poisson(x):
    """Apply Poisson noise to array, with safe maximum."""
    # Cap input to avoid overflow in Poisson sampler (max ~1e7)
    x_safe = np.minimum(np.maximum(x, 0), 1e6)
    return np.random.poisson(x_safe).astype(float)

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
    save_images_prompt = input("Use AO (y/n)? ")
    if save_images_prompt.lower() == "y":
        USE_AO = True
    elif save_images_prompt.lower() == "n":
        USE_AO = False
    else:
        exit("Not a valid input! Please enter 'y' or 'n'.")
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
    s_time = time.time()
    
    # ========================================================================
    # BUILD AOConfig from existing params
    # ========================================================================
    from main.utils import AOConfig, AOState, set_AO, calibration, save_calibration_data, save_correction_comparison
    
    cfg = AOConfig(
        pupil_grid=pupil_grid,
        focal_q=q,
        spatial_resolution=spatial_res,
        telescope_diameter=D,
        aperture=ap,
        wavelength=wavelength,
        tau0=tau0,
        L0=L0,
        f_number=50.0,
        num_lenslets=30,
        sh_diameter=5e-3,
        num_modes=64,
        probe_amp_factor=0.1,
        rcond=1e-3,
        delta_t=1e-3,
        num_iterations=50,  # Reduced back to 50 - system diverges after ~50 with gain=0.02
        burn_in_iterations=5,
        gain=0.02,  # Conservative gain - negative sign correct, need slow convergence
        leakage=0.02,  # Increased damping to prevent oscillation
        zero_magnitude_flux=3.9e10,
        stellar_magnitude=5.0,
        coro_inner_radius_lamD=4.0,
        base_output_dir="simulation_output",
        csv_output_path="Data/AO_simulation_log.csv",
    )
    
    # ========================================================================
    # STAGE 1: Set up AO hardware
    # ========================================================================
    print("\n🔧 Stage 1: Setting up AO config...")
    state = set_AO(num_airy=num_airy, cfg=cfg)
    print("✅ AO hardware initialized")
    
    # ========================================================================
    # STAGE 2: Calibration (response matrix)
    # ========================================================================
    print("\n📊 Stage 2: Calibrating (response matrix)...")
    state = calibration(state=state, cfg=cfg, show_progress=True)
    print("✅ Calibration complete")
    
    # ========================================================================
    # Create output directory: output_files/dd.mm.yyyy/AO_hh:mm:ss/
    # ========================================================================
    now = datetime.datetime.now()
    date_str = now.strftime("%d.%m.%Y")
    time_str = now.strftime("%H:%M:%S")
    output_dir = os.path.join("output_files", date_str, f"AO_{time_str}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 Output directory: {output_dir}\n")
    
    # ========================================================================
    # SAVE CALIBRATION DATA (matrices + SVD plots)
    # ========================================================================
    print("💾 Saving calibration data...")
    calib_files = save_calibration_data(state=state, cfg=cfg, output_dir=output_dir)
    print(f"✅ Calibration data saved to: {output_dir}/calibration/\n")
    
    # ========================================================================
    # VISUALIZE SHACK-HARTMANN REFERENCE IMAGE
    # ========================================================================
    print("📊 Visualizing Shack-Hartmann reference spots...")
    from main.utils import visualize_shack_hartmann_reference
    sh_vis_path = visualize_shack_hartmann_reference(
        image_ref=state.image_ref if hasattr(state, 'image_ref') else None,
        num_lenslets=cfg.num_lenslets,
        output_dir=os.path.join(output_dir, "calibration"),
        wavelength_val=wavelength
    )
    if sh_vis_path:
        print(f"✅ Shack-Hartmann visualization saved\n")
    
    # ========================================================================
    # STAGE 3: LOCAL correction loop with PAM modulation & logging
    # ========================================================================
    print("🔄 Stage 3: Running AO correction loop...\n")
    
    # PAM modulation levels (power fraction per level)
    pam_levels = [0.5, 0.75, 1.0]
    
    # Determine r0 list
    if Multy_use_AO:
        r0_list = r0_ref_list
    else:
        r0_list = [r0_ref]
    
    # Loop over r0 values
    for r0 in r0_list:
        fried_parameter = r0
        velocity = 0.314 * fried_parameter / tau0
        Cn2 = Cn_squared_from_fried_parameter(fried_parameter, 500e-9)
        
        try:
            layer_ao = InfiniteAtmosphericLayer(pupil_grid, Cn2, L0, velocity)
        except TypeError:
            layer_ao = InfiniteAtmosphericLayer(Cn2, L0, velocity)
        
        layer_ao.reset()
        state.deformable_mirror.flatten()
        
        # Data logs for this r0
        main_data_list = []
        debug_data_list = []
        
        # Capture initial turbulent wavefront (before correction)
        wf_before_correction = layer_ao(state.wf_wfs)
        wf_focal_before = state.propagator(wf_before_correction)
        
        print(f"  r0 = {fried_parameter:.4g} m ({fried_parameter*1e3:.3f} mm)")
        print("  Running AO loop...")
        print(f"Number Of itterations: {cfg.num_iterations}")
        for timestep in tqdm(range(cfg.num_iterations)):
            layer_ao.t = timestep * cfg.delta_t
            
            # ================================================================
            # PAM MODULATION: vary wavefront power per timestep
            # ================================================================
            # Temporarily disable PAM for debugging
            # pam_idx = timestep % len(pam_levels)
            # pam_factor = pam_levels[pam_idx]
            pam_factor = 1.0  # Keep constant power for now
            modulated_power = state.wf_wfs.total_power * pam_factor  # apply PAM
            state.wf_wfs.total_power = modulated_power
            
            # ================================================================
            # Propagate through atmosphere and DM
            # ================================================================
            wf_wfs_after_atmos = layer_ao(state.wf_wfs)
            wf_wfs_after_dm = state.deformable_mirror(wf_wfs_after_atmos)
            
            # WFS camera readout
            wf_wfs_on_sh = state.shwfs(state.magnifier(wf_wfs_after_dm))
            state.camera.integrate(wf_wfs_on_sh, cfg.delta_t)
            wfs_image_raw = state.camera.read_out()
            
            # Apply Poisson noise (convert to array, apply noise, then reconstruct as Field)
            wfs_image_array = large_poisson(np.asarray(wfs_image_raw)).astype("float")
            
            # Reconstruct as HCIPy Field with the SAME grid as the camera/shwfs
            # This is critical: the WFS image must use the SAME grid as used by the estimator
            wfs_image = Field(wfs_image_array, wfs_image_raw.grid)
            
            # Science focal plane
            wf_focal_plane = state.propagator(
                state.deformable_mirror(layer_ao(state.wf_wfs))
            )
            
            # ================================================================
            # MAIN DATA: Power in focal (9µm fiber PIB)
            # ================================================================
            # Compute fiber coupling (9µm radius)
            focal_power_map = wf_focal_plane.power
            x_focal = state.focal_grid.x
            y_focal = state.focal_grid.y
            r_focal = np.sqrt(x_focal**2 + y_focal**2)
            fiber_mask = r_focal <= 9e-6  # 9 µm radius
            power_in_focal = float(np.sum(focal_power_map[fiber_mask]))
            
            # ================================================================
            # DEBUG DATA: COG of Shack-Hartmann, DM stats
            # ================================================================
            # COG of WFS image (extract numpy array from Field)
            wfs_flat = np.asarray(wfs_image).ravel()
            
            # Get grid shape from focal_grid
            ny, nx = state.focal_grid.shape
            
            # Reshape WFS image to 2D using grid dimensions
            if wfs_flat.size == ny * nx:
                wfs_image_2d = wfs_flat.reshape((ny, nx))
            else:
                # Fallback: use actual size
                grid_size = int(np.sqrt(wfs_flat.size))
                wfs_image_2d = wfs_flat.reshape((grid_size, grid_size))
            
            xx, yy = np.meshgrid(np.arange(wfs_image_2d.shape[1]), np.arange(wfs_image_2d.shape[0]))
            xx_flat = xx.ravel()
            yy_flat = yy.ravel()
            wfs_flat_2d = wfs_image_2d.ravel()
            total_flux = np.sum(wfs_flat_2d)
            if total_flux > 0:
                cog_x = float(np.sum(xx_flat * wfs_flat_2d) / total_flux)
                cog_y = float(np.sum(yy_flat * wfs_flat_2d) / total_flux)
            else:
                cog_x, cog_y = 0.0, 0.0
            
            # DM statistics
            dm_actuators = np.asarray(state.deformable_mirror.actuators)
            dm_mean = float(np.mean(dm_actuators))
            dm_max = float(np.max(dm_actuators))
            dm_min = float(np.min(dm_actuators))
            
            # ================================================================
            # LOG DATA
            # ================================================================
            main_data_list.append({
                "timestep": timestep,
                "power_in_focal": power_in_focal,
                "power_send": modulated_power,
            })
            
            # ================================================================
            # WFS CONTROL & DM UPDATE - WITH DIAGNOSTICS
            # ================================================================
            # Estimate slopes from WFS image
            slopes = state.shwfse.estimate([wfs_image])
            
            # Subtract reference slopes (differential measurement)
            slopes_residual = slopes - state.slopes_ref
            slopes_residual_ravel = slopes_residual.ravel()
            
            # Get current DM state BEFORE update
            dm_actuators_before = np.asarray(state.deformable_mirror.actuators).copy()
            dm_rms_before = np.sqrt(np.mean(dm_actuators_before**2))
            
            # Compute modal commands via reconstruction matrix
            modal_commands = state.reconstruction_matrix.dot(slopes_residual_ravel)
            
            # **CORRECTED SIGN**: Negative sign is correct for this system
            state.deformable_mirror.actuators = (
                (1 - cfg.leakage) * state.deformable_mirror.actuators
                - cfg.gain * modal_commands  # NEGATIVE SIGN - corrected
            )
            
            # Clip DM actuators to prevent runaway (5 micrometers max)
            dm_max_allowed = 5e-6
            state.deformable_mirror.actuators = np.clip(
                state.deformable_mirror.actuators,
                -dm_max_allowed,
                dm_max_allowed
            )
            
            # Get DM state AFTER update for diagnostics
            dm_actuators_after = np.asarray(state.deformable_mirror.actuators)
            dm_rms_after = np.sqrt(np.mean(dm_actuators_after**2))
            
            # Compute diagnostics
            residual_rms = np.linalg.norm(slopes_residual_ravel) / len(slopes_residual_ravel)
            modal_cmd_rms = np.linalg.norm(modal_commands) / len(modal_commands)
            
            # Log diagnostic data
            debug_data_list.append({
                "timestep": timestep,
                "cog_x": cog_x,
                "cog_y": cog_y,
                "dm_mean": float(np.mean(dm_actuators_after)),
                "dm_max": float(np.max(dm_actuators_after)),
                "dm_min": float(np.min(dm_actuators_after)),
                "dm_rms_before": float(dm_rms_before),
                "dm_rms_after": float(dm_rms_after),
                "slopes_residual_rms": float(residual_rms),
                "modal_cmd_rms": float(modal_cmd_rms),
            })
        
        # ====================================================================
        # SAVE CSVs for this r0
        # ====================================================================
        df_main = pd.DataFrame(main_data_list)
        df_debug = pd.DataFrame(debug_data_list)
        
        main_csv = os.path.join(output_dir, f"AO_main_r0_{fried_parameter*1e3:.3f}mm.csv")
        debug_csv = os.path.join(output_dir, f"AO_debug_r0_{fried_parameter*1e3:.3f}mm.csv")
        
        df_main.to_csv(main_csv, index=False)
        df_debug.to_csv(debug_csv, index=False)
        
        print(f"  ✅ Saved: {main_csv}")
        print(f"  ✅ Saved: {debug_csv}")
        
        # ====================================================================
        # SAVE CORRECTION COMPARISON (before/after focal plane images)
        # ====================================================================
        print(f"\n  💾 Saving correction comparison images...")
        wf_focal_after = state.propagator(state.deformable_mirror(layer_ao(state.wf_wfs)))
        
        # Calculate PIB (power in bucket) for before and after
        power_before_map = np.asarray(wf_focal_before.power)
        power_after_map = np.asarray(wf_focal_after.power)
        
        x_focal = state.focal_grid.x
        y_focal = state.focal_grid.y
        r_focal = np.sqrt(x_focal**2 + y_focal**2)
        fiber_mask = r_focal <= 9e-6  # 9 µm radius (9mm fiber)
        
        pib_before_val = float(np.sum(power_before_map[fiber_mask]))
        pib_after_val = float(np.sum(power_after_map[fiber_mask]))
        
        # Fiber parameters (defaults: 9mm core, 10km length)
        fiber_diameter_mm = 9.0
        fiber_length_km = 10.0
        
        correction_files = save_correction_comparison(
            wf_before=wf_focal_before,
            wf_after=wf_focal_after,
            output_dir=output_dir,
            r0=fried_parameter,
            fiber_diameter_mm=fiber_diameter_mm,
            fiber_length_km=fiber_length_km,
            pib_before=pib_before_val,
            pib_after=pib_after_val,
        )
        print(f"  ✅ Correction images saved to: {output_dir}/correction/\n")
    
    # Get final wavefront for downstream processing
    wf_focal_plane = state.propagator(state.deformable_mirror(layer_ao(state.wf_wfs)))
    deformable_mirror = state.deformable_mirror
    wf_wfs_after_atmos = layer_ao(state.wf_wfs)
    wfs_image = wfs_image  # last iteration
    
    # Compute AO power (total power in focal plane, normalized)
    norm = state.norm_power if state.norm_power > 0 else 1.0
    AO_power = float(np.sum(wf_focal_plane.power)) / norm
    
    print(f"\n✨ AO complete!")
    print(f"AO correction took {datetime.timedelta(seconds=time.time()-s_time)} (hh:mm:ss)")

else:
    wf_focal_plane=prop(wf1)
    AO_power=0.0
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
    base_output_dir = "simulagition_output"
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
    utils.update_csv(wavelength=wavelength,
                      r0_ref_val=r0_ref, 
                      run_num=run_number,
                      power_in_bucket_before=power0_in_bucket,
                      total_power_before=power0_total,
                      precentage_before=frac0,
                      total_power_after=power1_in_bucket,
                      precentage_after=frac1,
                      conservation_of_energy=Energy_conservation,
                      num_airy=num_airy,
                      power_after_AO=AO_power,
                      power_in_bucket_after=power1_total)