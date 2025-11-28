import os
from params import *
import utils
import numpy as np
import matplotlib.pyplot as plt
try:
    from hcipy import *
except Exception as e:
    raise ImportError("HCIPy is required. Install it with: pip install hcipy") from e
import params
import pandas as pd
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
    r0_ref_list
except NameError:
    try:
        r0_ref
    except NameError:        
        r0_ref = 0.017
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
    USE_OA = True
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

# ---------- ignore runtime warning (num_additions = 2 * num_complex_multiplications + 2 * num_complex_additions ...) ----------
import warnings
warnings.filterwarnings("ignore", module="hcipy")
# ---------- 1) Pupil (VLT-like) ----------
pupil_grid = make_pupil_grid(N, D * oversz)

aperture_generator = make_obstructed_circular_aperture(D, eps, num_spiders=4, spider_width=spider_w)
aperture = evaluate_supersampled(aperture_generator, pupil_grid, 4) 

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
if USE_OA:
    f_number = 50
    num_lenslets = 40 # 40 lenslets along one diameter
    sh_diameter = 5e-3 # m
    telescope_diameter=D
    magnification = sh_diameter / telescope_diameter
    magnifier = Magnifier(magnification)
    shwfs = SquareShackHartmannWavefrontSensorOptics(pupil_grid.scaled(magnification), f_number, \
                                                 num_lenslets, sh_diameter)
    shwfse = ShackHartmannWavefrontSensorEstimator(shwfs.mla_grid, shwfs.micro_lens_array.mla_index)
    camera = NoiselessDetector(focal_grid)
    camera.integrate(shwfs(magnifier(wf0)), 1)
    image_ref = camera.read_out()
    fluxes = ndimage.measurements.sum(image_ref, shwfse.mla_index, shwfse.estimation_subapertures)
    flux_limit = fluxes.max() * 0.5

    estimation_subapertures = shwfs.mla_grid.zeros(dtype='bool')
    estimation_subapertures[shwfse.estimation_subapertures[fluxes > flux_limit]] = True

    shwfse = ShackHartmannWavefrontSensorEstimator(shwfs.mla_grid, shwfs.micro_lens_array.mla_index, estimation_subapertures)
    slopes_ref = shwfse.estimate([image_ref])
    num_modes = 500
    dm_modes = make_disk_harmonic_basis(pupil_grid, num_modes, telescope_diameter, 'neumann')
    dm_modes = ModeBasis([mode / np.ptp(mode) for mode in dm_modes], pupil_grid)
    deformable_mirror = DeformableMirror(dm_modes)
    probe_amp = 0.01 * wavelength
    response_matrix = []
    wf0.total_power = 1

    # Set up animation
    #plt.figure(figsize=(10, 6))
    #anim = FFMpegWriter('response_matrix.mp4', framerate=5)

    for i in tqdm(range(num_modes)):
        slope = 0

        # Probe the phase response
        amps = [-probe_amp, probe_amp]
        for amp in amps:
            deformable_mirror.flatten()
            deformable_mirror.actuators[i] = amp

            dm_wf = deformable_mirror.forward(wf0)
            wfs_wf = shwfs(magnifier(dm_wf))

            camera.integrate(wfs_wf, 1)
            image = camera.read_out()

            slopes = shwfse.estimate([image])

            slope += amp * slopes / np.var(amps)

        response_matrix.append(slope.ravel())
    response_matrix = ModeBasis(response_matrix)
    rcond = 1e-3
    reconstruction_matrix = inverse_tikhonov(response_matrix.transformation_matrix, rcond=rcond)
    zero_magnitude_flux = 3.9e10 #3.9e10 photon/s for a mag 0 star
    stellar_magnitude = 5
    delta_t = 1e-3 # sec, so a loop speed of 1kHz.

    wf_wfs = Wavefront(ap,wavelength)
    wf_wfs.total_power = zero_magnitude_flux *  10**(-stellar_magnitude / 2.5)
    propagator=FraunhoferPropagator(pupil_grid, focal_grid) # transforms from pupil_grid to focal_grid
    norm=int(wf_wfs.total_power)
    log_rows=[]
    for r0 in r0_ref_list:
        fried_parameter =r0
        Cn_squared = Cn_squared_from_fried_parameter(fried_parameter, 500e-9)
        layer = InfiniteAtmosphericLayer(pupil_grid, Cn_squared, L0, v)
        layer.reset()
        deformable_mirror.flatten()
        gain = 0.3
        leakage = 0.01
        num_iterations = 800
        burn_in_iterations = 5

        coro = PerfectCoronagraph(ap, 4)

        long_exposure = focal_grid.zeros()
        long_exposure_coro = focal_grid.zeros()

        # Set up animation
        plt.figure(figsize=(8, 8))
        anim = FFMpegWriter('AO_simulation_with_turbulence.mp4', framerate=10)

        for timestep in tqdm(range(num_iterations)):
            layer.t = timestep * delta_t

            # Propagate through atmosphere and deformable mirror.
            wf_wfs_after_atmos = layer(wf_wfs)
            wf_wfs_after_dm = deformable_mirror(wf_wfs_after_atmos)

            # Propagate through SH-WFS
            wf_wfs_on_sh = shwfs(magnifier(wf_wfs_after_dm))
            wf_wfs_on_sh_non_magnifier=shwfs(wf_wfs_after_dm)
            # Propagate the NIR wavefront
            wf_sci_focal_plane = propagator(deformable_mirror(layer(wf_wfs)))
            wf_sci_coro = propagator(coro(deformable_mirror(layer(wf_wfs))))

            # Read out WFS camera
            camera.integrate(wf_wfs_on_sh, delta_t)
            wfs_image = camera.read_out()
            wfs_image = large_poisson(wfs_image).astype('float')

            # Accumulate long-exposure image
            if timestep >= burn_in_iterations:
                long_exposure += wf_sci_focal_plane.power / (num_iterations - burn_in_iterations)
                long_exposure_coro += wf_sci_coro.power / (num_iterations - burn_in_iterations)

            # Calculate slopes from WFS image
            slopes = shwfse.estimate([wfs_image + 1e-10])
            slopes -= slopes_ref
            slopes = slopes.ravel()

            # Perform wavefront control and set DM actuators
            deformable_mirror.actuators = (1 - leakage) * deformable_mirror.actuators - gain * reconstruction_matrix.dot(slopes)

            # Plotting
            if timestep % 20 == 0:
                E_power = float(np.sum(propagator(wf_wfs_after_atmos).power))
                D_power = float(np.sum(wf_sci_focal_plane.power))

                log_rows.append({
                    "timestep": timestep,
                    "E_power_sum": E_power,
                    "D_power_sum": D_power,
                    "num_airy":num_airy,
                    "r0_ref":fried_parameter,
                })
                plt.clf()
                plt.suptitle('Timestep %d / %d' % (timestep, num_iterations))

                plt.subplot(3,2,1)
                plt.title("psf at the entry [c]")
                imshow_field(wf_wfs_after_atmos.phase, cmap='inferno')
                plt.xlabel('[M]')
                plt.ylabel('[M]')
                cb1=plt.colorbar()
                cb1.set_label("[rad]")

                plt.subplot(3,2,2)
                plt.title('WFS at camera [counts][E]')
                imshow_field(np.log10(propagator(wf_wfs_after_atmos).power/propagator(wf_wfs_after_atmos).power.max()),cmap="inferno")
                plt.xlabel('[M]')
                plt.ylabel('[M]')
                cb2=plt.colorbar()
                cb2.set_label("[W]")

                plt.subplot(3,2,3)
                plt.title('DM surface [$\\mu$m]-[H]')
                imshow_field(deformable_mirror.surface * 1e6, cmap='RdBu', vmin=-2, vmax=2, mask=ap)
                plt.xlabel('[M]')
                plt.ylabel('[M]')
                cb3=plt.colorbar()
                cb3.set_label("[rad]")

                plt.subplot(3,2,4)
                plt.title(' PSF at Shack-Hartmann [F]')
                imshow_field(wfs_image,cmap='inferno')
                plt.xlabel('[M]')
                plt.ylabel('[M]')
                cb4=plt.colorbar()
                cb4.set_label("[W]")

                plt.subplot(3,2,5)
                plt.title(' PSF at focal [D]')
                imshow_field(np.log10(wf_sci_focal_plane.power / wf_sci_focal_plane.power.max()), vmin=-6, vmax=0, cmap='inferno')
                plt.xlabel('[M]')
                plt.ylabel('[M]')
                cb5=plt.colorbar()
                cb5.set_label("[W]")


                plt.tight_layout()
                anim.add_frame()

        plt.close()
        anim.close()

        # Show created animation
    if save_images==True

        plt.clf
        plt.suptitle('Timestep %d / %d' % (timestep, num_iterations))

        plt.subplot(3,2,1)
        plt.title("psf at the entry [c]")
        imshow_field(wf_wfs_after_atmos.phase, cmap='inferno')
        plt.xlabel('[M]')
        plt.ylabel('[M]')
        cb1=plt.colorbar()
        cb1.set_label("[rad]")

        plt.subplot(3,2,2)
        plt.title('WFS at camera [counts][E]')
        imshow_field(np.log10(propagator(wf_wfs_after_atmos).power/propagator(wf_wfs_after_atmos).power.max()),cmap="inferno")
        plt.xlabel('[M]')
        plt.ylabel('[M]')
        cb2=plt.colorbar()
        cb2.set_label("[W]")

        plt.subplot(3,2,3)
        plt.title('DM surface [$\\mu$m]-[H]')
        imshow_field(deformable_mirror.surface * 1e6, cmap='RdBu', vmin=-2, vmax=2, mask=ap)
        plt.xlabel('[M]')
        plt.ylabel('[M]')
        cb3=plt.colorbar()
        cb3.set_label("[rad]")

        plt.subplot(3,2,4)
        plt.title(' PSF at Shack-Hartmann [F]')
        imshow_field(wfs_image,cmap='inferno')
        plt.xlabel('[M]')
        plt.ylabel('[M]')
        cb4=plt.colorbar()
        cb4.set_label("[W]")

        plt.subplot(3,2,5)
        plt.title(' PSF at focal [D]')
        imshow_field(np.log10(wf_sci_focal_plane.power / wf_sci_focal_plane.power.max()), vmin=-6, vmax=0, cmap='inferno')
        plt.xlabel('[M]')
        plt.ylabel('[M]')
        cb5=plt.colorbar()
        cb5.set_label("[W]")


        plt.tight_layout()
        base_output_dir = 'simulation_output'
        os.makedirs(base_output_dir, exist_ok=True)
        out_path = os.path.join(
            base_output_dir,
            f"OA_{wavelength*1e6:.2f}um_r0ref_{r0_ref*1e3:.1f}mm_run_{run_number}num_airy{num_airy}.png"
            )
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"✅ Saved combined figure to: {out_path}")


    output_path = "AO_simulation_log.csv"
    df_log = pd.DataFrame(log_rows)
    if os.path.exists(output_path):
        # לקרוא את הקובץ הקיים ולהוסיף לו את השורות החדשות
        df_old = pd.read_csv(output_path)
        df_all = pd.concat([df_old, df_log], ignore_index=True)

    else:
        # אם אין קובץ ישן – נתחיל חדש
        df_all = df_log

    df_all.to_csv(output_path, index=False)
    print("Appended log to AO_simulation_log.csv")
    OA_power=float(np.sum(wf_sci_focal_plane.power))/norm
else:
    OA_power=prop(wf1).power
print(OA_power)
# ---------- 8) Optional: overlay circle on both PSFs ----------
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
    utils.plot_phase_screen(fig,axes[0,3],prop(wf_after_dm).phase,extent_pupil_mm , title="AO correction (phase)")
    utils.plot_psf_on(fig,axes[1,3],prop(wf_after_dm).power,alpha,f_m,extent_focal_mm,scale_mm,title="AO correction (psf)",log_scale=True) 
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