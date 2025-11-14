import os
import utils
import numpy as np
import matplotlib.pyplot as plt
try:
    from hcipy import *
except Exception as e:
    raise ImportError("HCIPy is required. Install it with: pip install hcipy") from e
from tqdm.notebook import tqdm
import scipy.ndimage as ndimage

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
    USE_OA
except NameError:
    USE_OA = False
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
N=512
pupil_grid = make_pupil_grid(N, D * oversz)

ap_gen = make_obstructed_circular_aperture(D, eps, num_spiders=4, spider_width=spider_w)
ap = evaluate_supersampled(ap_gen, pupil_grid, 4) 
ref_wavelength=8e-7
# ---------- 2) Wavelength & focal grid ----------
# wavelength = 2.2e-6  # [m] science wavelength
q = 8
num_airy = 60
spatial_res = wavelength / D  # [rad] per λ/D
focal_grid = make_focal_grid(q=q, num_airy=num_airy,spatial_resolution=spatial_res, pupil_diameter=D,focal_length=focal_dim)

prop = FraunhoferPropagator(pupil_grid, focal_grid) # transforms from pupil_grid to focal_grid

# ---------- 3) Wavefront (before turbulence) ----------
wf0=Wavefront(ap,wavelength) #wf_pupil
#wf0.electric_field = Wavefront(ap, wavelength).electric_field+Wavefront(ap,ref_wavelength).electric_field
# wf0.total_power = 1.0

psf0 = prop(wf0).power  # unaberrated PSF (relative units)
phase0 = prop(wf0).phase  # unaberrated PSF (relative unitxs)

# ---------- 4) Single-layer turbulence ----------
seeing = 0.6       # [arcsec] @ 500 nm
L0 = 40.0          # [m]
tau0 = 5e-3        # [s]
lam_ref = 500e-9   # [m]

r0 = r0_ref * (wavelength / lam_ref) ** (6.0 / 5.0)    
# r0 = seeing_to_fried_parameter(seeing)                # [m] at 500 nm
Cn2 = Cn_squared_from_fried_parameter(r0, lam_ref)    # [m^(-2/3)]
v = 0.314 * r0 / tau0                                     # [m/s] 

# HCIPy 0.7.0 API fallback
try:
    layer = InfiniteAtmosphericLayer(pupil_grid, Cn2, L0, v)
except TypeError:
    layer = InfiniteAtmosphericLayer(Cn2, L0, v)

if TurbulencLayer is True:
    wf1 = layer(wf0)      
else:
    r0_ref=1
    wf1=wf0           # wavefront AFTER turbulence
psf1 = prop(wf1).power           # instantaneous PSF with turbulence
phase1 = prop(wf1).phase           # instantaneous PSF with turbulence
Wf_in_focal=prop(wf1)
# ---------- 5) Bucket definition: 9 µm circle in focal plane ----------
Fnum_sci = 50.0                 # adjust to your optics if needed 
f_eff = Fnum_sci * D            # [m] effective focal length
diam_phys = 9e-6                # [m]
rad_phys  = diam_phys / 2.0     # [m]
alpha = rad_phys / f_eff        # [rad] angular radius on focal plane


theta = np.sqrt(focal_grid.x**2 + focal_grid.y**2)  # [rad] 
bucket_mask = (theta <= alpha).astype(float)

# ---------- 6) Proper integration with grid weights (HCIPy 0.7.0) ----------
# w = focal_grid.weights 
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
#check for energy conservation

print("*"*80)
I_grid=np.abs(wf1.electric_field)**2
I_focal=np.abs(Wf_in_focal.electric_field)**2
weights_grid=wf1.grid.weights
print(weights_grid)
weights_focal=Wf_in_focal.grid.weights
print(weights_focal)
Wf_in_focal_power=np.sum(Wf_in_focal.power)
wf1_power=np.sum(wf1.power)
print("wf1 power: ",np.sum(Wf_in_focal.power))
print("focal power: ",np.sum(wf1.power))
Energy_conv=100*Wf_in_focal_power/wf1_power
print(Energy_conv)
print("*"*80)

#endregion
#------------
if USE_OA:
    wavelength_wfs=8e-7
    num_modes = 30
    num_modes = 30
    dm_modes = make_disk_harmonic_basis(
        pupil_grid, num_modes, D, 'neumann'
    )
    dm_modes = ModeBasis(
        [mode / np.ptp(mode) for mode in dm_modes], pupil_grid
    )
    deformable_mirror = DeformableMirror(dm_modes)
    deformable_mirror.flatten
    response_matrix = []
    probe_amp = 0.01 * wavelength_wfs

    wf_calib = Wavefront(ap, wavelength_wfs)
    wf_calib.total_power = 1.0
    #---------SH WFS Setup-----------
    f_number = 50
    num_lenslets = 40
    sh_diameter = 5e-3  # [m] SH beam diameter

    magnification = sh_diameter / D
    magnifier = Magnifier(magnification)
    shwfs = SquareShackHartmannWavefrontSensorOptics(
        pupil_grid.scaled(magnification),
        f_number,
        num_lenslets,
        sh_diameter
    )
    spatial_resolution_wfs = wavelength_wfs / D      # [rad] per λ/D (roughly)
    focal_grid_wfs = make_focal_grid(
        q=4,
        num_airy=2,             # enough to capture each SH spot
        spatial_resolution=spatial_resolution_wfs,
        pupil_diameter=D,
        focal_length=None
    )
    shwfse = ShackHartmannWavefrontSensorEstimator(
        shwfs.mla_grid,
        shwfs.micro_lens_array.mla_index
    )
    wf_ref_wfs = Wavefront(ap, wavelength_wfs)
    camera = NoiselessDetector(focal_grid_wfs)
    camera.integrate(shwfs(magnifier(wf_ref_wfs)), 1.0)
    image_ref = camera.read_out()
    slopes_ref = shwfse.estimate([image_ref])
    # ---- Select estimation subapertures based on flux ----
    fluxes = ndimage.sum(image_ref, shwfse.mla_index, shwfse.estimation_subapertures)
    flux_limit = fluxes.max() * 0.5  # לדוגמה – 50% מהפלוקס המקסימלי

    estimation_subapertures = shwfs.mla_grid.zeros(dtype='bool')
    estimation_subapertures[
        shwfse.estimation_subapertures[fluxes > flux_limit]
    ] = True

    # בונים מחדש את ה-estimator עם רק הסאב־אפרצ’רים הטובים
    shwfse = ShackHartmannWavefrontSensorEstimator(
        shwfs.mla_grid,
        shwfs.micro_lens_array.mla_index,
        estimation_subapertures
    )

    # מחשבים מחדש את slopes_ref עם ה-estimator החדש
    slopes_ref = shwfse.estimate([image_ref])

    # ----------------------------- Interaction matrix --------------------------
    response_matrix = []
    probe_amp = 0.01 * wavelength_wfs

    wf_calib = Wavefront(ap, wavelength_wfs)
    wf_calib.total_power = 1.0

    print("AO.py: Calibrating interaction matrix...")
    for i in tqdm(range(num_modes)):
        slope = 0
        amps = [-probe_amp, probe_amp]

        for amp in amps:
            deformable_mirror.flatten()
            deformable_mirror.actuators[i] = amp

            dm_wf = deformable_mirror.forward(wf_calib)
            wfs_wf = shwfs(magnifier(dm_wf))

            camera.integrate(wfs_wf, 1.0)
            image = camera.read_out()

            slopes = shwfse.estimate([image])
            slope += amp * slopes / np.var(amps)

        response_matrix.append(slope.ravel())

    response_matrix = ModeBasis(response_matrix)

    # Reconstruction matrix (Tikhonov regularization)
    rcond = 1e-3
    reconstruction_matrix = inverse_tikhonov(
        response_matrix.transformation_matrix,
        rcond=rcond
    )

    print("AO.py: Interaction and reconstruction matrices ready.")

    #-----------7.2 add Adaptive Optics both PSFs ----------
    leakage = 0.01
    num_iterations = 20
    wf0_wfs = Wavefront(ap, wavelength_wfs)
    delta_t = 0.001  # [s]
    burn_in_iterations = 5
    gain=0.3
    leakage=0.01
    coro = PerfectCoronagraph(ap, 4)
    long_exposure = focal_grid.zeros()
    long_exposure_coro = focal_grid.zeros()
    for timestep in tqdm(range(num_iterations)):
        layer.t = timestep * delta_t
        # Propagate through atmosphere and deformable mirror.
        wf_wfs_after_atmos = layer(wf0_wfs)
        wf_wfs_after_dm = deformable_mirror(wf_wfs_after_atmos)

        # Propagate through SH-WFS
        wf_wfs_on_sh = shwfs(magnifier(wf_wfs_after_dm))

        # Propagate the NIR wavefront
        wf_sci_focal_plane = prop(deformable_mirror(layer(wf0)))
        wf_sci_coro = prop(coro(deformable_mirror(layer(wf0))))

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

    print("AO.py: Closed-loop AO finished.")
    wf_wfs_after_dm_prop=prop(wf_wfs_after_dm)
    print("after AO: ",np.sum(wf_wfs_after_dm_prop.power))
    print(np.sum(wf_wfs_after_dm.power))
    print("PSF1: ",np.sum(psf1))
else:
    wf_wfs_after_dm_prop=prop(wf1)
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
    xmin_m = psf0.grid.x.min()*f_m
    xmax_m = psf0.grid.x.max()*f_m
    ymin_m = psf0.grid.y.min()*f_m
    ymax_m = psf0.grid.y.max()*f_m
    extent_focal_mm = [xmin_m*scale_mm, xmax_m*scale_mm, ymin_m*scale_mm, ymax_m*scale_mm]
    # 1) Figure & axes
    fig, axes = plt.subplots(2, 4, figsize=(20, 4.5), dpi=150)

    # --- (a) Phase screen (pupil plane) ---
    utils.plot_phase_screen(fig,axes[0,0],phase_screen,extent_pupil_mm,title="A) Atmosphere Phase")
    utils.plot_psf_on(fig,axes[1,0],power_screen,alpha,f_m,extent_focal_mm,scale_mm,title="A) Atmosphere PSF")
    # --- (b) Unaberrated PSF ---
    utils.plot_phase_screen(fig,axes[0,1],phase_screen,extent_pupil_mm,mask = ap,title="B) Unaberrated Phase (Before)")
    utils.plot_psf_on(fig,axes[1,1], ap,alpha,f_m,extent_focal_mm,scale_mm, title="B) Unaberrated PSF (before)")
    # --- (c) Turbulent PSF ---
    utils.plot_phase_screen(fig,axes[0,2],phase1,extent_pupil_mm , title="D) Turbulent Phase (after)")
    utils.plot_psf_on(fig,axes[1,2], psf1,alpha,f_m,extent_focal_mm,scale_mm,title="D) Turbulent PSF (after)")
    # --- (d) AO-corrected PSF ---
    utils.plot_phase_screen(fig,axes[0,3],wf_wfs_after_dm_prop.phase,extent_pupil_mm , title="AO correction (phase)")
    utils.plot_psf_on(fig,axes[1,3], wf_wfs_after_dm_prop.power,alpha,f_m,extent_focal_mm,scale_mm,title="AO correction (psf)") 
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

utils.update_csv(wavelength, r0_ref, run_number,focal_dim,power0_in_bucket,power0_total,frac0,power1_in_bucket,power1_total,frac1,conservation_of_energy=Energy_conv)
