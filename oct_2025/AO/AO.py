from hcipy import *
from utils import *
from oct_2025.main.params import *
check_dir()
try: 
    num_airy
except NameError:
    num_airy=5
from scipy.ndimage import sum as scipy_sum


def OA(num_airy,r0_ref_list,save_images,run_number):  
    focal_grid = make_focal_grid(q=q_OA, num_airy=num_airy,spatial_resolution=spatial_res)
    f_number = 50
    num_lenslets = 40 # 40 lenslets along one diameter
    sh_diameter = 5e-3 # m
    wf0=Wavefront(ap,wavelength)
    magnification = sh_diameter / telescope_diameter
    magnifier = Magnifier(magnification)
    shwfs = SquareShackHartmannWavefrontSensorOptics(pupil_grid.scaled(magnification), f_number, \
                                                 num_lenslets, sh_diameter)
    shwfse = ShackHartmannWavefrontSensorEstimator(shwfs.mla_grid, shwfs.micro_lens_array.mla_index)
    camera = NoiselessDetector(focal_grid)
    camera.integrate(shwfs(magnifier(wf0)), 1)
    image_ref = camera.read_out()
    fluxes = scipy_sum(image_ref, shwfse.mla_index, shwfse.estimation_subapertures)
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
        velocity = 0.314 * fried_parameter / tau0
        Cn_squared = Cn_squared_from_fried_parameter(fried_parameter, 500e-9)
        layer = InfiniteAtmosphericLayer(pupil_grid, Cn_squared, L0, velocity)
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
            wf_focal_plane = propagator(deformable_mirror(layer(wf_wfs)))
            wf_sci_coro = propagator(coro(deformable_mirror(layer(wf_wfs))))

            # Read out WFS camera
            camera.integrate(wf_wfs_on_sh, delta_t)
            wfs_image = camera.read_out()
            wfs_image = large_poisson(wfs_image).astype('float')

            # Accumulate long-exposure image
            if timestep >= burn_in_iterations:
                long_exposure += wf_focal_plane.power / (num_iterations - burn_in_iterations)
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
                D_power = float(np.sum(wf_focal_plane.power))

                log_rows.append({
                    "timestep": timestep,
                    "E_power_sum": E_power,
                    "D_power_sum": D_power,
                    "num_airy":num_airy,
                    "r0_ref":fried_parameter,
                })
        if save_images:
            FIG_OA=plt.clf()
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
            imshow_field(np.log10(wf_focal_plane.power / wf_focal_plane.power.max()), vmin=-6, vmax=0, cmap='inferno')
            plt.xlabel('[M]')
            plt.ylabel('[M]')
            cb5=plt.colorbar()
            cb5.set_label("[W]")


            plt.tight_layout()
            base_output_dir = 'simulation_output'
            os.makedirs(base_output_dir, exist_ok=True)
            out_path = os.path.join(
            base_output_dir,
            f"OA_{wavelength*1e6:.2f}um_r0ref_{fried_parameter*1e3:.1f}mm_run_{run_number}num_airy{num_airy}.png"
            )
            plt.savefig(out_path, dpi=300, bbox_inches="tight")
            print(f"✅ Saved combined figure to: {out_path}")
        

        plt.close()
        anim.close()
    return wf_focal_plane

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
try: 
    run_number
except NameError:
    run_number=99
print(type(r0_ref_list)!=list)    


a_list=[]
print("C1: ",a_list==[])
