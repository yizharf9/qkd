
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  # adds FSOC to sys.path

import main.params as p  # module alias

import itertools
import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator, NullFormatter, ScalarFormatter
from matplotlib import animation
from IPython.display import HTML
from mpl_toolkits.axes_grid1 import make_axes_locatable
from hcipy import *
try:
    from hcipy import *
except Exception as e:
    raise ImportError("HCIPy is required. Install it with: pip install hcipy") from e
from tqdm import tqdm
from main.params import *
from scipy.ndimage import sum as scipy_sum
def check_dir():
    print("Checking current working directory...")
    current_directory_name = os.path.basename(os.getcwd())
    if current_directory_name == 'FSOC':
        print(f"✅ Success: Script is running from the correct directory ('{current_directory_name}').")
    else:
        print(f"⚠️ Warning: Script is NOT running from the 'FSOC' directory.")
        print(f"   Current directory is: '{current_directory_name}'")
        exit("Execution stopped. Please run the script from the 'FSOC' directory.")
    print("-" * 60) # Visual separator

def cli_tool():
    save_images_prompt = input("save images (y/n) ?  ") #! <=== change to save photos in single turblance.py run 
    if save_images_prompt == "y" :
        save_images = True
    elif save_images_prompt == "n" :
        save_images = False
    else : 
        exit("not a valid input!")

    TurbulencLayer_prompt = input("add layer with no turbulance (y/n) ?  ") #! <=== change to add layer without turb photos in single turblance.py run 
    if TurbulencLayer_prompt == "y" :
        TurbulencLayer = True
    elif TurbulencLayer_prompt == "n" :
        TurbulencLayer = False
    else : 
        exit("not a valid input!")
    

def update_csv( wavelength,
                r0_ref_val,
                run_num,
                power_in_bucket_before,
                total_power_before,
                precentage_before,
                total_power_after,
                precentage_after,
                conservation_of_energy,
                num_airy,
                power_after_AO=None,
                power_in_bucket_after=1111111
                ):

#head= wavelength ,r0_ref,run_number,focsl_dim,power_in_bucket_before_turbulance,total_power_before_turbulance,precentage_before_turbulance,power_in_bucket_after_turbulance,total_power_after_turbulance,precentage_after_turbulance,conservation_of_energy[%],time

    file_path = "Data/massive_output.csv"
    columns = [
        "wavelength",
        "r0_ref",
        "run_number",
        "num_airy"
        "power_in_bucket_before_turbulance",
        "total_power_before_turbulance",
        "precentage_before_turbulance",
        "power_in_bucket_after_turbulance",
        "total_power_after_turbulance",
        "precentage_after_turbulance",
        "conservation_of_energy[%]",
        "power_after_AO",
        "time"
        ]

    new_row_df = pd.DataFrame([{
        "wavelength": wavelength,
        "r0_ref": r0_ref_val,
        "run_number": run_num,
        "num_airy": num_airy,
        "power_in_bucket_before_turbulance": power_in_bucket_before,
        "total_power_before_turbulance": total_power_before,
        "precentage_before_turbulance": precentage_before,
        "power_in_bucket_after_turbulance": power_in_bucket_after,
        "total_power_after_turbulance": total_power_after,
        "precentage_after_turbulance": precentage_after,
        "conservation_of_energy[%]" : conservation_of_energy,
        "power_after_AO": power_after_AO,
        "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }])

    if os.path.exists(file_path):
        new_row_df.to_csv(file_path, mode='a', header=False, index=False)
    else:
        new_row_df.to_csv(file_path, mode='w', header=columns, index=False)
    print("Operation complete. Data has been saved.\n")

# helper לציור phase_screen
def plot_phase_screen(fig,ax,phase_screen,extent_pupil_mm,mask=None,title="Turbulence phase screen [rad]"):
    # ax = axes[0]    
    # print(str(mask.shape) + "!!!!!")
    if mask is not None:
        im_phase_field = imshow_field(phase_screen,ax=ax,mask=mask, cmap='RdBu',vmin=-np.pi, vmax=np.pi)
    else:
        im_phase_field = imshow_field(phase_screen,ax=ax,cmap='RdBu',vmin=-np.pi, vmax=np.pi)
    # im_phase = ax.show(im_phase_field, origin='lower', cmap='RdBu',vmin=-np.pi, vmax=np.pi, extent=extent_pupil_mm)
    ax.set_title(title)
    ax.set_xlabel('x [mm]')
    ax.set_ylabel('y [mm]')
    ax.set_aspect('equal', adjustable='box')
    
    div = make_axes_locatable(ax)
    cax = div.append_axes("right", size="5%", pad=0.06)
    cb = fig.colorbar(im_phase_field, cax=cax)
    cb.set_label('Phase [rad]')
import numpy as np

def add_noise_to_wavefront(
        propagated_wavefront,
        telescope_diameter,
        stellar_magnitude=8.0,
        flux_zero_point=1.5e10, # Photons/sec/m^2
        throughput=0.8,
        exposure_time=0.01,     # Seconds
        read_noise=5.0,         # Moved to args for flexibility
        dark_current=0.1        # Moved to args
    ):
    """
    Simulates detector noise (Photon shot noise + Read noise + Dark current).
    Returns the detected image (electron counts), NOT a Wavefront.
    """
    focal_grid = propagated_wavefront.grid
    collecting_area = np.pi * (telescope_diameter / 2)**2

    # 1. Calculate Total Photons entering the telescope per second
    # Flux ZP * 10^(-0.4*mag) gives flux per m^2
    total_flux_in_aperture = (flux_zero_point * 10**(-0.4 * stellar_magnitude) * collecting_area * throughput)

    # 2. Create the Normalized Intensity Distribution
    # This ensures we distribute the total physical photons according to the wavefront shape
    intensity_distribution = propagated_wavefront.power
    if propagated_wavefront.total_power > 0:
        intensity_distribution /= propagated_wavefront.total_power
    
    # 3. Scale to Physical Photon Rate (Photons / sec / pixel)
    image_photons_per_sec = intensity_distribution * total_flux_in_aperture

    # 4. Detector Simulation
    # Assuming 'NoisyDetector' is a class from a library like HCIPy
    detector = NoisyDetector(focal_grid)

    detector.include_photon_noise = True
    detector.read_noise = read_noise            
    detector.dark_current_rate = dark_current   

    # Integrate accumulates photons over time -> converts to electrons
    detector.integrate(image_photons_per_sec, dt=exposure_time)

    # Read out adds read noise and discretizes
    image_noisy = detector.read_out()

    # Return the image array (Intensity), not a Wavefront (Field)
    return image_noisy

def use_adaptive_optics(
    wf0,
    psf1,
    pupil_grid,
    focal_grid,
    layer,
    prop,
    D,
    ap,
):
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
        camera.integrate(wf_wfs_on_sh, delta_ft)
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
    return wf_wfs_after_dm_prop

def check_energy_conservation(
    wf1,
    Wf_in_focal
    ):
    print("\n-- Checking Energy conservation between pupil and focal --")
    I_grid=np.abs(wf1.electric_field)**2
    I_focal=np.abs(Wf_in_focal.electric_field)**2
    weights_grid=wf1.grid.weights
    print("weights_grid: {weights_grid}")
    weights_focal=Wf_in_focal.grid.weights
    print("weights_focal: {weights_focal}")
    Wf_in_focal_power=np.sum(Wf_in_focal.power)
    wf1_power=np.sum(wf1.power)
    print("wf1 power: ",np.sum(Wf_in_focal.power))
    print("focal power: ",np.sum(wf1.power))
    Energy_conv=100*Wf_in_focal_power/wf1_power
    print(f"Energy_conv [%]: {Energy_conv}")
    return Energy_conv

# helper לציור PSF
def plot_psf_on(fig,ax, psf,alpha,f_m,extent_focal_mm,scale_mm, title = "PSF plot",log_scale=True):
    if log_scale:
        psf_img = np.log10((psf / psf.max()).shaped + 1e-12)
    else:
        psf_img = (psf / psf.max()).shaped
    im = ax.imshow(psf_img, origin='lower', extent=extent_focal_mm,cmap='inferno', vmin=-6, vmax=0)
    circ = mpatches.Circle((0.0, 0.0), radius=alpha*f_m*scale_mm,fill=False, linewidth=1.5)
    ax.add_patch(circ)
    ax.set_title(title)
    ax.set_xlabel('x [mm]'); ax.set_ylabel('y [mm]')
    ax.set_aspect('equal', adjustable='box')
    div = make_axes_locatable(ax) 
    cax = div.append_axes("right", size="5%", pad=0.06)
    cb = fig.colorbar(im, cax=cax)
    cb.set_label(r'$\log_{10}(\mathrm{Intensity}/\max)$')



def animate_wavefronts(images_folder_dir,image_titles=None, interval=50, repeat_delay=1000):
    """Creates an animation from a list of Matplotlib image artists.
    This function is designed to be used in a Jupyter/IPython notebook
    environment to display the animation inline.

    Args:
        images: A list of Matplotlib image artists (e.g., the objects
                returned by `hci.imshow_field` or `plt.imshow`).
        interval: Delay between frames in milliseconds.
        repeat_delay: Delay in milliseconds before repeating the animation.

    Returns:
        An IPython.display.HTML object for displaying the animation.
    """
        
    if not images_folder_dir:
        print("Warning: The image list is empty. No animation will be created.")
        return None
    
    images_dirs = []
    for image_dir in os.listdir(images_folder_dir):
        if image_dir.endswith(".png"):
            images_dirs.append(image_dir)
    images_dirs.sort()
    
    if image_titles is None:
        image_titles = [i for i in range(len(images_dirs))]
    
    images = [plt.imshow(plt.imread(os.path.join(images_folder_dir,images_dir))) for images_dir in images_dirs]
    plt.close()
    
    # Get the figure from the first image artist in the list
    fig = images[0].figure

    # ArtistAnimation requires a list of lists, where each inner list is a frame.
    # We'll wrap each of our images in its own list to create the frames.
    artist_list = [[img] for img in images]

    # Create the animation
    anim = animation.ArtistAnimation(
                                        fig, 
                                        artist_list,
                                        interval=interval,
                                        repeat_delay=repeat_delay,
                                        blit=True
                                    )

    # Close the static figure to prevent it from displaying
    plt.close(fig)

    # Return the animation as an HTML5 video
    return HTML(anim.to_jshtml())
#----------------methods for plot_and_graph------------------------
def Load_csv(path_file):
    df = pd.read_csv(path_file)
    structure="-"*20+"structure"+20*"-"+"\n"
    structure=structure+f"Loaded {len(df)} rows from {path_file}"+"\n"
    structure=structure+str("#"*50+"\n")
    df.columns = df.columns.str.strip().str.lower()
    structure=structure+"Columns:"+str(list(df.columns))+"\n"
    structure=structure+"-"*20+"structure"+20*"-"
    return [df,structure]
def pick(colnames, *candidates):
    for c in candidates:
        if c in colnames:
            return c
        hits = [cn for cn in colnames if c in cn]
        if hits:
            return hits[0]
    return None

#----------------time asstimate------------------------
def time_estimate(current_time,begin_time,current_run,num_of_run): 
    """
    return the number in sec until the code is finish 
    assume linear progression 
    """
    runtime=current_time-begin_time #time the code is running
    avg_TFR=runtime/current_run #avg time for run
    return avg_TFR*num_of_run-runtime
#-----plots----------


def plot_mean_line_loglin(
    df: pd.DataFrame,
    wl_col: str,
    focal_col: str,
    wavelengths: list,
    focals: list,
    x_col: str,
    y_col: str,
    ax: plt.Axes,
    title: str = None,
    x_label: str = None,
    y_label: str = None,
    required_x_values: list = None,   # pass exact x values if you want strict enforcement; otherwise inferred
    linewidth: float = 1.8,
    marker: str = 'o',
    markersize: float = 3.5,
    strict_missing: bool = True       # raise error if x values missing for any (wl,focal)
):
    """
    Plot mean(y_col) vs x_col on linear axes.
    - linestyle encodes wavelength
    - color encodes focal_dim
    - returns the provided Axes (ax)
    - raises ValueError if strict_missing and a (wl,focal) series is missing any required x values
    """

    # --- basic validation ---
    for col in (wl_col, focal_col, x_col, y_col):
        if col not in df.columns:
            raise ValueError(f"Missing column '{col}' in DataFrame.")
    if ax is None:
        raise ValueError("Must pass a valid Matplotlib Axes (ax).")

    # Work on a safe copy, ensure numeric
    work = df.copy()
    work[x_col] = pd.to_numeric(work[x_col], errors='coerce')
    work[y_col] = pd.to_numeric(work[y_col], errors='coerce')
    work = work.dropna(subset=[wl_col, focal_col, x_col, y_col])

    # Filter to requested wavelengths & focals (keeps styling stable across subplots)
    work = work[work[wl_col].isin(wavelengths) & work[focal_col].isin(focals)]

    if work.empty:
        raise ValueError("No data left after filtering by wavelengths and focal values.")

    # --- style maps (color by focal, linestyle by wavelength) ---
    # Colors from the default prop cycle (repeat if needed)
    default_colors = plt.rcParams['axes.prop_cycle'].by_key().get('color', ['C0','C1','C2','C3','C4','C5','C6','C7','C8','C9'])
    color_map = {fv: default_colors[i % len(default_colors)] for i, fv in enumerate(focals)}

    linestyles = ['solid', 'dashed', 'dashdot', 'dotted']
    ls_map = {wv: linestyles[i % len(linestyles)] for i, wv in enumerate(wavelengths)}

    # --- aggregation: mean of y per (wl, focal, x) ---
    g = (work
        .groupby([wl_col, focal_col, x_col], as_index=False)[y_col]
        .mean()
        )

    # --- required x values check ---
    if required_x_values is None:
        # infer exact x-grid from the data (strict)
        required_x_values = np.sort(g[x_col].unique())
    required_x_values = np.asarray(required_x_values)

    # For fast membership test:
    required_set = set(required_x_values.tolist())

    # --- plot per (wl, focal) series ---
    for wv in wavelengths:
        for fv in focals:
            sub = g[(g[wl_col] == wv) & (g[focal_col] == fv)]
            if sub.empty:
                if strict_missing:
                    raise ValueError(f"No data for wavelength={wv}, focal={fv}.")
                else:
                    continue

            # enforce exact x presence (strict “exact x values”)
            series_x = np.sort(sub[x_col].to_numpy())
            series_set = set(series_x.tolist())
            if strict_missing and series_set != required_set and len(list(series_set-required_set)) > 0:
                missing = sorted(list(required_set - series_set))
                extra   = sorted(list(series_set - required_set))
                raise ValueError(
                    f"Series (wavelength={wv}, focal={fv}) has x mismatch.\n"
                    f"  Missing x: {missing}\n"
                    f"  Extra x:   {extra}"
                )

            # sort by x
            sub = sub.sort_values(x_col, kind='mergesort')

            ax.plot(
                sub[x_col].to_numpy(),
                sub[y_col].to_numpy(),
                linestyle=ls_map[wv],
                color=color_map[fv],
                marker=marker,
                markersize=markersize,
                linewidth=linewidth,
                label=f"λ={wv}, f={fv}"  # combined legend (we’ll also add split legends)
            )

    # --- labels, grid, title ---
    if x_label:
        ax.set_xlabel(x_label)
    if y_label:
        ax.set_ylabel(y_label)
    if title:
        ax.set_title(title)
    ax.grid(True, which='both', linestyle=':', linewidth=0.7)
    ax.set_xscale('log')
    ax.set_yscale('linear')

    # --- split legends: one for wavelength (linestyle), one for focal (color) ---
    wl_handles = [Line2D([0], [0], color='black', linestyle=ls_map[wv], label=f"λ={wv}") for wv in wavelengths]
    focal_handles = [Line2D([0], [0], color=color_map[fv], linestyle='solid', label=f"f={fv}") for fv in focals]

    # Place legends without overlapping
    leg1 = ax.legend(handles=wl_handles, title="Wavelength (linestyle)", loc='best')
    ax.add_artist(leg1)
    ax.legend(handles=focal_handles, title="Focal (color)", loc='best')

    return ax





#--------------AO--------
"""
This part activates the adaptive optics correction
 - using this code may cause significant computation time. 
 An efficient computation would be for num-airy [3,5]
"""


def AO(num_airy,save_images,run_number,r0_ref_list=[],single_r0=None):  
    """
    num_airy -- number factor of lamda/D of the focal plane
    r0_ref_list -- list of r0 to test for, for a single use single_r0
    save_images -- boolean True/False
    run_number -- intefer_ for massive runs. 
    single_r0 -- for single use of AO
    """
    if single_r0 and r0_ref_list==[] :
        r0_ref_list=[single_r0]
    focal_grid = make_focal_grid(q=q_AO, num_airy=num_airy,spatial_resolution=spatial_res)
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
            FIG_AO=plt.clf()
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
            f"Final_AO_{wavelength*1e6:.2f}um_r0ref_{fried_parameter*1e3:.1f}mm_run_{run_number}num_airy {num_airy}.png"
            )
            plt.savefig(out_path, dpi=300, bbox_inches="tight")
            print(f"✅ Saved combined figure to: {out_path}")
        plt.close()    

        output_path ="Data/AO_simulation_log.csv"
        df_log=pd.DataFrame(log_rows)
    if os.path.exists(output_path):
        # לקרוא את הקובץ הקיים ולהוסיף לו את השורות החדשות
        df_old = pd.read_csv(output_path)
        df_all = pd.concat([df_old, df_log], ignore_index=True)
    else:
        # אם אין קובץ ישן – נתחיל חדש
        df_all = df_log

    df_all.to_csv(output_path, index=False)
    print("Appended log to AO_simulation_log.csv")
    return [wf_wfs_after_atmos,deformable_mirror,wf_focal_plane,wfs_image]

import os
import matplotlib.pyplot as plt

def save_figure(fig, filename, folder="plots", dpi=200):
    """
    Save a matplotlib figure into a folder (relative path).
    
    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure object to save.
    filename : str
        The file name, e.g., "before_after.png".
    folder : str, optional
        Relative folder path where the image will be saved (default: "plots").
    dpi : int, optional
        Image resolution.
    """
    # 1. Ensure directory exists
    save_path = os.path.join(os.getcwd(), folder)
    os.makedirs(save_path, exist_ok=True)

    # 2. Build full path
    full_path = os.path.join(save_path, filename)

    # 3. Save
    fig.savefig(full_path, dpi=dpi, bbox_inches='tight')
    print(f"✅ Saved figure to: {full_path}")


# ============================================================================
# AO PIPELINE: set_AO, calibration (correction is local in single_simulation)
# ============================================================================

from dataclasses import dataclass
from typing import Optional, Any

@dataclass
class AOConfig:
    """Configuration for AO pipeline."""
    
    # Grids / optics
    pupil_grid: Any
    focal_q: int
    spatial_resolution: float
    telescope_diameter: float
    aperture: Any  # aperture mask
    wavelength: float

    # Atmosphere parameters
    tau0: float
    L0: float

    # WFS parameters
    f_number: float = 50.0
    num_lenslets: int = 40
    sh_diameter: float = 5e-3

    # Calibration / DM parameters
    num_modes: int = 500
    probe_amp_factor: float = 0.01  # probe_amp = factor * wavelength
    rcond: float = 1e-3

    # Loop parameters
    delta_t: float = 1e-3
    num_iterations: int = 800
    burn_in_iterations: int = 5
    gain: float = 0.3
    leakage: float = 0.01

    # Flux / magnitude
    zero_magnitude_flux: float = 3.9e10
    stellar_magnitude: float = 5.0

    # Coronagraph
    coro_inner_radius_lamD: float = 4.0

    # Output
    base_output_dir: str = "simulation_output"
    csv_output_path: str = "Data/AO_simulation_log.csv"


@dataclass
class AOState:
    """Runtime state for AO pipeline."""
    
    # Objects created in set_AO
    focal_grid: Any
    magnifier: Any
    shwfs: Any
    shwfse: Any
    camera: Any
    dm_modes: Any
    deformable_mirror: Any
    propagator: Any
    coro: Any

    # Reference / calibration products
    image_ref: Any
    slopes_ref: Any
    response_matrix: Optional[Any] = None
    reconstruction_matrix: Optional[np.ndarray] = None

    # Convenience wavefronts
    wf0: Any = None
    wf_wfs: Any = None
    norm_power: Optional[int] = None


def set_AO(num_airy: int, cfg: AOConfig) -> AOState:
    """
    Stage 1: build all static AO objects (grids, WFS, DM basis, camera, propagator).
    Returns an AOState WITHOUT calibration matrices (those are built in calibration()).
    
    Args:
        num_airy: Number of Airy disk diameters in focal plane
        cfg: AOConfig object with all parameters
    
    Returns:
        AOState: AO system state (ready for calibration)
    """
    focal_grid = make_focal_grid(
        q=cfg.focal_q, 
        num_airy=num_airy, 
        spatial_resolution=cfg.spatial_resolution
    )

    # Wavefront for WFS reference image
    wf0 = Wavefront(cfg.aperture, cfg.wavelength)

    magnification = cfg.sh_diameter / cfg.telescope_diameter
    magnifier = Magnifier(magnification)

    shwfs = SquareShackHartmannWavefrontSensorOptics(
        cfg.pupil_grid.scaled(magnification),
        cfg.f_number,
        cfg.num_lenslets,
        cfg.sh_diameter
    )

    # Initial estimator (we'll refine estimation_subapertures based on flux)
    shwfse = ShackHartmannWavefrontSensorEstimator(shwfs.mla_grid, shwfs.micro_lens_array.mla_index)

    camera = NoiselessDetector(focal_grid)
    camera.integrate(shwfs(magnifier(wf0)), 1)
    image_ref = camera.read_out()

    # Select illuminated subapertures based on flux threshold
    fluxes = scipy_sum(image_ref, shwfse.mla_index, shwfse.estimation_subapertures)
    flux_limit = fluxes.max() * 0.5

    estimation_subapertures = shwfs.mla_grid.zeros(dtype="bool")
    estimation_subapertures[shwfse.estimation_subapertures[fluxes > flux_limit]] = True

    shwfse = ShackHartmannWavefrontSensorEstimator(
        shwfs.mla_grid,
        shwfs.micro_lens_array.mla_index,
        estimation_subapertures
    )

    slopes_ref = shwfse.estimate([image_ref])

    # DM basis
    dm_modes = make_disk_harmonic_basis(
        cfg.pupil_grid, 
        cfg.num_modes, 
        cfg.telescope_diameter, 
        "neumann"
    )
    dm_modes = ModeBasis([mode / np.ptp(mode) for mode in dm_modes], cfg.pupil_grid)
    deformable_mirror = DeformableMirror(dm_modes)

    # Science propagation
    propagator = FraunhoferPropagator(cfg.pupil_grid, focal_grid)
    coro_order = int(cfg.coro_inner_radius_lamD * 2) 
    coro = PerfectCoronagraph(cfg.aperture, coro_order)
    # WFS wavefront power normalization
    wf_wfs = Wavefront(cfg.aperture, cfg.wavelength)
    wf_wfs.total_power = cfg.zero_magnitude_flux * 10 ** (-cfg.stellar_magnitude / 2.5)
    norm_power = int(wf_wfs.total_power)

    return AOState(
        focal_grid=focal_grid,
        magnifier=magnifier,
        shwfs=shwfs,
        shwfse=shwfse,
        camera=camera,
        dm_modes=dm_modes,
        deformable_mirror=deformable_mirror,
        propagator=propagator,
        coro=coro,
        image_ref=image_ref,
        slopes_ref=slopes_ref,
        wf0=wf0,
        wf_wfs=wf_wfs,
        norm_power=norm_power,
    )


def calibration(state: AOState, cfg: AOConfig, show_progress: bool = True) -> AOState:
    """
    Stage 2: compute response matrix and reconstruction matrix (Tikhonov inverse)
    by probing each DM mode with +/- probe_amp.
    
    Args:
        state: AOState from set_AO()
        cfg: AOConfig object
        show_progress: Show tqdm progress bar
    
    Returns:
        AOState: Updated with response_matrix and reconstruction_matrix
    """
    probe_amp = cfg.probe_amp_factor * cfg.wavelength
    response_matrix_rows = []

    # Ensure reference WF has deterministic power
    state.wf0.total_power = 1

    iterator = range(cfg.num_modes)
    if show_progress:
        iterator = tqdm(iterator, desc="📊 Calibrating (response matrix)")

    for i in iterator:
        slope_accum = 0.0
        amps = [-probe_amp, probe_amp]

        for amp in amps:
            state.deformable_mirror.flatten()
            state.deformable_mirror.actuators[i] = amp

            dm_wf = state.deformable_mirror.forward(state.wf0)
            wfs_wf = state.shwfs(state.magnifier(dm_wf))

            state.camera.integrate(wfs_wf, 1)
            image = state.camera.read_out()

            slopes = state.shwfse.estimate([image])
            slope_accum += amp * slopes / np.var(amps)

        response_matrix_rows.append(slope_accum.ravel())

    response_matrix = ModeBasis(response_matrix_rows)
    reconstruction_matrix = inverse_tikhonov(
        response_matrix.transformation_matrix, 
        rcond=cfg.rcond
    )

    state.response_matrix = response_matrix
    state.reconstruction_matrix = reconstruction_matrix
    return state


# ============================================================================
# CALIBRATION DATA SAVING & SVD PLOTTING
# ============================================================================

def save_calibration_data(
    state: AOState,
    cfg: AOConfig,
    output_dir: str,
) -> dict:
    """
    Save calibration matrices as heatmaps and SVD plots to output_dir/calibration/
    
    Args:
        state: AOState with response_matrix and reconstruction_matrix
        cfg: AOConfig with configuration
        output_dir: Base output directory (e.g., output_files/dd.mm.yyyy/AO_hh:mm:ss/)
    
    Returns:
        dict with paths to saved files
    """
    from scipy.linalg import svd as scipy_svd
    
    calib_dir = os.path.join(output_dir, "calibration")
    os.makedirs(calib_dir, exist_ok=True)
    
    saved_files = {}
    
    # ========================================================================
    # Save response matrix as .npy + HEATMAP
    # ========================================================================
    if state.response_matrix is not None:
        response_matrix_array = state.response_matrix.transformation_matrix
        response_npy = os.path.join(calib_dir, "response_matrix.npy")
        
        np.save(response_npy, response_matrix_array)
        saved_files["response_matrix_npy"] = response_npy
        print(f"  ✅ Response matrix saved: {response_npy}")
        
        # Heatmap of response matrix
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(response_matrix_array, aspect='auto', cmap='RdBu_r', interpolation='nearest')
        ax.set_xlabel("Slope Index")
        ax.set_ylabel("Mode Index")
        ax.set_title("Response Matrix Heatmap")
        plt.colorbar(im, ax=ax, label="Amplitude")
        plt.tight_layout()
        
        response_heatmap = os.path.join(calib_dir, "response_matrix_heatmap.png")
        plt.savefig(response_heatmap, dpi=150, bbox_inches="tight")
        plt.close()
        
        saved_files["response_matrix_heatmap"] = response_heatmap
        print(f"  ✅ Response matrix heatmap: {response_heatmap}")
    
    # ========================================================================
    # Save reconstruction matrix as .npy + HEATMAP
    # ========================================================================
    if state.reconstruction_matrix is not None:
        recon_npy = os.path.join(calib_dir, "reconstruction_matrix.npy")
        
        np.save(recon_npy, state.reconstruction_matrix)
        saved_files["reconstruction_matrix_npy"] = recon_npy
        print(f"  ✅ Reconstruction matrix saved: {recon_npy}")
        
        # Heatmap of reconstruction matrix
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(state.reconstruction_matrix, aspect='auto', cmap='RdBu_r', interpolation='nearest')
        ax.set_xlabel("Slope Index")
        ax.set_ylabel("Mode Index")
        ax.set_title("Reconstruction Matrix Heatmap")
        plt.colorbar(im, ax=ax, label="Amplitude")
        plt.tight_layout()
        
        recon_heatmap = os.path.join(calib_dir, "reconstruction_matrix_heatmap.png")
        plt.savefig(recon_heatmap, dpi=150, bbox_inches="tight")
        plt.close()
        
        saved_files["reconstruction_matrix_heatmap"] = recon_heatmap
        print(f"  ✅ Reconstruction matrix heatmap: {recon_heatmap}")
    
    # ========================================================================
    # Plot SVD for response matrix
    # ========================================================================
    if state.response_matrix is not None:
        try:
            response_matrix_array = state.response_matrix.transformation_matrix
            
            # Ensure it's a 2D numpy array
            if response_matrix_array.ndim != 2:
                print(f"  ⚠️  Response matrix has unexpected shape: {response_matrix_array.shape}, skipping SVD")
            else:
                U, s_resp, Vt = scipy_svd(response_matrix_array, full_matrices=False)
                
                # Filter out zero or near-zero singular values
                s_resp = s_resp[s_resp > 1e-15]
                
                if len(s_resp) == 0:
                    print(f"  ⚠️  No valid singular values for response matrix")
                else:
                    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
                    
                    # Linear scale (log Y)
                    axes[0].semilogy(range(1, len(s_resp) + 1), s_resp, 'b.-', linewidth=1, markersize=4)
                    axes[0].set_xlabel("Singular Value Index")
                    axes[0].set_ylabel("Singular Value (log scale)")
                    axes[0].set_title("Response Matrix - SVD (Linear X, Log Y)")
                    axes[0].grid(True, alpha=0.3)
                    
                    # Log-log scale
                    axes[1].loglog(range(1, len(s_resp) + 1), s_resp, 'r.-', linewidth=1, markersize=4)
                    axes[1].set_xlabel("Singular Value Index (log)")
                    axes[1].set_ylabel("Singular Value (log)")
                    axes[1].set_title("Response Matrix - SVD (Log-Log)")
                    axes[1].grid(True, alpha=0.3, which="both")
                    
                    plt.tight_layout()
                    svd_resp_path = os.path.join(calib_dir, "svd_response_matrix.png")
                    plt.savefig(svd_resp_path, dpi=150, bbox_inches="tight")
                    plt.close()
                    
                    saved_files["svd_response_plot"] = svd_resp_path
                    print(f"  ✅ Response matrix SVD plot: {svd_resp_path}")
                    
                    # Save SVD values as .npy
                    svd_resp_npy = os.path.join(calib_dir, "svd_response_values.npy")
                    np.save(svd_resp_npy, s_resp)
                    saved_files["svd_response_values"] = svd_resp_npy
        except Exception as e:
            print(f"  ❌ Error plotting response matrix SVD: {e}")
    
    # ========================================================================
    # Plot SVD for reconstruction matrix
    # ========================================================================
    if state.reconstruction_matrix is not None:
        try:
            # Ensure it's a 2D numpy array
            recon_matrix = np.asarray(state.reconstruction_matrix)
            if recon_matrix.ndim != 2:
                print(f"  ⚠️  Reconstruction matrix has unexpected shape: {recon_matrix.shape}, skipping SVD")
            else:
                U_recon, s_recon, Vt_recon = scipy_svd(recon_matrix, full_matrices=False)
                
                # Filter out zero or near-zero singular values
                s_recon = s_recon[s_recon > 1e-15]
                
                if len(s_recon) == 0:
                    print(f"  ⚠️  No valid singular values for reconstruction matrix")
                else:
                    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
                    
                    # Linear scale (log Y)
                    axes[0].semilogy(range(1, len(s_recon) + 1), s_recon, 'g.-', linewidth=1, markersize=4)
                    axes[0].set_xlabel("Singular Value Index")
                    axes[0].set_ylabel("Singular Value (log scale)")
                    axes[0].set_title("Reconstruction Matrix - SVD (Linear X, Log Y)")
                    axes[0].grid(True, alpha=0.3)
                    
                    # Log-log scale
                    axes[1].loglog(range(1, len(s_recon) + 1), s_recon, 'm.-', linewidth=1, markersize=4)
                    axes[1].set_xlabel("Singular Value Index (log)")
                    axes[1].set_ylabel("Singular Value (log)")
                    axes[1].set_title("Reconstruction Matrix - SVD (Log-Log)")
                    axes[1].grid(True, alpha=0.3, which="both")
                    
                    plt.tight_layout()
                    svd_recon_path = os.path.join(calib_dir, "svd_reconstruction_matrix.png")
                    plt.savefig(svd_recon_path, dpi=150, bbox_inches="tight")
                    plt.close()
                    
                    saved_files["svd_reconstruction_plot"] = svd_recon_path
                    print(f"  ✅ Reconstruction matrix SVD plot: {svd_recon_path}")
                    
                    # Save SVD values as .npy
                    svd_recon_npy = os.path.join(calib_dir, "svd_reconstruction_values.npy")
                    np.save(svd_recon_npy, s_recon)
                    saved_files["svd_reconstruction_values"] = svd_recon_npy
        except Exception as e:
            print(f"  ❌ Error plotting reconstruction matrix SVD: {e}")
    
    # ========================================================================
    # Save configuration
    # ========================================================================
    config_csv = os.path.join(calib_dir, "calibration_config.csv")
    config_dict = {
        "Parameter": [
            "num_modes",
            "probe_amp_factor",
            "rcond",
            "f_number",
            "num_lenslets",
            "wavelength",
            "telescope_diameter",
            "gain",
            "leakage",
        ],
        "Value": [
            cfg.num_modes,
            cfg.probe_amp_factor,
            cfg.rcond,
            cfg.f_number,
            cfg.num_lenslets,
            cfg.wavelength,
            cfg.telescope_diameter,
            cfg.gain,
            cfg.leakage,
        ]
    }
    df_config = pd.DataFrame(config_dict)
    df_config.to_csv(config_csv, index=False)
    saved_files["config_csv"] = config_csv
    print(f"  ✅ Calibration config saved: {config_csv}")
    
    return saved_files


# ============================================================================
# CORRECTION MONITORING: BEFORE/AFTER WAVEFRONT IMAGES
# ============================================================================

def save_correction_comparison(
    wf_before: Any,
    wf_after: Any,
    output_dir: str,
    r0: float,
    fiber_diameter_mm: float = 9.0,
    fiber_length_km: float = 10.0,
    pib_before: float = None,
    pib_after: float = None,
) -> dict:
    """
    Save before/after focal plane wavefront images for correction comparison.
    
    Args:
        wf_before: Wavefront before correction (from layer_ao(state.wf_wfs))
        wf_after: Wavefront after correction (final focal plane)
        output_dir: Base output directory
        r0: Fried parameter value
        fiber_diameter_mm: Fiber diameter in mm (default 9.0)
        fiber_length_km: Fiber length in km (default 10.0)
        pib_before: Power in bucket before correction (optional)
        pib_after: Power in bucket after correction (optional)
    
    Returns:
        dict with paths to saved images
    """
    correction_dir = os.path.join(output_dir, "correction")
    os.makedirs(correction_dir, exist_ok=True)
    
    saved_files = {}
    
    # Get power maps
    power_before = np.asarray(wf_before.power)
    power_after = np.asarray(wf_after.power)
    
    # Reshape to 2D if necessary (WFS grid is 1D)
    if power_before.ndim == 1:
        size = int(np.sqrt(power_before.size))
        power_before = power_before.reshape((size, size))
    if power_after.ndim == 1:
        size = int(np.sqrt(power_after.size))
        power_after = power_after.reshape((size, size))
    
    # Normalize for comparison (LINEAR scale - no log)
    # Normalize to [0, 1] range for better visibility
    power_before_norm = power_before / power_before.max()
    power_after_norm = power_after / power_after.max()
    
    # Create figure: before / after comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Before correction (LINEAR scale)
    im1 = axes[0].imshow(power_before_norm, cmap='hot')
    axes[0].set_title("Before Correction (Turbulent)", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("X [pixels]")
    axes[0].set_ylabel("Y [pixels]")
    cbar1 = plt.colorbar(im1, ax=axes[0], label="Normalized Power")
    
    # Add text info on before image
    info_text_before = f"r₀ = {r0*1e3:.1f} mm\nFiber: {fiber_diameter_mm:.1f} mm"
    if pib_before is not None:
        info_text_before += f"\nPIB = {pib_before:.2e}"
    axes[0].text(0.02, 0.98, info_text_before, transform=axes[0].transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # After correction (LINEAR scale)
    im2 = axes[1].imshow(power_after_norm, cmap='hot')
    axes[1].set_title("After Correction (AO-corrected)", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("X [pixels]")
    axes[1].set_ylabel("Y [pixels]")
    cbar2 = plt.colorbar(im2, ax=axes[1], label="Normalized Power")
    
    # Add text info on after image
    info_text_after = f"Fiber Length: {fiber_length_km:.1f} km\nFiber: {fiber_diameter_mm:.1f} mm"
    if pib_after is not None:
        info_text_after += f"\nPIB = {pib_after:.2e}"
    axes[1].text(0.02, 0.98, info_text_after, transform=axes[1].transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    comparison_path = os.path.join(correction_dir, f"correction_before_after_r0_{r0*1e3:.3f}mm.png")
    plt.savefig(comparison_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    saved_files["comparison"] = comparison_path
    print(f"  ✅ Correction comparison image: {comparison_path}")
    
    # Create separate before image (LINEAR scale)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(power_before_norm, cmap='hot')
    ax.set_title("Before Correction (Turbulent)", fontsize=12, fontweight='bold')
    ax.set_xlabel("X [pixels]")
    ax.set_ylabel("Y [pixels]")
    
    # Add info box
    info_text = f"r₀ = {r0*1e3:.1f} mm\nFiber: {fiber_diameter_mm:.1f} mm\nFiber Length: {fiber_length_km:.1f} km"
    if pib_before is not None:
        info_text += f"\nPIB = {pib_before:.2e}"
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
           fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.colorbar(im, ax=ax, label="Normalized Power")
    plt.tight_layout()
    
    before_path = os.path.join(correction_dir, f"focal_before_r0_{r0*1e3:.3f}mm.png")
    plt.savefig(before_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    saved_files["before"] = before_path
    print(f"  ✅ Before correction image: {before_path}")
    
    # Create separate after image (LINEAR scale)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(power_after_norm, cmap='hot')
    ax.set_title("After Correction (AO-corrected)", fontsize=12, fontweight='bold')
    ax.set_xlabel("X [pixels]")
    ax.set_ylabel("Y [pixels]")
    
    # Add info box
    info_text = f"r₀ = {r0*1e3:.1f} mm\nFiber: {fiber_diameter_mm:.1f} mm\nFiber Length: {fiber_length_km:.1f} km"
    if pib_after is not None:
        info_text += f"\nPIB = {pib_after:.2e}"
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
           fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.colorbar(im, ax=ax, label="Normalized Power")
    plt.tight_layout()
    
    after_path = os.path.join(correction_dir, f"focal_after_r0_{r0*1e3:.3f}mm.png")
    plt.savefig(after_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    saved_files["after"] = after_path
    print(f"  ✅ After correction image: {after_path}")
    
    return saved_files


def visualize_shack_hartmann_reference(image_ref, num_lenslets, output_dir, wavelength_val):
    """
    Visualize Shack-Hartmann reference image with spot peaks and COG arrows.
    
    Args:
        image_ref: Reference WFS image (Field object)
        num_lenslets: Number of lenslets per side (e.g., 12 = 12x12 array)
        output_dir: Directory to save visualization
        wavelength_val: Wavelength value for labeling
    
    Returns:
        Path to saved visualization
    """
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        from scipy import ndimage
        
        # Extract image array
        image_array = np.asarray(image_ref)
        
        # Reshape if 1D
        if image_array.ndim == 1:
            size = int(np.sqrt(image_array.size))
            if size * size == image_array.size:
                image_array = image_array.reshape((size, size))
        
        # Normalize for display
        image_norm = image_array / (image_array.max() + 1e-10)
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        
        # ====================================================================
        # LEFT: Full WFS image
        # ====================================================================
        ax = axes[0]
        im = ax.imshow(image_norm, cmap='hot', origin='lower')
        ax.set_title("Shack-Hartmann Reference Image (Full)", fontsize=14, fontweight='bold')
        ax.set_xlabel("X [pixels]")
        ax.set_ylabel("Y [pixels]")
        plt.colorbar(im, ax=ax, label="Normalized Intensity")
        
        # ====================================================================
        # RIGHT: Spot analysis with peaks and COG arrows
        # ====================================================================
        ax = axes[1]
        im = ax.imshow(image_norm, cmap='hot', origin='lower', alpha=0.8)
        ax.set_title("Shack-Hartmann Spots with COG Arrows", fontsize=14, fontweight='bold')
        ax.set_xlabel("X [pixels]")
        ax.set_ylabel("Y [pixels]")
        
        # Find spots by local maxima
        coords = ndimage.maximum_filter(image_norm, size=5) == image_norm
        coords[image_norm < image_norm.max() * 0.3] = False  # Filter weak peaks - only keep top 30%
        
        # Get peak positions
        y_peaks, x_peaks = np.where(coords)
        
        if len(x_peaks) > 0:
            # Plot only the brightest peaks
            peak_values = image_norm[coords]
            bright_threshold = np.percentile(peak_values, 70)  # Only top 30% of peaks
            bright_coords = peak_values >= bright_threshold
            
            y_bright = y_peaks[bright_coords]
            x_bright = x_peaks[bright_coords]
            
            # Plot peak markers (only bright ones)
            ax.plot(x_bright, y_bright, 'r*', markersize=20, label=f'Peak Positions ({len(x_bright)} spots)', 
                   markeredgecolor='white', markeredgewidth=1.0)
            
            # Estimate COG for each subaperture and draw arrows
            num_lenslets = num_lenslets
            subaperture_size = image_array.shape[0] // num_lenslets
            
            arrow_count = 0
            for i in range(num_lenslets):
                for j in range(num_lenslets):
                    # Extract subaperture
                    y_start = i * subaperture_size
                    y_end = (i + 1) * subaperture_size
                    x_start = j * subaperture_size
                    x_end = (j + 1) * subaperture_size
                    
                    subap = image_norm[y_start:y_end, x_start:x_end]
                    
                    # Only process bright subapertures (top 40%)
                    if subap.max() > np.percentile(image_norm, 60):
                        # Compute COG within subaperture
                        yy, xx = np.meshgrid(np.arange(subap.shape[1]), np.arange(subap.shape[0]))
                        flux = subap.ravel()
                        total_flux = flux.sum()
                        
                        if total_flux > 0:
                            cog_x = np.sum(xx.ravel() * flux) / total_flux
                            cog_y = np.sum(yy.ravel() * flux) / total_flux
                            
                            # Center of subaperture
                            center_x = x_start + subaperture_size / 2
                            center_y = y_start + subaperture_size / 2
                            
                            # Draw arrow from center to COG (only if displacement is visible)
                            dx = cog_x - subaperture_size/2
                            dy = cog_y - subaperture_size/2
                            arrow_length = np.sqrt(dx**2 + dy**2)
                            
                            if arrow_length > 0.5:  # Only draw if COG is displaced by >0.5 pixels
                                arrow = plt.Arrow(
                                    center_x, center_y,
                                    dx, dy,
                                    width=3,
                                    color='cyan',
                                    alpha=0.8,
                                    edgecolor='white',
                                    linewidth=0.5
                                )
                                ax.add_patch(arrow)
                                arrow_count += 1
            
            ax.legend(loc='upper right', fontsize=11, framealpha=0.9)
            print(f"    📊 Visualized {arrow_count} subaperture COG arrows (out of {num_lenslets**2} total)")
        
        
        plt.tight_layout()
        
        # Save visualization
        sh_vis_path = os.path.join(output_dir, f"shack_hartmann_reference_wl_{wavelength_val*1e6:.2f}um.png")
        plt.savefig(sh_vis_path, dpi=150, bbox_inches="tight")
        plt.close()
        
        print(f"  ✅ Shack-Hartmann reference visualization: {sh_vis_path}")
        return sh_vis_path
        
    except Exception as e:
        print(f"  ❌ Error visualizing Shack-Hartmann reference: {e}")
        import traceback
        traceback.print_exc()
        return None