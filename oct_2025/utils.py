import os
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
from tqdm.notebook import tqdm
import scipy.ndimage as ndimage

def check_dir():
    print("Checking current working directory...")
    current_directory_name = os.path.basename(os.getcwd())
    if current_directory_name == 'oct_2025':
        print(f"✅ Success: Script is running from the correct directory ('{current_directory_name}').")
    else:
        print(f"⚠️ Warning: Script is NOT running from the 'oct_2025' directory.")
        print(f"   Current directory is: '{current_directory_name}'")
        exit("Execution stopped. Please run the script from the 'oct_2025' directory.")
    print("-" * 60) # Visual separator

def update_csv( wavelength,
                r0_ref_val,
                run_num,
                focal_dim,
                power_in_bucket_before,
                total_power_before,
                precentage_before,
                power_in_bucket_after,
                total_power_after,
                precentage_after,
                conservation_of_energy,
                ):

#head= wavelength ,r0_ref,run_number,focsl_dim,power_in_bucket_before_turbulance,total_power_before_turbulance,precentage_before_turbulance,power_in_bucket_after_turbulance,total_power_after_turbulance,precentage_after_turbulance,conservation_of_energy[%],time

    file_path = "./massive_output.csv"
    columns = [
        "wavelength",
        "r0_ref",
        "run_number",
        "focsl_dim",
        "power_in_bucket_before_turbulance",
        "total_power_before_turbulance",
        "precentage_before_turbulance",
        "power_in_bucket_after_turbulance",
        "total_power_after_turbulance",
        "precentage_after_turbulance",
        "conservation_of_energy[%]",
        "time"
        ]

    new_row_df = pd.DataFrame([{
        "wavelength": wavelength,
        "r0_ref": r0_ref_val,
        "run_number": run_num,
        "focal_dim":focal_dim,
        "power_in_bucket_before_turbulance": power_in_bucket_before,
        "total_power_before_turbulance": total_power_before,
        "precentage_before_turbulance": precentage_before,
        
        "power_in_bucket_after_turbulance": power_in_bucket_after,
        "total_power_after_turbulance": total_power_after,
        "precentage_after_turbulance": precentage_after,
        "conservation_of_energy[%]" : conservation_of_energy,
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

def add_noise_to_wavefront(
        propagated_wavefront,
        telescope_diameter,
        SNR = 5,
        stellar_magnitude = 8.0,
        flux_zero_point = 1.5e10 ,
        throughput = 0.8,
        exposure_time = 0.01  ,
    ):
    focal_grid = propagated_wavefront.grid
    collecting_area = np.pi * (telescope_diameter/2)**2

    # Total photons/sec from the star entering the telescope
    photon_flux = flux_zero_point * 10**(-0.4 * stellar_magnitude) * collecting_area * throughput

    # Scale it to physical units (Photons / second)
    image_photons_per_sec = propagated_wavefront.power * photon_flux * SNR

    detector = NoisyDetector(focal_grid)

    # Configure Noise Properties
    detector.include_photon_noise = True
    detector.read_noise = 5.0           # rms electrons
    detector.dark_current_rate = 0.1     # electrons/sec/pixel (usually low for IR)
    # detector.flat_field = 0.05         # Optional: 5% pixel-to-pixel sensitivity variation

    detector.integrate(image_photons_per_sec, dt=exposure_time)

    image_noisy = detector.read_out()

    wf_noisy = Wavefront(image_noisy)
    return wf_noisy

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

def check_energy_conservation(
    wf1,
    Wf_in_focal
    ):
    print("\n-- Checking Energy conservation between pupil and focal --")
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
    print(f"Energy_conv [%]: {Energy_conv}")
    return Energy_conv

# helper לציור PSF
def plot_psf_on(fig,ax, psf,alpha,f_m,extent_focal_mm,scale_mm, title = "PSF plot"):
    psf_img = np.log10((psf / psf.max()).shaped + 1e-12)
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
