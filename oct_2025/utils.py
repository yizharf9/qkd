import os
import itertools
import numpy as np
import pandas as pd
from matplotlib.ticker import LogLocator, NullFormatter, ScalarFormatter
import datetime
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import animation
from IPython.display import HTML
from mpl_toolkits.axes_grid1 import make_axes_locatable
from hcipy import *
try:
    from hcipy import *
except Exception as e:
    raise ImportError("HCIPy is required. Install it with: pip install hcipy") from e
import time
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
        "focsl_dim"
        "power_in_bucket_before_turbulance",
        "total_power_before_turbulance",
        "precentage_before_turbulance",
        "power_in_bucket_after_turbulance",
        "total_power_after_turbulance",
        "precentage_after_turbulance",
        "conservation_of_energy[%]"
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
        "conservation of energy[%]" : conservation_of_energy,
        "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }])

    if os.path.exists(file_path):
        new_row_df.to_csv(file_path, mode='a', header=False, index=False)
    else:
        new_row_df.to_csv(file_path, mode='w', header=columns, index=False)
    print("Operation complete. Data has been saved.")

def add_noise_to_psf(
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

def time_asstimate(current_time,begin_time,current_run,num_of_run): 
    """
    return the number in sec until the code is finish 
    assume linear progression 
    """
    runtime=current_time-begin_time #time the code is running
    avg_TFR=runtime/current_run #avg time for run
    return avg_TFR*num_of_run-runtime
#-----plots----------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

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
