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
#from mpl_toolkits.axes_grid1 import make_axes_locatable
#from hcipy import *
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

def _norm_scale_arg(mode: str) -> str:
    """Normalize scale string to 'lin' or 'log'."""
    if mode is None:
        return 'lin'
    m = str(mode).strip().lower()
    return 'log' if m in {'log', 'log10', 'logarithmic'} else 'lin'

def _pretty_x_units(x_axis: str) -> str:
    """Optional: add units to x label if known."""
    xl = x_axis.lower()
    if xl == 'cn2' or 'cn2' in xl:
        return " [m^(-2/3)]"
    if 'r0' in xl or 'r_0' in xl:
        return " [m]"
    return ""

def plot_dots_mean_by_scale(
    df, stats,
    wl_col: str,
    x_axis: str,
    y_col: str,
    *,
    x_mode='lin',          # 'lin' or 'log'
    y_mode='lin',          # 'lin' or 'log'
    after: bool = True,    # only for filename tag
    outdir: str = "plots",
    title: str = None,
    show: bool = False
):
    """
    Draws 'dots (rows) + mean line per wavelength' with selectable x/y scales.

    Parameters
    ----------
    df : DataFrame
        Row-level data containing columns wl_col, x_axis, y_col.
    stats : DataFrame
        Grouped stats with columns wl_col, x_axis and ['mean','min','max','count'] for y_col.
    wl_col : str
        Column name of wavelength.
    x_axis : str
        X variable column name (e.g., 'r0' or 'cn2').
    y_col : str
        Y variable column name (e.g., 'power_in_bucket_after_turbulance').
    x_mode, y_mode : {'lin','log'}
        Axis scaling modes (case-insensitive, accepts 'log10' etc. → 'log').
    after : bool
        Used only to tag the output filename.
    outdir : str
        Directory to save figure into.
    title : str or None
        Custom title; if None, a default is composed.
    show : bool
        If True, plt.show() after saving (can block on some systems).
    """
    x_mode = _norm_scale_arg(x_mode)
    y_mode = _norm_scale_arg(y_mode)

    # If any log scale, restrict to positive domain
    work_df = df.copy()
    work_stats = stats.copy()
    if x_mode == 'log':
        work_df = work_df[work_df[x_axis] > 0]
        work_stats = work_stats[work_stats[x_axis] > 0]
    if y_mode == 'log':
        work_df = work_df[work_df[y_col] > 0]
        # ensure mean/min/max positive as well
        for c in ['mean', 'min', 'max']:
            if c in work_stats:
                work_stats = work_stats[work_stats[c] > 0]

    if work_df.empty or work_stats.empty:
        raise ValueError("No data to plot after applying scale/domain filters. "
                         f"(x_mode={x_mode}, y_mode={y_mode})")

    # Color per wavelength
    unique_wls = sorted(work_df[wl_col].dropna().unique())
    color_cycle = itertools.cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])
    wl_color = {wl: next(color_cycle) for wl in unique_wls}

    fig, ax = plt.subplots(figsize=(9, 6))

    # Scatter points per λ
    for wl, d in work_df.groupby(wl_col, sort=True):
        ax.scatter(d[x_axis], d[y_col],
                   s=14, alpha=0.55, linewidths=0,
                   color=wl_color.get(wl), label=f"λ = {wl:g} (rows)")

    # Mean lines per λ
    for wl, d in work_stats.groupby(wl_col, sort=True):
        d = d.sort_values(x_axis)
        ax.plot(d[x_axis], d['mean'],
                linewidth=2.2, color=wl_color.get(wl),
                label=f"λ = {wl:g} (mean)")

    # Scales
    if x_mode == 'log':
        ax.set_xscale('log')
        ax.xaxis.set_major_locator(LogLocator(base=10.0))
        ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))
        ax.xaxis.set_minor_formatter(NullFormatter())
    if y_mode == 'log':
        ax.set_yscale('log')
        ax.yaxis.set_major_locator(LogLocator(base=10.0))
        ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))
        ax.yaxis.set_minor_formatter(NullFormatter())

    # Labels & title
    x_unit = _pretty_x_units(x_axis)
    x_lab = f"{x_axis}{x_unit}" + (" (log)" if x_mode == 'log' else "")
    y_lab = f"{y_col}" + (" (log)" if y_mode == 'log' else "")
    ax.set_xlabel(x_lab)
    ax.set_ylabel(y_lab)
    if title is None:
        title = f"{y_col} vs {x_axis} — dots + mean ({x_mode}/{y_mode})"
    ax.set_title(title)

    ax.grid(True, which='both', alpha=0.3)
    ax.legend(ncol=2, frameon=True)
    fig.tight_layout()

    # Save
    os.makedirs(outdir, exist_ok=True)
    fname = f"{y_col}_vs_{x_axis}_{'after' if after else 'before'}_{x_mode}-{y_mode}_{int(time.time())}.png"
    save_path = os.path.join(outdir, fname)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"✅ Saved plot to: {save_path}")
    if show:
        plt.show()
    plt.close(fig)

    return save_path
def time_asstimate(current_time,begin_time,current_run,num_of_run): 
    """
    return the number in sec until the code is finish 
    assume linear progression 
    """
    runtime=current_time-begin_time #time the code is running
    avg_TFR=runtime/current_run #avg time for run
    return avg_TFR*num_of_run-runtime

