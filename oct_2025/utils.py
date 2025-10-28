import os
import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import animation
from IPython.display import HTML
from mpl_toolkits.axes_grid1 import make_axes_locatable
try:
    from hcipy import *
except Exception as e:
    raise ImportError("HCIPy is required. Install it with: pip install hcipy") from e

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
                power_in_bucket_before,
                total_power_before,
                precentage_before,
                power_in_bucket_after,
                total_power_after,
                precentage_after
                ):

    file_path = "./massive_output.csv"
    columns = [
        "wavelength",
        "r0_ref",
        "run_number",
        "power_in_bucket_before_turbulance",
        "total_power_before_turbulance",
        "precentage_before_turbulance",
        "power_in_bucket_after_turbulance",
        "total_power_after_turbulance",
        "precentage_after_turbulance",
        "time"
        ]

    new_row_df = pd.DataFrame([{
        "wavelength": wavelength,
        "r0_ref": r0_ref_val,
        "run_number": run_num,
        
        "power_in_bucket_before_turbulance": power_in_bucket_before,
        "total_power_before_turbulance": total_power_before,
        "precentage_before_turbulance": precentage_before,
        
        "power_in_bucket_after_turbulance": power_in_bucket_after,
        "total_power_after_turbulance": total_power_after,
        "precentage_after_turbulance": precentage_after,
        
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
