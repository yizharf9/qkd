import hcipy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
import os
import sys

# Make sure Python can find both this folder and the project root.
sys.path.append(os.path.dirname(__file__))  # for multylayer_utils
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  # adds oct_2025 to sys.path

import multylayer_utils as mlu
import main.params as p  # module alias
import main.utils as utils

utils.check_dir()
def main():
    # --- Grid and Propagator Setup (using params) ---
    focal_grid = hcipy.make_focal_grid(q=4, num_airy=5, spatial_resolution=p.wavelength / p.telescope_diameter)
    propagator = hcipy.FraunhoferPropagator(p.pupil_grid, focal_grid)

    # --- Initial Wavefront (using params) ---
    wf_in = hcipy.Wavefront(p.ap, p.wavelength)
    wf_in.total_power = 1

    # --- Atmospheric Site Definition ---
    # Using a single r0 value for this example. You can modify this as needed.
    r0 = 0.15 # Fried parameter in meters
    site_parameters = [
        {
            'altitude': 0,
            'Cn_squared': mlu.cn_squared_from_r0(r0, p.wavelength) * 0.8, # 80% turbulence at ground
            'outer_scale': p.L0,
            'velocity': 3
        },
        {
            'altitude': 10000,
            'Cn_squared': mlu.cn_squared_from_r0(r0, p.wavelength) * 0.2, # 20% at 10km
            'outer_scale': p.L0,
            'velocity': 3
        }
    ]

    # --- Build Atmosphere ---
    layers = mlu.build_site(p.pupil_grid, site_parameters, L0=p.L0)
    atmosphere = mlu.build_multilayer_screen(layers)

    # --- Animation Setup ---
    anim_duration = 5.0  # seconds
    fps = 10
    num_frames = int(anim_duration * fps)
    times = np.linspace(0, anim_duration, num_frames, endpoint=False)

    # --- Matplotlib Figure for Animation ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    ((ax_pupil_phase, ax_pupil_int), (ax_focal_phase, ax_focal_int)) = axes
    fig.tight_layout(pad=4.0)

    # --- Animation Writer ---
    writer = FFMpegWriter(fps=fps)
    output_filename = "multilayer_animation.mp4"

    # Colorbars (create once, update mappables each frame)
    cb_phase_pupil = cb_int_pupil = cb_phase_focal = cb_int_focal = None

    print("Starting animation rendering...")
    with writer.saving(fig, output_filename, dpi=150):
        for i, t in enumerate(times):
            print(f"Rendering frame {i+1}/{num_frames} at t={t:.2f}s")
            atmosphere.t = t
            wf_out = atmosphere(wf_in)
            img_out = propagator(wf_out)

            # --- Clear previous frames ---
            for ax in axes.flat:
                ax.clear()

            # 1. Pupil Plane Phase
            im1 = hcipy.imshow_field(wf_out.phase, ax=ax_pupil_phase, cmap='RdBu')
            ax_pupil_phase.set_title(f"Pupil Phase (t={t:.2f}s)")
            if cb_phase_pupil is None:
                cb_phase_pupil = fig.colorbar(im1, ax=ax_pupil_phase)
            else:
                cb_phase_pupil.update_normal(im1)

            # 2. Pupil Plane Intensity
            im2 = hcipy.imshow_field(wf_out.intensity, ax=ax_pupil_int, cmap='inferno')
            ax_pupil_int.set_title("Pupil Intensity")
            if cb_int_pupil is None:
                cb_int_pupil = fig.colorbar(im2, ax=ax_pupil_int)
            else:
                cb_int_pupil.update_normal(im2)

            # 3. Focal Plane Phase
            im3 = hcipy.imshow_field(img_out.phase, ax=ax_focal_phase, cmap='RdBu')
            ax_focal_phase.set_title("Focal Plane Phase")
            if cb_phase_focal is None:
                cb_phase_focal = fig.colorbar(im3, ax=ax_focal_phase)
            else:
                cb_phase_focal.update_normal(im3)

            # 4. Focal Plane Intensity (Log scale)
            im4 = hcipy.imshow_field(np.log10(img_out.intensity / img_out.intensity.max()), ax=ax_focal_int, vmin=-4, cmap='inferno')
            ax_focal_int.set_title("Focal Plane Intensity (log)")
            if cb_int_focal is None:
                cb_int_focal = fig.colorbar(im4, ax=ax_focal_int)
            else:
                cb_int_focal.update_normal(im4)

            writer.grab_frame()

    plt.close(fig)
    print(f"\\nAnimation saved to '{output_filename}'")


if __name__ == "__main__":
    main()
