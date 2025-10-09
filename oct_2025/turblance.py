from hcipy import *
from tqdm.notebook import tqdm

import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
import time
import os
num_modes = 500
telescope_diameter = 8. # meter
central_obscuration = 1.2 # meter
central_obscuration_ratio = central_obscuration / telescope_diameter
spider_width = 0.05 # meter
oversizing_factor = 16 / 15
num_pupil_pixels = 240 * oversizing_factor
pupil_grid_diameter = telescope_diameter * oversizing_factor
pupil_grid = make_pupil_grid(num_pupil_pixels, pupil_grid_diameter)

VLT_aperture_generator = make_obstructed_circular_aperture(telescope_diameter,
    central_obscuration_ratio, num_spiders=4, spider_width=spider_width)

VLT_aperture = evaluate_supersampled(VLT_aperture_generator, pupil_grid, 4)

dm_modes = make_disk_harmonic_basis(pupil_grid, num_modes, telescope_diameter, 'neumann')
dm_modes = ModeBasis([mode / np.ptp(mode) for mode in dm_modes], pupil_grid)

wavelength_wfs = 0.7e-6
wavelength_sci = 2.2e-6
deformable_mirror = DeformableMirror(dm_modes)

#wf = Wavefront(VLT_aperture, wavelength_sci)
#wf.total_power = 1

# Put actuators at random values, putting a little more power in low-order modes
deformable_mirror.actuators = np.random.randn(num_modes) / (np.arange(num_modes) + 10)

# Normalize the DM surface so that we get a reasonable surface RMS.
deformable_mirror.actuators *= 0.3 * wavelength_sci / np.std(deformable_mirror.surface)

imshow_field(deformable_mirror.phase_for(wavelength_wfs), mask=VLT_aperture, cmap='RdBu')
plt.colorbar()
plt.show()