import os
import datetime as dt
from hcipy import *
"""
This file contains all the parameters for the project.
"""


#------setting parameters for AO.py-----------

#-------massive_simulation.py-----------
m="\n"
m+="Starting massive simution runs..."
m+="Current working directory:"+str(os.getcwd())
m+="Script start time:"+str(dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
massive_simulation_begin_massage=m
#--------AO PARAMS.py-----------
TurbulencLayer = True          # keep your original flag name
wavelength=1.55e-6            # [m]
r0_ref_list=[0.01,0.05,0.03,0.07,0.2]
q_AO=4
num_modes = 500
telescope_diameter=0.3
N=240
oversz=16.0/15.0
oversz*N
spider_w = 0.005         # [m]
D_obs =0.1       # [m]
eps = D_obs / telescope_diameter # central obscuration ratio
pupil_grid = make_pupil_grid(N*oversz, telescope_diameter * oversz)
ap_gen = make_obstructed_circular_aperture(telescope_diameter, eps, num_spiders=4, spider_width=spider_w)
ap = evaluate_supersampled(ap_gen, pupil_grid, 4) 
wf = Wavefront(ap, wavelength)
wf.total_power = 1
#_--------
spatial_res = wavelength / telescope_diameter  # [rad] per λ/D

# atmosphere params -----
L0 = 40.0          # [m]
tau0 = 5e-3        # [s]
lam_ref = 500e-9   # [m]
pupil_grid = make_pupil_grid(N*oversz, telescope_diameter * oversz)
ap_gen = make_obstructed_circular_aperture(telescope_diameter, eps, num_spiders=4, spider_width=spider_w)
ap = evaluate_supersampled(ap_gen, pupil_grid, 4) 
ref_wavelength=8e-7
focal_dim=9e-6
Fnum_sci = 50.0                 # adjust to your optics if needed 
f_eff = Fnum_sci * telescope_diameter           # [m] effective focal length
diam_phys = 9e-6 
rad_phys  = diam_phys / 2.0     # [m]               # [m]
alpha = rad_phys / f_eff        # [rad] angular radius on focal plane

# Adaptive Optics params --------------
f_number = 50
num_lenslets = 40
sh_diameter = 5e-3  # [m] SH beam diameter
wf_wfs = Wavefront(ap,1.55e-6)
zero_magnitude_flux = 3.9e10 #3.9e10 photon/s for a mag 0 star
stellar_magnitude = 5
delta_t = 1e-3 # sec, so a loop speed of 1kHz.

wf_wfs = Wavefront(ap, wavelength)
#wf_sci = Wavefront(ap, wavelength_sci)

wf_wfs.total_power = zero_magnitude_flux * 10**(-stellar_magnitude / 2.5)
norm=float(wf_wfs.total_power)
