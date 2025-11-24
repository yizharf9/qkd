import os
import datetime as dt
from hcipy import *
"""
This file contains all the parameters for the project.
"""


#------setting parameters for OA.py-----------
TurbulencLayer = True          # keep your original flag name
#-------massive_simulation.py-----------
m="\n"
m+="Starting massive simution runs..."
m+="Current working directory:"+str(os.getcwd())
m+="Script start time:"+str(dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
massive_simulation_begin_massage=m
#--------AO PARAMS.py-----------
wavelength_sci = 1.55e-6       # [m] science wavelength (your main one)
wavelength_wfs = 8e-7          # [m] WFS wavelength (AO channel)

D = 8.0                 # [m]
D_obs = 1.2             # [m]
eps = D_obs / D # central obscuration ratio
spider_w = 0.05         # [m]
oversz = 16.0/15.0
N = int(240 * oversz) #256 # number of pixels across pupil diameter
N=512

# focal grid params --------
q = 8
num_airy = 60

# atmosphere params -----
L0 = 40.0          # [m]
tau0 = 5e-3        # [s]
lam_ref = 500e-9   # [m]


# atmosphere params -----
ref_wavelength=8e-7
focal_dim=9e-6
Fnum_sci = 50.0                 # adjust to your optics if needed 
f_eff = Fnum_sci * D            # [m] effective focal length
diam_phys = 9e-6 
rad_phys  = diam_phys / 2.0     # [m]               # [m]
alpha = rad_phys / f_eff        # [rad] angular radius on focal plane

# Adaptive Optics params --------------
f_number = 50
num_lenslets = 40
sh_diameter = 5e-3  # [m] SH beam diameter
