
path_file="./massive_output.csv"
#wavelength,r0_ref,run_number,power_in_bucket_before_turbulance,total_power_before_turbulance,precentage_before_turbulance,power_in_bucket_after_turbulance,total_power_after_turbulance,precentage_after_turbulance,time
# run this to plot the massive_output.csv file
import pandas as pd
import matplotlib.pyplot as plt
import os
from hcipy import *
import numpy as np
import utils
from math import e
from datetime import datetime,date
from utils import *
def check_axis(run_Cn2: bool,r0_col) -> str:
    if run_Cn2:
        pass # switch to Cn2 for x axis
    else:
        r0_col=r0_col
    return r0_col
    
    

#-------prams
To_run=True
after=True
run_Cn2=False
save_images=True

#_-----prams for calculations
lam_ref = 500e-9  
SR_SHORT_COEFF = 0.134
D=8
#--- 0) Ensure correct working directory ---
utils.check_dir()
# --- 1) Load CSV ---
[df,structure]=utils.Load_csv(path_file)


#print(structure)
#-------pick-------------
r0_col  =utils.pick(df.columns, "r0", "r_0", "r0_ref", "r_0_ref","r0_ref")    
PIB_col = utils.pick(df.columns,"total_power_after_turbulance")
wl_col  = utils.pick(df.columns, "wavelength", "lambda", "lam", "wl")
conservation_of_energy=utils.pick(df.columns,"conservation of energy[%]")
focal_dim_col=utils.pick(df.columns, 'focal_dim', 'focsl_dim', 'focsl', 'focal', 'focal_dim')
#df = df[df[focal_dim_col]==1]

if not (r0_col and wl_col and focal_dim_col ):
    raise ValueError(f"Need columns for r0, smf, wavelength. Found: {list(df.columns)}")
    # --- 4) Clean types ---

df[r0_col]  = pd.to_numeric(df[r0_col], errors="coerce")
df[r0_col] = np.where(df[r0_col] == 1000000000, 1, df[r0_col])
df['actual_r0_col'] = df[r0_col] * (df[wl_col] / lam_ref) ** (6.0 / 5.0)    
df[PIB_col] = pd.to_numeric(df[PIB_col], errors="coerce")
df[wl_col]  = pd.to_numeric(df[wl_col], errors="coerce")
df = df.dropna(subset=['actual_r0_col', wl_col])
df["Cn2"]= Cn_squared_from_fried_parameter(df[r0_col],df[wl_col]).to_numpy()
df["Cn2"] =pd.to_numeric(df["Cn2"], errors="coerce")
df['conservation_of_energy[%]'] = pd.to_numeric(df['conservation_of_energy[%]'], errors='coerce')




    # 1) Group by settings and compute stats of the measured value (SMF)
x_axis=check_axis(run_Cn2,r0_col='actual_r0_col')
print("x_axis:",x_axis)
group_cols = [wl_col, x_axis,focal_dim_col]          # settings that define a case
val_col    = PIB_col                   # the measured metric to summarize
if To_run:
    # Consistent styling across calls (optional)
    wavelengths_value = sorted(df[wl_col].unique())
    focal_value = sorted(df[focal_dim_col].unique())
    #print("wavelengths:", wavelengths_value)
    #print("focal values:", focal_value)
    # Subplot A: lin–lin
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    plot_mean_line_loglin(
        df, wl_col, focal_dim_col,wavelengths_value,focal_value,x_axis, PIB_col,axes[0],
        title="PIB after vs r0 (log-lin)", x_label="r0_ref [m]", y_label="Mean PIB")
    plot_mean_line_loglin(
        df, wl_col, focal_dim_col,wavelengths_value,focal_value,x_axis,'conservation_of_energy[%]', axes[1],
        title="Conservation of energy vs r0 (log-lin)", x_label="r0_ref [m]", y_label="Conservation of energy [%]")
    plt.tight_layout()
    print("done plotting")
    print("Date: ",datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
plt.tight_layout()

if save_images:
    base_output_dir = "./plots"
    os.makedirs(base_output_dir, exist_ok=True)
    fname = f"PIB_and_energy_conservation_vs_{'Cn2' if run_Cn2 else 'r0'}.png"
    out_path = os.path.join(base_output_dir, fname)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved image to: {out_path}")

plt.show()