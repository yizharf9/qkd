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
#-------prams
To_run=True
after=True
path_file="./massive_output.csv"
#_-----prams for calculations
lam_ref = 500e-9  
SR_SHORT_COEFF = 0.134
D=8
#--- 0) Ensure correct working directory ---
utils.check_dir()
# --- 1) Load CSV ---
[df,structure]=utils.Load_csv(path_file)
print(structure)


# --- 3) Locate columns ---
#df=df[0:2200]
#df=df.sorted()
 # for after , for before set to False

def proces_file(df):
    proces_values={}
    print(f"Loaded {len(df)} rows from {path_file}")
    print(df.head)
    r0_col  =utils.pick(df.columns, "r0", "r_0", "r0_ref", "r_0_ref","r0_ref")
    smf_col_after = utils.pick(df.columns,"power_in_bucket_after_turbulance")
    smf_col_before = utils.pick(df.columns,"power_in_bucket_before_turbulance")
    wl_col  = utils.pick(df.columns, "wavelength", "lambda", "lam", "wl")
    conservation_of_energy=utils.pick(df.columns,"conservation of energy[%]")
    print(r0_col)
    print("#"*50)
    print(f"Loaded {len(r0_col)} rows from {path_file}")
    if after:
        smf_col=smf_col_after
    else:
        smf_col_before
    if not (r0_col and wl_col ):
        raise ValueError(f"Need columns for r0, smf, wavelength. Found: {list(df.columns)}")
    # --- 4) Clean types ---
    df[r0_col]  = pd.to_numeric(df[r0_col], errors="coerce")
    df['actual_r0_col'] = df[r0_col] * (df[wl_col] / lam_ref) ** (6.0 / 5.0)    
    df[smf_col] = pd.to_numeric(df[smf_col], errors="coerce")
    df[wl_col]  = pd.to_numeric(df[wl_col], errors="coerce")
    df = df.dropna(subset=['actual_r0_col', wl_col])
    df["Cn2"]= Cn_squared_from_fried_parameter(df[r0_col],df[wl_col]).to_numpy()
    df["Cn2"] =pd.to_numeric(df["Cn2"], errors="coerce")
    df[conservation_of_energy]=pd.to_numeric(df[conservation_of_energy],errors="coerce")
    # 1) Group by settings and compute stats of the measured value (SMF)
    
    group_cols = [wl_col, r0_col,"Cn2"]          # settings that define a case
    val_col    = smf_col                   # the measured metric to summarize


    stats = (
        df.groupby(group_cols, dropna=False)[val_col]
        .agg(count='count', mean='mean', min='min', max='max')
        .reset_index()
        .sort_values(group_cols)
    )
    print("data for display:(Cn2)")
    print("#"*50)
    print(df["Cn2"])
    proces_values["stats"]=stats
    proces_values["wl_col"]=wl_col
    proces_values["val_col"]=val_col
    proces_values["df"]=df
    proces_values["r0_col"]=r0_col
    proces_values["COF"]=conservation_of_energy
    return proces_values
def check_axis(run_Cn2: bool,r0_col) -> str:
    if run_Cn2:
        r0_col="Cn2" # switch to Cn2 for x axis
    return r0_col
    
if To_run:
    proces_values={}
    proces_values=proces_file(df)
    r0_col=proces_values["r0_col"]
    x_axis=check_axis(run_Cn2=False,r0_col=r0_col)
    print(df[r0_col][0])
    stats,wl_col,val_col,df=[0]*4
    stats,wl_col,val_col,df,cof=proces_values["stats"],proces_values["wl_col"],\
    proces_values["val_col"],proces_values["df"],proces_values["COF"]
    params=[stats,wl_col,val_col,df,x_axis,after,r0_col]
    #rest_of_code(stats,wl_col,val_col,df,x_axis,after,r0_col)
    r0=0.1
    r_ref=0.1* (1.55e-6/lam_ref) ** (6.0 / 5.0)   
    D_over_r0 = D /r_ref
    print("r_ref:",r_ref)

    """
    1.55e-06,0.1,0,0.5732063905798354,48.01889078999652,0.011937101860321354,0.0133460713186899,47.96155491325191,0.00027826602667134015,99.36935463098774,2025-11-04 18:44:29
    """
    print("SR: ",100*np.exp(-SR_SHORT_COEFF*((D_over_r0)** (5.0 / 3.0))))
    SR_theoretical = np.exp(-SR_SHORT_COEFF * (D_over_r0) ** (5.0 / 3.0)) 
    utils.plot_dots_mean_by_scale(df,stats,wl_col,x_axis=x_axis,y_col=val_col,x_mode="log",y_mode="log",title="Phase and amplitude at different stages",show=False)
    utils.plot_dots_mean_by_scale(df,stats,wl_col,x_axis=x_axis,y_col=cof,x_mode="lin",y_mode="lin",title="Percentage of energy conservation",show=False)
    utils.plot_simulated_vs_theoretical_sr(df,stats,wl_col,x_axis,val_col,"Percentage of energy conservation with rispect to SR",show=True)