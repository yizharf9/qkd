
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
import params
def check_axis(run_Cn2: bool,r0_col) -> str:
    if run_Cn2:
        pass # switch to Cn2 for x axis
    else:
        r0_col=r0_col
    return r0_col
    
df=pd.read_csv(path_file)
df1=df[1:91]
print(len(df1))
def plot_E_and_D():
        # --- Load data ---
    df = pd.read_csv("./AO_simulation_log.csv")
    norm=params.norm
    # --- Columns ---
    t_col   = "timestep"
    E_col   = "E_power_sum"
    D_col   = "D_power_sum"
    r0_col  = "r0_ref"
    airy_col = "num_airy"
    # --- Compute new columns ---
    df["E_norm"] = df[E_col] / norm
    df["D_norm"] = df[D_col] / norm  # ← תיקון: בעבר כתבת פעמיים E_col
    # --- Prepare figure ---
    plt.figure(figsize=(10,6))
    # --- unique Airy groups ---
    airy_groups = sorted(df[airy_col].unique())

    # colormap
    colors = plt.cm.viridis(np.linspace(0,1,len(airy_groups)))

    for color, airy in zip(colors, airy_groups):

        sub = df[df[airy_col] == airy]

        # group by r0 and compute mean/std
        grouped = sub.groupby(r0_col)

        r0_vals = grouped[r0_col].mean().values
        E_mean  = grouped["E_norm"].mean().values
        E_std   = grouped["E_norm"].std().values
        D_mean  = grouped["D_norm"].mean().values
        D_std   = grouped["D_norm"].std().values
        frac_01=D_mean-E_mean
        fn_01=frac_01
        print(fn_01,airy)
        plt.clf
        Fig_E_and_D=plt.figure
        # E_norm – dashed
        plt.errorbar(r0_vals, E_mean, yerr=E_std, 
                    fmt='--o', color=color, label=f"without correction (airy={airy} $\\frac{{\\lambda}}{{D}}$)")

        # D_norm – solid
        plt.errorbar(r0_vals, D_mean, yerr=D_std, 
                    fmt='-s', color=color, label=f"with correction (airy={airy} $\\frac{{\\lambda}}{{D}}$)")    
    plt.xlabel("r0_ref[m]")
    plt.ylabel("Normalized Power")
    plt.title("without correction[E] and with correction[D] vs r0_ref for different num_airy")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    base_output_dir = 'Plots'
    os.makedirs(base_output_dir, exist_ok=True)
    out_path = os.path.join(
    base_output_dir,
    f"without correction[E] and with correction[D] vs r0_ref for different num_airy.png"
    )
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"✅ Saved combined figure to: {out_path}")
    plt.close()    
    plt.show()
def plot_pib_vs_r0_for_column(
    df,
    y_col,
    ax,
    x_col="r0_ref",
    group_col="num_airy",
    run_col="run_number",
    x_label="r0_ref[m]",
    y_label=None,
    title="",
    scatter_alpha=0.4,
    line_width=2.0,
):
    """
    Plot all points of y_col vs x_col, colored by group_col (num_airy),
    and add mean trend line per (group_col, x_col) over run_number.
    """
    if y_label is None:
        y_label = y_col

    # Loop over each num_airy and plot
    for num_airy_val, group in df.groupby(group_col):
        # Scatter: all raw points
        ax.scatter(
            group[x_col],
            group[y_col],
            alpha=scatter_alpha,
            label=f"{group_col}={num_airy_val}",
        )

        # Mean trend: mean over run_number for each r0_ref (per num_airy)
        means = (
            group
            .groupby(x_col)[y_col]
            .mean()
            .reset_index()
            .sort_values(by=x_col)
        )

        ax.plot(
            means[x_col],
            means[y_col],
            linewidth=line_width,
        )

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True)
    ax.legend()
def plot_before_after_from_out_csv(
    csv_path="./out.csv",
    x_col="r0_ref",
    y_before_col="power_in_bucket_before_turbulance",
    y_after_col="power_in_bucket_after_turbulance",
    group_col="num_airy",
    run_col="run_number",
    # labels & titles (you can change these when you call the function)
    x_label="r0_ref",
    y_before_label="Power in bucket (before turbulence)[w]",
    y_after_label="Power in bucket (after turbulence)[w]",
    title_before="Power in bucket vs r0_ref (before turbulence)",
    title_after="Power in bucket vs r0_ref (after turbulence)",
    figsize=(14, 6),
):
    """
    Create a figure with two subplots:
    1) y_before_col vs x_col
    2) y_after_col vs x_col
    Both colored by num_airy with mean trend lines.
    """
    # Load CSV
    df = pd.read_csv(csv_path)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Figure 1 – before turbulence
    plot_pib_vs_r0_for_column(
        df1,
        y_col=y_before_col,
        ax=axes[0],
        x_col=x_col,
        group_col=group_col,
        run_col=run_col,
        x_label=x_label,
        y_label=y_before_label,
        title=title_before,
    )

    # Figure 2 – after turbulence
    plot_pib_vs_r0_for_column(
        df1,
        y_col=y_after_col,
        ax=axes[1],
        x_col=x_col,
        group_col=group_col,
        run_col=run_col,
        x_label=x_label,
        y_label=y_after_label,
        title=title_after,
    )

    fig.tight_layout()
    base_output_dir = 'Plots'
    os.makedirs(base_output_dir, exist_ok=True)
    out_path = os.path.join(
    base_output_dir,
    f"{title_after}"
    )
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"✅ Saved combined figure to: {out_path}")

    return fig, axes
if __name__ == "__main__":
    # מריץ את הפונקציה על הקובץ הקבוע
    fig, axes = plot_before_after_from_out_csv(
        csv_path=path_file,
        x_label="r0_ref",
        y_before_label="Power in bucket (before turbulence)",
        y_after_label="Power in bucket (after turbulence)",
        title_before="Power in bucket vs r0_ref (before turbulence)",
        title_after="Power in bucket vs r0_ref (after turbulence)",
    )
    Fig_E_and_D=plot_E_and_D()
 

