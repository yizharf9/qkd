
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
import oct_2025.main.params as params

run_as_AO=True #!run PIB VS r0_ref from AO
if run_as_AO:
    path_file="./AO_simulation_log.csv"
else:
    path_file="./massive_output.csv"
def check_axis(run_Cn2: bool,r0_col) -> str:
    if run_Cn2:
        pass # switch to Cn2 for x axis
    else:
        r0_col=r0_col
    return r0_col
def plot_E_and_D(csv_path: str = "./AO_simulation_log.csv"):
    """
    Plot normalized E (without correction) and D (with correction) on
    a single figure. Style is similar to the original version,
    but with a logarithmic y-axis for better separation.
    """
    # --- Load and prepare data ---
    df = pd.read_csv(csv_path)
    norm = params.norm

    E_col = "E_power_sum"
    D_col = "D_power_sum"
    r0_col = "r0_ref"
    airy_col = "num_airy"
    timestep="timestep"
    # Keep only relevant rows and normalize once.
    df = df[df[airy_col] > 1].copy()
    df = df[df[airy_col].isin([2, 3, 4])]
    
    r0_drop = [
        0.01, 0.05, 0.03, 0.07, 0.2, 0.02,0.014
    ]
    df=df[df[r0_col].isin(r0_drop)]
    #df = df[df[r0_col].isin(r0_drop)]
    # Optional per-point exclusions: list of (r0_ref, num_airy)
    drop_points = []

    if drop_points:
        mask = ~df.apply(lambda row: (row[r0_col], row[airy_col]) in drop_points, axis=1)
        df = df[mask]
    df = df[df[timestep] == 780]
    df = df[df[r0_col] < 0.075]  # keep r0 < 0.075 m
    df["E_norm"] = df[E_col] / norm
    df["D_norm"] = df[D_col] / norm
    airy_groups = sorted(df[airy_col].unique())
    # Use a smooth scientific colormap (logical progression in Airy size).
    dark_colors = [
    "#000000",  # black
    "#303030",  # dark gray
    "#4a1486",  # very dark purple
    "#283593",  # dark indigo / blue
    "#00695c",  # dark teal
    "#b71c1c",  # dark red (still not bright)
    ]
    colors = [dark_colors[i % len(dark_colors)] for i in range(len(airy_groups))]
    fig, ax = plt.subplots(figsize=(10, 6))

    for color, airy in zip(colors, airy_groups):
        sub = df[df[airy_col] == airy]
        grouped = sub.groupby(r0_col)

        r0_vals = grouped[r0_col].mean().values
        E_mean = grouped["E_norm"].mean().values
        E_std = grouped["E_norm"].std().values
        D_mean = grouped["D_norm"].mean().values
        D_std = grouped["D_norm"].std().values

        # convert to cm for plotting
        r0_vals_cm = r0_vals * 100

        airy_label_map = {
            2: "single Mode(100GHz)",
            3: "multy Mode(10Ghz)",
            4: "multy Mode(1Ghz)",
        }
        airy_label = airy_label_map.get(airy, f"airy={airy} $\\frac{{\\lambda}}{{D}}$")

        # E: without correction – dashed line
        ax.errorbar(
            r0_vals_cm,
            E_mean,
            yerr=E_std,
            fmt="--o",
            color=color,
            markersize=5,
            alpha=0.9,
            linewidth=1.2,
            label=f"without AO ({airy_label})",
        )

        # D: with correction – solid line
        ax.errorbar(
            r0_vals_cm,
            D_mean,
            yerr=D_std,
            fmt="-s",
            color=color,
            markersize=5,
            alpha=0.9,
            linewidth=1.6,
            label=f"with AO ({airy_label})",
        )

    ax.set_xlabel("r_0[cm]")
    ax.set_ylabel("Normalized power")
    ax.set_title("AO effectiveness")
    ax.set_xlim(left=None, right=7.5)

    # Logarithmic y-axis for better visual spread of normalized power.
    ax.set_yscale("linear")

    # Grid for both major and minor ticks.
    ax.grid(True, which="both", alpha=0.3)

    # Standard legend with all series.
    ax.legend(fontsize=8)

    fig.tight_layout()

    base_output_dir = "Plots"
    os.makedirs(base_output_dir, exist_ok=True)
    out_path = os.path.join(
        base_output_dir,
        "AO_effectiveness.png",
    )

    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"✅ Saved E/D offset figure to: {out_path}")
    plt.close(fig)
    return fig
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
        stats = (
            group
            .groupby(x_col)[y_col]
            .agg(['mean', 'std'])
            .reset_index()
            .sort_values(by=x_col)
        )
        stats['std'] = stats['std'].fillna(0.0)

        ax.errorbar(
            stats[x_col],
            stats['mean'],
            yerr=stats['std'],
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
    x_label="r0_ref[m]",
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
        df,
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
        df,
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
    utils.check_dir()
    # מריץ את הפונקציה על הקובץ הקבוע
    if run_as_AO:
        ybl='E_power_sum'
        yal="D_power_sum"
    else:
        ybl="Power in bucket (before turbulence)"
        yal="Power in bucket (after turbulence)"
        
    fig, axes = plot_before_after_from_out_csv(
        csv_path=path_file,
        x_label="r0_ref[m]",
        y_before_col=ybl,
        y_after_col=yal,
        title_before="Power in bucket vs r0_ref (before turbulence)",
        title_after="Power in bucket vs r0_ref (after turbulence)",
    )
    Fig_E_and_D = plot_E_and_D()
    # Log today's date
    print(date.today())
