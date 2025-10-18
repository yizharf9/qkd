path_file="./massive_output.csv"
#wavelength,r0_ref,run_number,power_in_bucket_before_turbulance,total_power_before_turbulance,precentage_before_turbulance,power_in_bucket_after_turbulance,total_power_after_turbulance,precentage_after_turbulance,time
# run this to plot the massive_output.csv file
import pandas as pd
import matplotlib.pyplot as plt
import time
import os
#--- 0) Ensure correct working directory ---
print("Checking current working directory...")
current_directory_name = os.path.basename(os.getcwd())
if current_directory_name == 'oct_2025':
    print(f"✅ Success: Script is running from the correct directory ('{current_directory_name}').")
else:
    print(f"⚠️ Warning: Script is NOT running from the 'oct_2025' directory.")
    print(f"   Current directory is: '{current_directory_name}'")
    exit("Execution stopped. Please run the script from the 'oct_2025' directory.")
print("-" * 60) # Visual separator
long_wl = 1.55e-6  # [m]
# --- 1) Load CSV ---
df = pd.read_csv(path_file)
print(f"Loaded {len(df)} rows from {path_file}")
print("#"*50)
# --- 2) Normalize column names (robust to spaces/case) ---
df.columns = df.columns.str.strip().str.lower()
print(df.columns)
#df=df[df["wavelength"]<6e-7]
print(f"Loaded {len(df)} rows from {path_file}")
def pick(colnames, *candidates):
    for c in candidates:
        if c in colnames:
            return c
        hits = [cn for cn in colnames if c in cn]
        if hits:
            return hits[0]
    return None

# --- 3) Locate columns ---
#df=df[0:2200]
#df=df.sorted()
after =True # for after , for before set to False


print(f"Loaded {len(df)} rows from {path_file}")
r0_col  = pick(df.columns, "r0", "r_0", "r0_ref", "r_0_ref")
smf_col_after = pick(df.columns,"power_in_bucket_after_turbulance")
smf_col_before = pick(df.columns,"power_in_bucket_before_turbulance")
wl_col  = pick(df.columns, "wavelength", "lambda", "lam", "wl")
print("#"*50)
print(f"Loaded {len(r0_col)} rows from {path_file}")
if after:
    smf_col = smf_col_after
else:
    smf_col = smf_col_before
if not (r0_col and smf_col and wl_col ):
    raise ValueError(f"Need columns for r0, smf, wavelength. Found: {list(df.columns)}")

# --- 4) Clean types ---
df[r0_col]  = pd.to_numeric(df[r0_col], errors="coerce")
df[smf_col] = pd.to_numeric(df[smf_col], errors="coerce")
df[wl_col]  = pd.to_numeric(df[wl_col], errors="coerce")
df = df.dropna(subset=[r0_col, smf_col, wl_col])

# 1) Group by settings and compute stats of the measured value (SMF)
group_cols = [wl_col, r0_col]          # settings that define a case
val_col    = smf_col                   # the measured metric to summarize

stats = (
    df.groupby(group_cols, dropna=False)[val_col]
      .agg(count='count', mean='mean', min='min', max='max')
      .reset_index()
      .sort_values(group_cols)
)

# Optional: sanity check — do we have ~100 runs per setting?
print(stats.groupby(wl_col)['count'].describe())

# 2) Plot: per wavelength, mean curve with a shaded min–max envelope
plt.figure(figsize=(8, 5))
for wl, d in stats.groupby(wl_col, sort=True):
    d = d.sort_values(r0_col)
    plt.plot(d[r0_col], d['mean'], label=f"λ = {wl:g} (mean)")
    plt.fill_between(d[r0_col], d['min'], d['max'], alpha=0.2, linewidth=0)

plt.xlabel(r0_col)
plt.ylabel(val_col)
plt.title(f"{val_col} vs {r0_col} — mean with min–max band per wavelength")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
#plt.show()
time.sleep(1)  # ensure the plot is rendered before printing below
plt.close()

# 3) (Optional) Show a compact table for quick inspection
print("\nHead of grouped stats:")
print(stats.head(12).to_string(index=False))
print("\nTail of grouped stats:")
print(stats.tail(12).to_string(index=False))    



print("data for display:")
print(df.tail())
# 2) Plot A: mean curve with min–max envelope (optional figure)
fig_stats, ax_stats = plt.subplots(figsize=(8, 5))
for wl, d in stats.groupby(wl_col, sort=True):
    d = d.sort_values(r0_col)
    ax_stats.plot(d[r0_col], d['mean'], label=f"λ = {wl:g} (mean)")
    ax_stats.fill_between(d[r0_col], d['min'], d['max'], alpha=0.2, linewidth=0)
ax_stats.set(
    xlabel=r0_col,
    ylabel=val_col,
    title=f"{val_col} vs {r0_col} — mean with min–max band per wavelength"
)
ax_stats.grid(True, alpha=0.3)
ax_stats.legend()
fig_stats.tight_layout()
plt.close(fig_stats)  # we’re not saving this one

# 3) Plot B: dot-per-row + mean line per wavelength (this is the one we save)
import itertools
fig_scatter, ax = plt.subplots(figsize=(9, 6))

# consistent colors per wavelength
color_cycle = itertools.cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])
unique_wls = sorted(df[wl_col].unique())
wl_color = {wl: next(color_cycle) for wl in unique_wls}

# dots
for wl, d in df.groupby(wl_col, sort=True):
    ax.scatter(d[r0_col], d[val_col], s=14, alpha=0.55, linewidths=0,
               color=wl_color[wl], label=f"λ = {wl:g} dots")

# mean lines
for wl, d in stats.groupby(wl_col, sort=True):
    d = d.sort_values(r0_col)
    ax.plot(d[r0_col], d['mean'], color=wl_color[wl], linewidth=2.2,
            label=f"λ = {wl:g} mean")

ax.set(
    xlabel=r0_col,
    ylabel=val_col,
    title=f"{val_col} vs {r0_col} — dots (rows) + mean line per wavelength"
)
ax.grid(True, alpha=0.3)
ax.legend(ncol=2, frameon=True)
fig_scatter.tight_layout()

# ---- SAVE THE CORRECT FIGURE (the dots+mean one) ----
os.makedirs("plots", exist_ok=True)
fname = f"{val_col}_vs_{r0_col}_{'after' if after else 'before'}_{int(time.time())}.png"
save_path = os.path.join("plots", fname)
fig_scatter.savefig(save_path, dpi=300, bbox_inches="tight")  # <-- save THIS figure
print(f"✅ Saved plot to: {save_path}")

# (optional) show after saving
#plt.show()
plt.close(fig_scatter)

import itertools
import numpy as np
from matplotlib.ticker import LogLocator, NullFormatter, ScalarFormatter

# 3) Plot B: dot-per-row + mean line per wavelength (log-log)
fig_scatter, ax = plt.subplots(figsize=(9, 6))

# -------- filter to positive domain for log scale --------
pos_df = df[(df[r0_col] > 0) & (df[val_col] > 0)].copy()
pos_stats = stats[
    (stats[r0_col] > 0) &
    (stats['mean'] > 0) &
    (stats['min']  > 0) &
    (stats['max']  > 0)
].copy()

# consistent colors per wavelength
color_cycle = itertools.cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])
unique_wls = sorted(pos_df[wl_col].unique())
wl_color = {wl: next(color_cycle) for wl in unique_wls}

# dots (each row)
for wl, d in pos_df.groupby(wl_col, sort=True):
    ax.scatter(d[r0_col], d[val_col],
               s=14, alpha=0.55, linewidths=0,
               color=wl_color[wl], label=f"λ = {wl:g} dots")

# mean lines
for wl, d in pos_stats.groupby(wl_col, sort=True):
    d = d.sort_values(r0_col)
    ax.plot(d[r0_col], d['mean'],
            color=wl_color[wl], linewidth=2.2,
            label=f"λ = {wl:g} mean")

# ---- log scales + tidy ticks ----
ax.set_xscale('log')
ax.set_yscale('log')

# Major & minor locators (auto is fine; this cleans up clutter)
ax.xaxis.set_major_locator(LogLocator(base=10.0))
ax.yaxis.set_major_locator(LogLocator(base=10.0))
ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10)*0.1))
ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10)*0.1))
ax.xaxis.set_minor_formatter(NullFormatter())
ax.yaxis.set_minor_formatter(NullFormatter())

# Labels / title
ax.set_xlabel(r0_col + " (log scale)")
ax.set_ylabel(val_col + " (log scale)")
ax.set_title(f"{val_col} vs {r0_col} — dots + mean (log–log)")

# Grid on both major & minor
ax.grid(True, which='both', alpha=0.3)

# Legend & layout
ax.legend(ncol=2, frameon=True)
fig_scatter.tight_layout()

# ---- SAVE ----
os.makedirs("plots", exist_ok=True)
fname = f"{val_col}_vs_{r0_col}_{'after' if after else 'before'}_loglog_{int(time.time())}.png"
save_path = os.path.join("plots", fname)
fig_scatter.savefig(save_path, dpi=300, bbox_inches="tight")
print(f"✅ Saved plot to: {save_path}")

plt.show()
plt.close(fig_scatter)
