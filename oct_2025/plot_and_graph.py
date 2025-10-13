path_file="/Users/idoshlomy/PycharmProjects/qkd/massive_output.csv"
import pandas as pd
import matplotlib.pyplot as plt
import time
long_wl = 1.55e-6  # [m]
# --- 1) Load CSV ---
df = pd.read_csv(path_file)
print(f"Loaded {len(df)} rows from {path_file}")
# --- 2) Normalize column names (robust to spaces/case) ---
df.columns = df.columns.str.strip().str.lower()
print(df.columns)
df=df[df["wavelength"]<6e-7]
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
df=df[0:2200]
#df=df.sorted()
print(f"Loaded {len(df)} rows from {path_file}")
r0_col  = pick(df.columns, "r0", "r_0")
smf_col = pick(df.columns, "smf", "single_mode_power", "single-mode", "single mode")
wl_col  = pick(df.columns, "wavelength", "lambda", "lam", "wl")

if not (r0_col and smf_col and wl_col):
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
plt.show()
time.sleep(1)  # ensure the plot is rendered before printing below
plt.close()

# 3) (Optional) Show a compact table for quick inspection
print("\nHead of grouped stats:")
print(stats.head(12).to_string(index=False))
print("\nTail of grouped stats:")
print(stats.tail(12).to_string(index=False))    



print("data for display:")
print(df.tail())
