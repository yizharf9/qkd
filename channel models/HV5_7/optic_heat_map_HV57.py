# -*- coding: utf-8 -*-
"""
uplink_phase_screens.py
-----------------------
סימולציית uplink לווייני עם טורבולנציה (HV5/7) בשיטת phase-screens.
- גובה לווין: 200 ק"מ
- זווית גבהה: 60°
- אורך גל: 1550 nm (ניתן לשינוי)
- ספקטרום: von-Kármán (עם inner/outer scales)
- פלט: שני קבצי PNG — A1 (יחס קבלה מול שטח משדר) ו-A2 (מול שטח מקלט)

תלויות: numpy, matplotlib בלבד.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft2, ifft2, fftshift, ifftshift
from math import pi, sqrt


# ===================== פרמטרים ניתנים לשינוי =====================

# גיאומטריה
ELEV_DEG   = 60.0        # elevation (מעלות)
H_SAT_M    = 200e3       # גובה לווין [m]
H_TURB_TOP = 30e3        # טווח גבהים עם טורבולנציה [m]

# אופטיקה
LAM        = 1550e-9     # אורך גל [m]
LAM_str=str(int(LAM*10**9))+"nm"
# פרופיל HV5/7 (יבשתי) + von-Kármán
HV57_A     = 1.7e-14     # ground term [m^(-2/3)]
HV57_V     = 21.0        # RMS wind [m/s]
L0         = 100.0       # outer scale [m]
l0         = 0.002       # inner scale [m]

# phase screens
NUM_SCREENS  = 24
REALIZATIONS = 6         # מס' מימושים להממוצע

# רשת חישוב
N  = 256                 # grid size (N×N)
DX = 0.10                # מרווח דגימה במישור המקור [m] (חלון ≈ N*DX)

# סריקות לאיור A1/A2
DT_VALUES    = np.array(np.linspace(0.08,1,15))                # קוטר משדר [m]
DR_VALUES    = np.array(np.linspace(0.08,1,15))      # קוטר מקלט [m]
DT_BASELINE  = 0.20
DR_BASELINE  = 0.20

# שמות קבצים
FIG_A1 = "TRANS_ps_calibrated_"+str(LAM_str)+".png"
FIG_A2 = "RECIEVE_ps_calibrated_"+str(LAM_str)+".png"
# ===================== נגזרות וחישוב מקדים =====================

k = 2.0 * pi / LAM
L_TOTAL = H_SAT_M / np.sin(np.deg2rad(ELEV_DEG))
L_TURB  = H_TURB_TOP / np.sin(np.deg2rad(ELEV_DEG))
L_VAC   = max(L_TOTAL - L_TURB, 0.0)

k0 = 1.0 / L0
km = 5.92 / l0

x  = (np.arange(-N//2, N//2)) * DX
X, Y = np.meshgrid(x, x)

fx = fftshift(np.fft.fftfreq(N, d=DX))
FX, FY = np.meshgrid(fx, fx)
FSQ    = FX**2 + FY**2

KX = 2*np.pi*FX
KY = 2*np.pi*FY
K2 = KX**2 + KY**2

# --- Coupled sweep controls ---
COUPLED_MODE   = True                 # הפעלה / כיבוי של הסריקה המצומדת
COUPLING_KIND  = "diameter"           # "diameter": D_R = r * D_T   |  "area": A_R = r * A_T
RATIO_LIST     = [0.5, 1.0, 1.5]      # דוגמאות יחס

# --- Heatmap grid (radii in meters) ---
GRID_R_MIN   = 0.2
GRID_R_MAX   = 0.8
GRID_R_STEP  = 0.1
HEATMAP_FILE = f"ETA_heatmap_R{GRID_R_MIN:.1f}-{GRID_R_MAX:.1f}m_{int(LAM*1e9)}nm.png"


# ===================== פונקציות מודל =====================

def Cn2_HV57(h: np.ndarray) -> np.ndarray:
    """HV5/7 land profile: Cn^2(h) עם h במטרים."""
    return (0.00594*(HV57_V/27.0)**2*(1e-5*h)**10*np.exp(-h/1000.0)
            + 2.7e-16*np.exp(-h/1500.0)
            + HV57_A*np.exp(-h/100.0))

def asm_transfer(dz: float) -> np.ndarray:
    """Angular Spectrum (paraxial) transfer function."""
    return np.exp(1j*k*dz) * np.exp(-1j*np.pi*LAM*dz*FSQ)

def propagate(U: np.ndarray, dz: float) -> np.ndarray:
    """תעבורת גל באמצעות ASM."""
    return ifft2( fft2(U) * ifftshift(asm_transfer(dz)) )

def launch_field(D_T: float) -> np.ndarray:
    """שיגור Gaussian: w0 = D_T/2 (רדיוס 1/e^2)."""
    w0 = D_T/2.0
    return np.exp(-(X**2 + Y**2)/w0**2)

def phase_screen_vk_calibrated(dz: float, Cn2_val: float, rng: np.random.Generator) -> np.ndarray:
    """
    מסך פאזה von-Kármán לשכבה dz עם Cn^2 נתון.
    מכויל כך ש-Var[phi] יתאים לאינטגרל התאורטי של Φ_φ.
    """
    Phi_n  = 0.033 * Cn2_val * (K2 + k0**2)**(-11/6.0) * np.exp(-K2/(km**2))
    Phi_ph = 2*np.pi * (k**2) * dz * Phi_n

    dfx = float(fx[1] - fx[0])
    dfy = dfx
    var_theory = np.sum(Phi_ph) * (dfx*dfy)

    W    = (rng.normal(size=(N,N)) + 1j*rng.normal(size=(N,N))) / sqrt(2.0)
    Fphi = W * np.sqrt(Phi_ph) * (dfx*dfy)**0.5
    phi_raw = np.real(ifft2(ifftshift(Fphi))) * (N*DX)**2

    var_emp = np.var(phi_raw)
    scale   = 1.0 if (var_emp <= 0.0 or var_theory <= 0.0) else np.sqrt(var_theory/var_emp)
    return np.exp(1j * (phi_raw * scale))

def encircled_energy(I: np.ndarray, a_R: float) -> float:
    """יחס אנרגיה בתוך פתח עגול ברדיוס a_R."""
    R = np.sqrt(X**2 + Y**2)
    return float((I * (R <= a_R)).sum() / I.sum())

def eta_propagate(D_T: float, D_R: float, use_turb: bool,
                  Cn2_layers: np.ndarray, dz_s: float,
                  realizations: int = REALIZATIONS, seed0: int = 1234) -> float:
    """חישוב יחס קליטה עם/בלי טורבולנציה (ממוצע על מימושים)."""
    rng  = np.random.default_rng(seed0)
    a_R  = D_R/2.0
    runs = realizations if use_turb else 1
    etas = []

    for _ in range(runs):
        U = launch_field(D_T)
        if use_turb:
            for Cn2v in Cn2_layers:
                U = propagate(U, dz_s/2.0)
                U = U * phase_screen_vk_calibrated(dz_s, Cn2v, rng)
                U = propagate(U, dz_s/2.0)
            if L_VAC > 0:
                U = propagate(U, L_VAC)
        else:
            U = propagate(U, L_TOTAL)

        I = np.abs(U)**2
        etas.append(encircled_energy(I, a_R))

    return float(np.mean(etas))

# ===================== Main =====================

# ---- Coupled sweep: Rx grows with Tx by a chosen ratio ---------------



def main():
    if COUPLED_MODE:
        # build layers once
        dz_s = L_TURB / NUM_SCREENS
        s_pos = (np.arange(NUM_SCREENS) + 0.5) * dz_s
        h_alts = s_pos * np.sin(np.deg2rad(ELEV_DEG))
        Cn2_layers = Cn2_HV57(h_alts)

        # ---------- (1) Coupled curves: η vs D_T for several ratios ----------
        plt.figure()
        for ratio in RATIO_LIST:
            DT_list = DT_VALUES.copy()
            DR_list = []

            # map DT -> DR according to the coupling rule
            if COUPLING_KIND.lower().startswith("diam"):
                # diameter ratio: D_R = ratio * D_T
                DR_list = [ratio * DT for DT in DT_list]
                curve_label = f"DR = {ratio:.2f} · DT"
            elif COUPLING_KIND.lower().startswith("area"):
                # area ratio: A_R = ratio * A_T  => D_R = sqrt(ratio) * D_T
                s = np.sqrt(ratio)
                DR_list = [s * DT for DT in DT_list]
                curve_label = f"AR = {ratio:.2f} · AT  (DR = {s:.2f}·DT)"
            else:
                raise ValueError("COUPLING_KIND must be 'diameter' or 'area'")

            # compute η for each (DT, DR) pair with turbulence
            etas = []
            for DT, DR in zip(DT_list, DR_list):
                etas.append(eta_propagate(DT, DR, True, Cn2_layers, dz_s, seed0=2025))

            # x-axis: DT (you can switch to area if you prefer)
            x_vals = DT_list  # DT on x
            # x_vals = np.pi * (DT_list/2)**2         # uncomment to plot vs AT

            plt.plot(x_vals, etas, marker='s', linestyle='--', label=curve_label)

        plt.xlabel("Transmitter diameter D_T [m]")  # or "Transmitter area A_T [m²]"
        plt.ylabel("Received fraction η (P_R / P_T)")
        plt.title(
            f"Coupled sweep (Rx grows with Tx) | L≈{L_TOTAL / 1e3:.0f} km, elev {ELEV_DEG}°, λ={LAM * 1e9:.0f} nm")
        plt.grid(True);
        plt.legend();
        plt.tight_layout()
        plt.savefig("COUPLED_ps.png", dpi=150)
        plt.close()
        print("Wrote COUPLED_ps.png")

        # ---------- (2) Heatmap: η over all radius pairs (R_T × R_R) ----------
        radii_T = np.arange(GRID_R_MIN, GRID_R_MAX + 1e-9, GRID_R_STEP)
        radii_R = np.arange(GRID_R_MIN, GRID_R_MAX + 1e-9, GRID_R_STEP)

        grid_turb = np.zeros((len(radii_T), len(radii_R)))
        for i, RT in enumerate(radii_T):
            DT = 2.0 * RT
            for j, RR in enumerate(radii_R):
                DR = 2.0 * RR
                grid_turb[i, j] = eta_propagate(DT, DR, True, Cn2_layers, dz_s, seed0=3000 + i * 100 + j)

        # example marker: RT=0.4 m (DT=0.8), RR=0.6 m (DR=1.2)
        RT_ex, RR_ex = 0.4, 0.6
        i_ex = int(round((RT_ex - GRID_R_MIN) / GRID_R_STEP))
        j_ex = int(round((RR_ex - GRID_R_MIN) / GRID_R_STEP))

        plt.figure()
        extent = [radii_R[0], radii_R[-1], radii_T[0], radii_T[-1]]  # x=R_R, y=R_T
        im = plt.imshow(grid_turb, origin="lower", aspect="auto", extent=extent)
        plt.colorbar(im, label="η (P_R / P_T)")
        plt.xlabel("Receiver radius R_R [m]")
        plt.ylabel("Transmitter radius R_T [m]")
        plt.title(f"η heatmap (Turbulence) | L≈{L_TOTAL / 1e3:.0f} km, elev {ELEV_DEG}°, λ={LAM * 1e9:.0f} nm")
        # annotate the example point if it falls on the grid
        if 0 <= i_ex < len(radii_T) and 0 <= j_ex < len(radii_R):
            plt.plot(RR_ex, RT_ex, marker='o', markersize=6, fillstyle='none', markeredgewidth=1.5)
            plt.text(RR_ex + 0.01, RT_ex + 0.01, f"RT={RT_ex:.1f}, RR={RR_ex:.1f}", fontsize=9)

        plt.tight_layout()
        plt.savefig(HEATMAP_FILE, dpi=150)
        plt.close()
        print(f"Wrote {HEATMAP_FILE}")
if __name__ == "__main__":
    main()
