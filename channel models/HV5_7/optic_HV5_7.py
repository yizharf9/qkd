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
FIG_A1 = "A1_ps_calibrated"+str(LAM)+".png"
FIG_A2 = "A2_ps_calibrated.png"

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

def main():
    # שכבות לאורך המסלול האלכסוני
    dz_s     = L_TURB / NUM_SCREENS
    s_pos    = (np.arange(NUM_SCREENS) + 0.5) * dz_s
    h_alts   = s_pos * np.sin(np.deg2rad(ELEV_DEG))
    Cn2_layers = Cn2_HV57(h_alts)

    # --- A1: יחס קליטה מול שטח המשדר (מקלט קבוע) ---
    A1_Area, A1_vac, A1_turb = [], [], []
    for DT in DT_VALUES:
        A1_Area.append(np.pi*(DT/2)**2)
        A1_vac.append (eta_propagate(DT, DR_BASELINE, False, Cn2_layers, dz_s, seed0=2025))
        A1_turb.append(eta_propagate(DT, DR_BASELINE, True,  Cn2_layers, dz_s, seed0=2025))

    plt.figure()
    plt.plot(A1_Area, A1_vac,  'o-', label='Vacuum (propagated)')
    plt.plot(A1_Area, A1_turb, 's--', label='Turbulence (phase screens)')
    plt.xlabel("Transmitter area A_T [m²]")
    plt.ylabel("Received fraction η (P_R / P_T)")
    plt.title(f"A1: η vs Tx area  |  L~{L_TOTAL/1e3:.0f} km, elev {ELEV_DEG}°, λ={LAM*1e9:.0f} nm")
    plt.grid(True); plt.legend(); plt.tight_layout()
    plt.savefig(FIG_A1, dpi=150)

    # --- A2: יחס קליטה מול שטח המקלט (משדר קבוע) ---
    A2_Area, A2_vac, A2_turb = [], [], []
    for DR in DR_VALUES:
        A2_Area.append(np.pi*(DR/2)**2)
        A2_vac.append (eta_propagate(DT_BASELINE, DR, False, Cn2_layers, dz_s, seed0=777))
        A2_turb.append(eta_propagate(DT_BASELINE, DR, True,  Cn2_layers, dz_s, seed0=777))

    plt.figure()
    plt.plot(A2_Area, A2_vac,  'o-', label='Vacuum (propagated)')
    plt.figure()
    plt.plot(A2_Area, A2_turb, 's--', label='Turbulence (phase screens)')
    plt.xlabel("Receiver area A_R [m²]")
    plt.ylabel("Received fraction η (P_R / P_T)")
    plt.title(f"A2: η vs Rx area  |  L~{L_TOTAL/1e3:.0f} km, elev {ELEV_DEG}°, λ={LAM*1e9:.0f} nm")
    plt.grid(True); plt.legend(); plt.tight_layout()
    plt.savefig(FIG_A2, dpi=150)

    print(f"Done. Wrote {FIG_A1} and {FIG_A2}")

if __name__ == "__main__":
    main()
