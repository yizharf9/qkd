# -*- coding: utf-8 -*-
"""
fsoc_classes.py
----------------
Two collaborating classes:
- TransmittedSignal (base) + GaussianSignal (current concrete implementation)
- Channel: holds the physical medium, grid, phase-screens and propagation;
           computes η and generates figures.

Public behavior: produce plots as external artifacts.
"""

from __future__ import annotations
import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift
from math import pi, sqrt
import matplotlib.pyplot as plt

import params as P


# ===================== Helper: grids & params =====================
def make_grids(N: int, DX: float):
    """Return spatial grids (X,Y) and frequency grids (FX,FY) with their squared magnitudes."""
    x = (np.arange(-N//2, N//2)) * DX
    X, Y = np.meshgrid(x, x)

    fx = fftshift(np.fft.fftfreq(N, d=DX))
    FX, FY = np.meshgrid(fx, fx)
    FSQ    = FX**2 + FY**2

    KX = 2.0*np.pi*FX
    KY = 2.0*np.pi*FY
    K2 = KX**2 + KY**2
    return X, Y, FX, FY, FSQ, K2

def _pget(name: str, default):
    """Safe params getter with fallback defaults if attribute missing in params.py."""
    return getattr(P, name, default)


# ===================== Signals =====================
class TransmittedSignal:
    """Abstract signal. Subclasses implement sample_on(X,Y) -> U0(x,y)."""
    name: str = "AbstractSignal"

    def sample_on(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Override in subclasses.")


class GaussianSignal(TransmittedSignal):
    """Circular Gaussian launcher; w0 = D_T/2 (1/e^2 intensity radius)."""
    name: str = "Gaussian"

    def __init__(self, D_T: float):
        self.D_T = float(D_T)

    def sample_on(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        w0 = self.D_T / 2.0
        return np.exp(-(X**2 + Y**2)/w0**2)


# ===================== Channel =====================
class Channel:
    """
    Represents the FSOC uplink medium + propagation engine.
    - Owns the grid (N, DX) and k-space grids.
    - Computes phase screens (von-Kármán) calibrated to theory.
    - Provides propagate() and η (encircled-energy) over a circular Rx aperture.
    - Produces figures (plots) as the only public "outputs".
    """

    def __init__(self,
                 elev_deg: float = P.ELEV_DEG,
                 h_sat_m: float = P.H_SAT_M,
                 h_turb_top: float = P.H_TURB_TOP,
                 lam: float = P.LAM,
                 hv57_A: float = P.HV57_A,
                 hv57_V: float = P.HV57_V,
                 L0: float = P.L0,
                 l0: float = P.l0,
                 num_screens: int = P.NUM_SCREENS,
                 N: int = P.N,
                 DX: float = P.DX,
                 rx_offset=(0.0, 0.0),
                 pointing_jitter_std: float = 0.0
                 ):
        # Geometry/optics
        self.elev_deg = float(elev_deg)
        self.h_sat_m = float(h_sat_m)
        self.h_turb_top = float(h_turb_top)
        self.lam = float(lam)

        # HV57 + von-Kármán
        self.hv57_A = float(hv57_A)
        self.hv57_V = float(hv57_V)
        self.L0 = float(L0)
        self.l0 = float(l0)

        # Discretization
        self.num_screens = int(num_screens)
        self.N = int(N)
        self.DX = float(DX)

        # Receiver model (defaults to centered circular aperture)
        self.rx_offset = (float(rx_offset[0]), float(rx_offset[1]))
        self.pointing_jitter_std = float(pointing_jitter_std)

        # Derived constants
        self.k = 2.0 * np.pi / self.lam
        self.L_total = self.h_sat_m / np.sin(np.deg2rad(self.elev_deg))
        self.L_turb = self.h_turb_top / np.sin(np.deg2rad(self.elev_deg))
        self.L_vac = max(self.L_total - self.L_turb, 0.0)

        self.k0 = 1.0 / self.L0
        self.km = 5.92 / self.l0

        # Build grids once  (FIX: removed stray 'ac')
        self.X, self.Y, self.FX, self.FY, self.FSQ, self.K2 = make_grids(self.N, self.DX)

        # Precompute layer heights and Cn^2 profile for the turbulent segment
        self.dz_s = self.L_turb / self.num_screens if self.num_screens > 0 else 0.0
        if self.num_screens > 0:
            s_pos = (np.arange(self.num_screens) + 0.5) * self.dz_s
            self.h_alts = s_pos * np.sin(np.deg2rad(self.elev_deg))   # store for attenuation/jitter models
            self.Cn2_layers = self._Cn2_HV57(self.h_alts)
        else:
            self.h_alts = np.array([])
            self.Cn2_layers = np.array([])

        # ===== Attenuation settings (read from params.py if exist; else use safe defaults) =====
        self.use_attenuation     = _pget("USE_ATTENUATION", False)
        self.atten_model         = _pget("ATTEN_MODEL", "none")             # "none" | "constant_beta" | "koschmieder" | "angstrom" | "rayleigh"
        self.beta_const          = _pget("BETA_CONST", 0.0)                 # [1/m]
        self.visibility_m        = _pget("VISIBILITY_M", 23000.0)           # [m]
        self.scale_with_alt      = _pget("SCALE_WITH_ALT", False)
        self.beta_scale_height_m = _pget("BETA_SCALE_HEIGHT_M", 1500.0)     # [m]
        self.angstrom_alpha      = _pget("ANGSTROM_ALPHA", 1.3)
        self.beta_ref            = _pget("BETA_REF", 2.0e-4)                # [1/m] at LAM_REF
        self.lam_ref             = _pget("LAM_REF", 550e-9)                 # [m]
        self.rayleigh_beta_ref   = _pget("RAYLEIGH_BETA_REF", 0.0)          # [1/m] at LAM_REF_R
        self.lam_ref_r           = _pget("LAM_REF_R", 550e-9)               # [m]

    # ---------- Turbulence profile (HV5/7 land) ----------
    def _Cn2_HV57(self, h: np.ndarray) -> np.ndarray:
        """HV5/7 land profile: Cn^2(h) with h in meters."""
        return (0.00594*(self.hv57_V/27.0)**2*(1e-5*h)**10*np.exp(-h/1000.0)
                + 2.7e-16*np.exp(-h/1500.0)
                + self.hv57_A*np.exp(-h/100.0))

    # ---------- Angular Spectrum transfer ----------
    def _asm_transfer(self, dz: float) -> np.ndarray:
        """Angular Spectrum (paraxial) transfer function for distance dz."""
        return np.exp(1j*self.k*dz) * np.exp(-1j*np.pi*self.lam*dz*self.FSQ)

    def _propagate(self, U: np.ndarray, dz: float) -> np.ndarray:
        """Single-step angular spectrum propagation by dz."""
        return np.fft.ifft2(np.fft.fft2(U) * ifftshift(self._asm_transfer(dz)))

    # ---------- Calibrated von-Kármán phase screen ----------
    def _phase_screen_vk_calibrated(self, dz: float, Cn2_val: float, rng: np.random.Generator) -> np.ndarray:
        """
        One calibrated phase screen for layer thickness dz and given Cn^2.
        Ensures Var[phi] matches the theoretical integral of Φ_φ.
        """
        Phi_n  = 0.033 * Cn2_val * (self.K2 + self.k0**2)**(-11/6.0) * np.exp(-self.K2/(self.km**2))
        Phi_ph = 2*np.pi * (self.k**2) * dz * Phi_n

        # Numerical integral (frequency domain) for variance
        dfx = float(self.FX[0,1] - self.FX[0,0])
        dfy = dfx
        var_theory = np.sum(Phi_ph) * (dfx*dfy)

        # Complex white noise in Fourier plane
        W    = (rng.normal(size=(self.N,self.N)) + 1j*rng.normal(size=(self.N,self.N))) / sqrt(2.0)
        Fphi = W * np.sqrt(Phi_ph) * (dfx*dfy)**0.5
        phi_raw = np.real(ifft2(ifftshift(Fphi))) * (self.N*self.DX)**2

        var_emp = np.var(phi_raw)
        scale   = 1.0 if (var_emp <= 0.0 or var_theory <= 0.0) else np.sqrt(var_theory/var_emp)
        return np.exp(1j * (phi_raw * scale))

    # ---------- Attenuation (Beer–Lambert on field) ----------
    def _beta_ext(self, h: float, lam: float) -> float:
        """
        Return extinction coefficient beta_ext(h, lam) [1/m] according to the selected model.
        Models: "none" | "constant_beta" | "koschmieder" | "angstrom" | "rayleigh".
        Optional altitude scaling: beta(h) *= exp(-h/H).
        """
        model = (self.atten_model or "none").lower()

        if not self.use_attenuation or model == "none":
            beta = 0.0
        elif model == "constant_beta":
            beta = max(self.beta_const, 0.0)
        elif model == "koschmieder":
            V = max(self.visibility_m, 1e-6)
            beta = 3.912 / V
        elif model == "angstrom":
            lam_ref = max(self.lam_ref, 1e-12)
            beta = max(self.beta_ref, 0.0) * (lam / lam_ref)**(-self.angstrom_alpha)
        elif model == "rayleigh":
            lam_ref_r = max(self.lam_ref_r, 1e-12)
            beta = max(self.rayleigh_beta_ref, 0.0) * (lam / lam_ref_r)**(-4.0)
        else:
            beta = 0.0

        if self.scale_with_alt and self.beta_scale_height_m > 0.0:
            beta *= np.exp(-float(h) / float(self.beta_scale_height_m))

        return float(beta)

    # ---------- Encircled energy ----------
    def _encircled_energy(self, I: np.ndarray, a_R: float, dx: float, dy: float) -> float:
        """Fraction of total intensity inside a circular aperture of radius a_R centered at (dx,dy)."""
        total = I.sum()
        if total <= 0:
            return 0.0
        R = np.sqrt((self.X - dx)**2 + (self.Y - dy)**2)
        return float((I * (R <= a_R)).sum() / total)

    # ---------- Core η computation ----------
    def eta(self,
            signal: TransmittedSignal,
            D_R: float,
            use_turb: bool = True,
            realizations: int = P.REALIZATIONS,
            seed_phase: int = 1234,
            seed_jitter: int = 9876) -> float:
        """
        Compute received fraction η (P_R / P_T) for a given transmitted signal and receiver diameter.
        If use_turb=False, a single vacuum propagation is applied.
        Phase-screens are regenerated deterministically from seeds to keep sweeps comparable.
        """
        a_R = float(D_R) / 2.0

        if not use_turb or self.num_screens == 0:
            U = signal.sample_on(self.X, self.Y)
            U = self._propagate(U, self.L_total)
            I = np.abs(U)**2
            dx, dy = self.rx_offset  # center (can be non-zero if user sets it)
            return self._encircled_energy(I, a_R, dx, dy)

        etas = []
        for r in range(realizations):
            # Deterministic RNGs per realization
            rng_ps = np.random.default_rng(seed_phase + r)   # for phase screens
            rng_jt = np.random.default_rng(seed_jitter + r)  # for pointing jitter

            # Sample jitter offset (added to any static rx_offset)
            if self.pointing_jitter_std > 0.0:
                dx_j = rng_jt.normal(scale=self.pointing_jitter_std)
                dy_j = rng_jt.normal(scale=self.pointing_jitter_std)
            else:
                dx_j = dy_j = 0.0

            U = signal.sample_on(self.X, self.Y)
            for idx, Cn2v in enumerate(self.Cn2_layers):
                U = self._propagate(U, self.dz_s/2.0)
                U = U * self._phase_screen_vk_calibrated(self.dz_s, Cn2v, rng_ps)
                U = self._propagate(U, self.dz_s/2.0)
                # ---- Apply attenuation per layer on the FIELD (intensity decays as exp(-beta*dz)) ----
                if self.use_attenuation and self.dz_s > 0.0:
                    h_layer = float(self.h_alts[idx]) if self.h_alts.size > idx else 0.0
                    beta = self._beta_ext(h_layer, self.lam)
                    if beta > 0.0:
                        U = U * np.exp(-0.5 * beta * self.dz_s)

            if self.L_vac > 0:
                U = self._propagate(U, self.L_vac)

            I = np.abs(U)**2
            dx_eff = self.rx_offset[0] + dx_j
            dy_eff = self.rx_offset[1] + dy_j
            etas.append(self._encircled_energy(I, a_R, dx_eff, dy_eff))

        return float(np.mean(etas))

    # ===================== Figure helpers (public behavior) =====================
    def plot_coupled_sweep(self,
                           D_T_values: np.ndarray,
                           ratios: list[float],
                           kind: str = "diameter",
                           realizations: int = P.REALIZATIONS,
                           outfile: str = P.FIG_COUPLED,
                           seed_phase: int = 2025) -> None:
        """
        Plot η vs D_T where the receiver grows with the transmitter by a given ratio set.
        kind="diameter": D_R = r * D_T
        kind="area":     A_R = r * A_T  => D_R = sqrt(r) * D_T
        """
        plt.figure()
        for ratio in ratios:
            if kind.lower().startswith("diam"):
                DR_list = [ratio * DT for DT in D_T_values]
                curve_label = f"DR = {ratio:.2f} · DT"
            elif kind.lower().startswith("area"):
                s = np.sqrt(ratio)
                DR_list = [s * DT for DT in D_T_values]
                curve_label = f"AR = {ratio:.2f} · AT  (DR = {s:.2f}·DT)"
            else:
                raise ValueError("kind must be 'diameter' or 'area'")

            etas = []
            for DT, DR in zip(D_T_values, DR_list):
                sig = GaussianSignal(DT)  # current transmitter stays Gaussian
                etas.append(self.eta(sig, D_R=DR, use_turb=True,
                                     realizations=realizations, seed_phase=seed_phase))

            plt.plot(D_T_values, etas, marker='s', linestyle='--', label=curve_label)

        plt.xlabel("Transmitter diameter D_T [m]")
        plt.ylabel("Received fraction η (P_R / P_T)")
        plt.title(f"Coupled sweep (Rx grows with Tx) | L≈{self.L_total / 1e3:.0f} km, "
                  f"elev {self.elev_deg}°, λ={self.lam * 1e9:.0f} nm")
        plt.grid(True); plt.legend(); plt.tight_layout()
        plt.savefig(outfile, dpi=150); plt.close()

    def plot_eta_heatmap(self,
                         r_min: float,
                         r_max: float,
                         r_step: float,
                         realizations: int = P.REALIZATIONS,
                         outfile: str = P.HEATMAP_FILE,
                         seed_phase_base: int = 3000) -> None:
        """
        Plot a heatmap of η over all (R_T, R_R) pairs on the specified radius grid.
        Uses deterministic seeds per cell (i,j) for reproducibility.
        """
        radii_T = np.arange(r_min, r_max + 1e-12, r_step)
        radii_R = np.arange(r_min, r_max + 1e-12, r_step)

        grid_turb = np.zeros((len(radii_T), len(radii_R)))
        for i, RT in enumerate(radii_T):
            DT = 2.0 * RT
            for j, RR in enumerate(radii_R):
                DR = 2.0 * RR
                sig = GaussianSignal(DT)
                seed_ij = seed_phase_base + i*100 + j
                grid_turb[i, j] = self.eta(sig, D_R=DR, use_turb=True,
                                           realizations=realizations, seed_phase=seed_ij)

        plt.figure()
        extent = [radii_R[0], radii_R[-1], radii_T[0], radii_T[-1]]  # x=R_R, y=R_T
        im = plt.imshow(grid_turb, origin="lower", aspect="auto", extent=extent)
        plt.colorbar(im, label="η (P_R / P_T)")
        plt.xlabel("Receiver radius R_R [m]")
        plt.ylabel("Transmitter radius R_T [m]")
        plt.title(f"η heatmap (Turbulence) | L≈{self.L_total / 1e3:.0f} km, "
                  f"elev {self.elev_deg}°, λ={self.lam * 1e9:.0f} nm")
        plt.tight_layout()
        plt.savefig(outfile, dpi=150); plt.close()
