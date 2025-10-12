# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import hcipy as hci

# ===================== 0) CONFIG (as requested) =====================
telescope_diameter = 8.0       # [m]
central_obscuration = 1.2      # [m]
central_obscuration_ratio = central_obscuration / telescope_diameter
spider_width = 0.05            # [m]

oversizing_factor = 16.0 / 15.0
num_pupil_pixels_base = 240
num_pupil_pixels = int(num_pupil_pixels_base * oversizing_factor)  # keep as-is (even ~256)
pupil_grid_diameter = telescope_diameter * oversizing_factor

wavelength_sci = 2.2e-6        # [m]
wavelength_ref = 500e-9        # [m] for r0 input
r0_ref_500 = 0.02               # [m] r0 at 500 nm (as given)
L0 = None                      # von Kármán outer scale -> None => pure Kolmogorov
rng_seed = 2025

# Focal grid (keep your call, i.e., 'naive' path allowed)
q = 4
num_airy = 30
spatial_resolution = wavelength_sci / telescope_diameter  # your original choice

# PIB bucket: 9 µm DIAMETER on the physical focal plane with f = 1.2 m
bucket_diameter_m = 9e-6               # [m]
bucket_radius_m   = bucket_diameter_m / 2.0
focal_length_m    = 1.2                # [m] (as given)

# ===================== 1) PUPIL & FOCAL GRIDS =====================
pupil_grid = hci.make_pupil_grid(num_pupil_pixels, pupil_grid_diameter)

VLT_aperture_generator = hci.make_obstructed_circular_aperture(
    telescope_diameter,
    central_obscuration_ratio,
    num_spiders=4,
    spider_width=spider_width
)
VLT_aperture = hci.evaluate_supersampled(VLT_aperture_generator, pupil_grid, 4)

# Keep your original focal grid call (this may use the 'naive' FT path)
focal_grid = hci.make_focal_grid(q=q, num_airy=num_airy, spatial_resolution=spatial_resolution)

# ===================== 2) IDEAL PSF =====================
wf = hci.Wavefront(VLT_aperture.copy(), wavelength_sci)
wf.total_power = 1.0

propagator = hci.FraunhoferPropagator(pupil_grid, focal_grid)
E_img = propagator.forward(wf)          # complex field
I_img = E_img.power                     # intensity Field on focal_grid (PSF)
I2    = I_img.shaped

# ===================== 3) TURBULENCE PHASE SCREEN (single, Kolmogorov) =====================
def _grid_span_m(pgrid, fallback):
    # Prefer spacing 'delta' if present
    if hasattr(pgrid, 'delta') and pgrid.delta is not None:
        Ny, Nx = pgrid.shape
        dx = float(pgrid.delta[0]) if np.ndim(pgrid.delta) else float(pgrid.delta)
        return dx * max(Nx, Ny)
    # Otherwise derive from coordinates
    try:
        xs = np.asarray(pgrid.x); ys = np.asarray(pgrid.y)
        return float(max(xs.max()-xs.min(), ys.max()-ys.min()))
    except Exception:
        return float(fallback)

def make_phase_screen_vonK(pgrid, ap_field, D_tel, lam_sci, r0_ref_500, lam_ref,
                           L0=None, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    r0_sci = r0_ref_500 * (lam_sci/lam_ref)**(6.0/5.0)

    Ny, Nx = pgrid.shape
    assert Ny == Nx, "pupil_grid must be square."
    N = Ny

    L  = _grid_span_m(pgrid, pupil_grid_diameter)  # physical span [m]
    dx = L / N
    fx = np.fft.fftfreq(N, d=dx)
    FX, FY = np.meshgrid(fx, fx, indexing='xy')
    K = 2*np.pi*np.sqrt(FX**2 + FY**2)  # [rad/m]

    if L0 is None or L0 <= 0:
        A = np.where(K == 0, 0.0, K**(-11.0/6.0))
    else:
        K0 = 1.0 / L0
        A = (K**2 + K0**2)**(-11.0/12.0)

    Wc = (rng.standard_normal((N, N)) + 1j*rng.standard_normal((N, N))) / np.sqrt(2.0)
    S  = Wc * A
    phi = np.real(np.fft.ifft2(S))
    phi -= np.mean(phi)

    # Scale RMS over the aperture: Var ≈ 1.03 * (D/r0)^ (5/3)
    ap_mask = (ap_field.shaped > 0.5)
    var_target   = 1.03 * (D_tel / r0_sci)**(5.0/3.0)
    sigma_target = np.sqrt(var_target)
    sigma_curr   = np.std(phi[ap_mask]) if np.any(ap_mask) else np.std(phi)
    if sigma_curr > 0:
        phi *= (sigma_target / sigma_curr)

    return hci.Field(phi.ravel(), pgrid), phi, L

rng = np.random.default_rng(rng_seed)
phi_field, phi2, L_ps = make_phase_screen_vonK(
    pupil_grid, VLT_aperture, telescope_diameter, wavelength_sci,
    r0_ref_500, wavelength_ref, L0=L0, rng=rng
)

wf_turb = hci.Wavefront(wf.electric_field.copy(), wavelength_sci)
wf_turb.electric_field *= np.exp(1j * phi_field)

I2_turb = propagator.forward(wf_turb).power.shaped

# ===================== 4) PIB (circular, 9 µm diameter) =====================
# Convert physical radius (meters) to a radius in PIXELS on the *angular* focal grid.
# pixel_angle ≈ (spatial_resolution / q) [radians per pixel]
theta_per_pix = spatial_resolution / q            # [rad/pixel]
phys_per_pix  = focal_length_m * theta_per_pix    # [m/pixel]
bucket_radius_pix = bucket_radius_m / phys_per_pix

ny, nx = focal_grid.shape
cy, cx = ny // 2, nx // 2
Y, X = np.indices((ny, nx))
Rpix = np.sqrt((X - cx)**2 + (Y - cy)**2)
mask_bucket = (Rpix <= bucket_radius_pix).astype(float)

# Weighted (support scalar/array/None)
w = getattr(focal_grid, 'weights', None)
if w is None:
    W = np.ones_like(I2, dtype=float)
elif np.isscalar(w):
    W = np.full_like(I2, w, dtype=float)
else:
    W = np.asarray(w).reshape(I2.shape)

Itot_ideal = np.sum(I2 * W)
Ib_ideal   = np.sum(I2 * W * mask_bucket)
PIB_ideal  = Ib_ideal / Itot_ideal

Itot_turb = np.sum(I2_turb * W)
Ib_turb   = np.sum(I2_turb * W * mask_bucket)
PIB_turb  = Ib_turb / Itot_turb

print(f"[INFO] focal pixel scale: theta/pix = {theta_per_pix:.3e} rad, phys/pix = {phys_per_pix*1e6:.3f} µm")
print(f"[INFO] bucket radius (pix) = {bucket_radius_pix:.3f}  (for 9 µm diameter)")

# ===================== 5) PLOTS =====================
fig, axes = plt.subplots(2, 2, figsize=(12.5, 10.2))

# (1) Pupil
hci.imshow_field(VLT_aperture, cmap='gray', ax=axes[0,0])
axes[0,0].set_title('Pupil (VLT-like)')
axes[0,0].set_xlabel('pupil x [m]'); axes[0,0].set_ylabel('pupil y [m]')

# (2) PSF (no turbulence)
im1 = axes[0,1].imshow(np.log10(np.maximum(I2 / I2.max(), 1e-12)),
                       origin='lower', cmap='inferno', vmin=-6)
axes[0,1].set_title('PSF (no turbulence) — log10')
axes[0,1].set_xlabel('focal x [λ/D]'); axes[0,1].set_ylabel('focal y [λ/D]')
# overlay circular bucket
circ = plt.Circle((cx, cy), bucket_radius_pix, edgecolor='c', fill=False, linewidth=1.6)
axes[0,1].add_patch(circ)
plt.colorbar(im1, ax=axes[0,1], fraction=0.046, pad=0.04)

# (3) Turbulence view (phase screen)
im0 = axes[1,0].imshow(phi2, origin='lower', cmap='twilight',
                       extent=[-L_ps/2, L_ps/2, -L_ps/2, L_ps/2])
axes[1,0].set_title('Synthetic phase screen [rad] (Kolmogorov)')
axes[1,0].set_xlabel('pupil x [m]'); axes[1,0].set_ylabel('pupil y [m]')
plt.colorbar(im0, ax=axes[1,0], fraction=0.046, pad=0.04)

# (4) PSF (with turbulence)
im2 = axes[1,1].imshow(np.log10(np.maximum(I2_turb / I2_turb.max(), 1e-12)),
                       origin='lower', cmap='inferno', vmin=-6)
axes[1,1].set_title('PSF (with turbulence) — log10')
axes[1,1].set_xlabel('focal x [λ/D]'); axes[1,1].set_ylabel('focal y [λ/D]')
circ2 = plt.Circle((cx, cy), bucket_radius_pix, edgecolor='c', fill=False, linewidth=1.6)
axes[1,1].add_patch(circ2)
plt.colorbar(im2, ax=axes[1,1], fraction=0.046, pad=0.04)

plt.suptitle(
    f"PIB (circular ϕ=9 µm @ f={focal_length_m} m): "
    f"ideal={PIB_ideal:.3f} | turbulent={PIB_turb:.3f}"
)
plt.tight_layout(); plt.show()
