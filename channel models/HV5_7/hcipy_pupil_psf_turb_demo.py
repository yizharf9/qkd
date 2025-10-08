"""
Free-Space Optical Communication (FSOC) Simulation
Simulates wavefront propagation through atmospheric turbulence layers

This code compares:
- Multiple wavelengths (850nm, 1550nm, 10.6μm)
- Diffraction-limited (ideal) vs Turbulence-limited (realistic) cases
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import hcipy

# ============================================================================
# PARAMETERS - Adjust these to change simulation
# ============================================================================

# Toggle turbulence on/off for comparison
ENABLE_TURBULENCE = True

# Wavelengths to simulate (in meters)
WAVELENGTHS = [
    850e-9,    # 850 nm - Near infrared
    1550e-9,   # 1550 nm - Telecom wavelength
    10.6e-6    # 10.6 μm - Far infrared
]

# Grid parameters
GRID_SIZE = 512           # Number of pixels (512x512)
GRID_DIAMETER = 1.0       # Physical size of grid in meters

# Telescope parameters
APERTURE_DIAMETER = 0.3   # 30 cm telescope

# Turbulence parameters (Kolmogorov/von Kármán model)
R0 = 0.1                  # Fried parameter in meters (10 cm = strong turbulence)
L0 = 25                   # Outer scale in meters
NUM_LAYERS = 3            # Number of turbulence layers

# Layer altitudes and relative strengths
LAYER_ALTITUDES = [0, 5000, 10000]   # meters (ground, 5km, 10km)
LAYER_STRENGTHS = [0.6, 0.3, 0.1]    # Relative contribution (sum to 1.0)

# Focal length for telescope
FOCAL_LENGTH = 3.0        # meters

# ============================================================================
# STEP 1: Create Computational Grid
# ============================================================================

def create_grid():
    """Create the computational grid for simulation"""
    print("Step 1: Creating computational grid...")
    pupil_grid = hcipy.make_pupil_grid(dims=GRID_SIZE, diameter=GRID_DIAMETER)
    print(f"  Grid: {GRID_SIZE}x{GRID_SIZE} points")
    print(f"  Physical size: {GRID_DIAMETER} m")
    print(f"  Sampling: {GRID_DIAMETER/GRID_SIZE*1000:.2f} mm/pixel")
    return pupil_grid

# ============================================================================
# STEP 2: Define Telescope Aperture
# ============================================================================

def create_aperture(grid):
    """Create circular telescope aperture"""
    print("\nStep 2: Creating telescope aperture...")
    aperture = hcipy.circular_aperture(APERTURE_DIAMETER)
    aperture_field = hcipy.evaluate_supersampled(aperture, grid, 4)
    print(f"  Aperture diameter: {APERTURE_DIAMETER} m")
    return aperture_field

# ============================================================================
# STEP 3: Create Initial Wavefront (Gaussian beam)
# ============================================================================

def create_wavefront(grid, aperture_field, wavelength):
    """Create initial Gaussian beam wavefront clipped by the aperture."""
    # Waist ~ half the aperture diameter (tweak as desired)
    beam_ratio = 0.5
    w0 = (APERTURE_DIAMETER * beam_ratio) / 2.0  # waist radius (meters)

    # Gaussian amplitude over the pupil grid
    r2 = grid.x**2 + grid.y**2
    gauss_amp = np.exp(-(r2) / (w0**2))

    # Clip by aperture and make a complex field (plane phase)
    ef = gauss_amp * aperture_field
    wf = hcipy.Wavefront(ef.astype(np.complex128), wavelength=wavelength)

    # Normalize power
    ef_abs2 = np.sum(np.abs(wf.electric_field)**2)
    if ef_abs2 > 0:
        wf.electric_field /= np.sqrt(ef_abs2)

    return wf

# ============================================================================
# STEP 4: Generate Turbulence Phase Screens
# ============================================================================

def generate_phase_screens(grid, wavelength):
    """Generate atmospheric turbulence phase screens using von Karman model"""
    print("\nStep 4: Generating turbulence phase screens...")

    if not ENABLE_TURBULENCE:
        print("  Turbulence DISABLED - returning empty screens")
        return []

    phase_screens = []

    for i, (altitude, strength) in enumerate(zip(LAYER_ALTITUDES, LAYER_STRENGTHS)):
        print(f"  Layer {i+1}: altitude={altitude} m, strength={strength*100:.0f}%")

        # Distribute r0 across layers so that sum_i r0_i^{-5/3} = r0_total^{-5/3}
        # -> r0_i = R0 * strength^{-3/5}
        layer_r0 = R0 * (strength ** (-3/5))

        # Build a single layer at the specified altitude.
        # Note: 'heights' must be a list; velocities can be zero for static screen.
        layers = hcipy.make_standard_atmospheric_layers(
            grid,
            r0=layer_r0,
            L0=L0,
            wavelength=wavelength,
            heights=[altitude],
            velocities=[0.0]
        )
        # Take the single layer object
        phase_screens.append(layers[0])

    print(f"  Generated {len(phase_screens)} phase screens")
    print(f"  Fried parameter r0 (total): {R0} m")
    print(f"  Outer scale L0: {L0} m")

    return phase_screens

# ============================================================================
# STEP 5: Propagate Through Turbulence
# ============================================================================

def propagate_through_turbulence(wf, phase_screens, layer_altitudes):
    """Propagate wavefront through turbulence layers"""
    print("\nStep 5: Propagating through turbulence...")

    if (not ENABLE_TURBULENCE) or len(phase_screens) == 0:
        print("  No turbulence - skipping phase screens")
        return wf

    for i, (phase_screen, altitude) in enumerate(zip(phase_screens, layer_altitudes)):
        # Apply phase screen (turbulence phase)
        phase = phase_screen.phase_for(wf.wavelength)
        wf.electric_field *= np.exp(1j * phase)
        print(f"  Applied layer {i+1} at {altitude} m")

        # Propagate to next layer (if not last layer)
        if i < len(phase_screens) - 1:
            distance = layer_altitudes[i+1] - altitude
            if distance > 0:
                # Some HCIPy builds require an explicit wavelength kwarg
                try:
                    propagator = hcipy.FresnelPropagator(wf.grid, distance)
                except TypeError:
                    propagator = hcipy.FresnelPropagator(wf.grid, distance, wavelength=wf.wavelength)
                wf = propagator(wf)
                print(f"    Propagated {distance} m to next layer")

    return wf

# ============================================================================
# STEP 6: Apply Telescope Aperture (receiver stop)
# ============================================================================

def apply_aperture(wf, aperture_field):
    """Apply telescope aperture to collect light"""
    print("\nStep 6: Applying telescope aperture...")
    wf.electric_field *= aperture_field
    collected_power = np.sum(np.abs(wf.electric_field)**2)
    print(f"  Collected power: {collected_power:.4f} (normalized)")
    return wf

# ============================================================================
# STEP 7: Propagate to Focal Plane
# ============================================================================

def propagate_to_focus(wf, focal_grid):
    """Propagate to focal plane using Fraunhofer diffraction"""
    print("\nStep 7: Propagating to focal plane...")
    try:
        propagator = hcipy.FraunhoferPropagator(wf.grid, focal_grid)
    except TypeError:
        # Some versions require explicit wavelength kwarg
        propagator = hcipy.FraunhoferPropagator(wf.grid, focal_grid, wavelength=wf.wavelength)
    focal_wf = propagator(wf)
    print("  Computed focal plane image")
    return focal_wf

# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def calculate_strehl_ratio(turbulent_intensity_2d, ideal_intensity_2d):
    """Calculate Strehl ratio (peak intensity ratio)"""
    peak_turbulent = np.max(turbulent_intensity_2d)
    peak_ideal = np.max(ideal_intensity_2d)
    return (peak_turbulent / peak_ideal) if peak_ideal > 0 else 0.0

def calculate_fwhm(centerline_profile, x_axis):
    """
    FWHM from a 1D profile and its coordinate axis.
    Returns width in the same units as x_axis (here: radians).
    """
    peak = np.max(centerline_profile)
    if peak <= 0:
        return np.nan
    half = 0.5 * peak

    i0 = int(np.argmax(centerline_profile))  # center (near the peak)
    # Find first index to the right where profile drops below half
    right = np.where(centerline_profile[i0:] <= half)[0]
    if right.size == 0:
        return np.nan
    i_half_right = i0 + right[0]

    # Assume symmetry: FWHM ≈ 2 * |x(i_half_right) - x(i0)|
    return 2.0 * abs(x_axis[i_half_right] - x_axis[i0])

# ============================================================================
# STEP 8: VISUALIZATION
# ============================================================================

def plot_results(results, wavelengths):
    """Create comprehensive comparison plots"""
    print("\n" + "="*70)
    print("STEP 8: VISUALIZATION AND ANALYSIS")
    print("="*70)

    n_wavelengths = len(wavelengths)
    fig = plt.figure(figsize=(14, 4*n_wavelengths))
    gs = GridSpec(n_wavelengths, 3, figure=fig, hspace=0.3, wspace=0.3)

    for i, wl in enumerate(wavelengths):
        wl_nm = wl * 1e9 if wl < 1e-6 else wl * 1e6
        wl_unit = 'nm' if wl < 1e-6 else 'μm'

        ideal_data = results[wl]['ideal']
        turb_data = results[wl]['turbulent']

        # 2D arrays for plotting
        ideal_img = ideal_data['intensity'].shaped
        turb_img  = turb_data['intensity'].shaped

        # Angular extent (rad) for axes
        extent = ideal_data['extent']  # radians
        extent_u = extent * 1e6        # microradians for axes

        # Strehl
        strehl = calculate_strehl_ratio(turb_img, ideal_img)

        # Centerline profiles and FWHM
        center_idx = ideal_img.shape[0] // 2
        prof_ideal = ideal_img[center_idx, :]
        prof_turb  = turb_img[center_idx, :]

        x_axis = np.linspace(-extent, extent, prof_ideal.size)  # radians
        fwhm_ideal = calculate_fwhm(prof_ideal, x_axis)
        fwhm_turb  = calculate_fwhm(prof_turb,  x_axis)

        print(f"\nWavelength: {wl_nm:.1f} {wl_unit}")
        print(f"  Strehl Ratio: {strehl:.3f}")
        print(f"  FWHM (ideal): {fwhm_ideal*1e6:.2f} μrad")
        print(f"  FWHM (turbulent): {fwhm_turb*1e6:.2f} μrad")
        print(f"  Degradation: {(1-strehl)*100:.1f}%")

        # Plot 1: Diffraction-limited focal plane
        ax1 = fig.add_subplot(gs[i, 0])
        im1 = ax1.imshow(
            ideal_img,
            extent=[-extent_u, extent_u, -extent_u, extent_u],
            cmap='hot', origin='lower'
        )
        ax1.set_title(f'{wl_nm:.1f} {wl_unit} - Diffraction Limited\nStrehl: 1.00')
        ax1.set_xlabel('x (μrad)')
        ax1.set_ylabel('y (μrad)')
        plt.colorbar(im1, ax=ax1, label='Intensity')

        # Plot 2: Turbulent focal plane
        ax2 = fig.add_subplot(gs[i, 1])
        im2 = ax2.imshow(
            turb_img,
            extent=[-extent_u, extent_u, -extent_u, extent_u],
            cmap='hot', origin='lower'
        )
        ax2.set_title(f'{wl_nm:.1f} {wl_unit} - With Turbulence (r₀={R0*100:.0f} cm)\nStrehl: {strehl:.3f}')
        ax2.set_xlabel('x (μrad)')
        ax2.set_ylabel('y (μrad)')
        plt.colorbar(im2, ax=ax2, label='Intensity')

        # Plot 3: Radial (centerline) profiles
        ax3 = fig.add_subplot(gs[i, 2])
        ax3.plot(x_axis*1e6, prof_ideal/np.max(prof_ideal), 'b-', lw=2, label='Diffraction-limited')
        ax3.plot(x_axis*1e6, prof_turb /np.max(prof_ideal),  'r-', lw=2, label='With turbulence')
        ax3.set_xlabel('Angle (μrad)')
        ax3.set_ylabel('Normalized Intensity')
        ax3.set_title('Centerline Profile')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim([0, 1.1])

    plt.suptitle(
        f'FSOC Simulation: Turbulence Impact on Different Wavelengths\n'
        f'Aperture: {APERTURE_DIAMETER} m, Turbulence: r₀={R0} m, {NUM_LAYERS} layers',
        fontsize=14, fontweight='bold'
    )
    return fig

# ============================================================================
# MAIN SIMULATION
# ============================================================================

def run_simulation(wavelength, enable_turbulence_local):
    """Run complete simulation for one wavelength"""
    wl_nm = wavelength * 1e9 if wavelength < 1e-6 else wavelength * 1e6
    wl_unit = 'nm' if wavelength < 1e-6 else 'μm'

    print("\n" + "="*70)
    print(f"SIMULATING: {wl_nm:.1f} {wl_unit} - "
          f"{'WITH TURBULENCE' if enable_turbulence_local else 'DIFFRACTION LIMITED'}")
    print("="*70)

    # Step 1: Grid
    pupil_grid = create_grid()

    # Step 2: Aperture
    aperture_field = create_aperture(pupil_grid)

    # Step 3: Initial wavefront
    print(f"\nStep 3: Creating initial wavefront...")
    print(f"  Wavelength: {wl_nm:.1f} {wl_unit}")
    wf = create_wavefront(pupil_grid, aperture_field, wavelength)

    # Step 4: Phase screens (only if turbulence enabled)
    global ENABLE_TURBULENCE
    temp_enable = ENABLE_TURBULENCE
    ENABLE_TURBULENCE = enable_turbulence_local
    phase_screens = generate_phase_screens(pupil_grid, wavelength)
    ENABLE_TURBULENCE = temp_enable

    # Step 5: Propagate through turbulence
    wf = propagate_through_turbulence(wf, phase_screens, LAYER_ALTITUDES)

    # Step 6: Apply aperture (receiver stop)
    wf = apply_aperture(wf, aperture_field)

    # Step 7: Propagate to focus
    focal_length = FOCAL_LENGTH
    q = focal_length / wavelength

    # Some HCIPy versions want 'pupil_diameter' (not 'pupil_grid')
    try:
        focal_grid = hcipy.make_focal_grid(q=q, num_airy=20, pupil_diameter=APERTURE_DIAMETER)
    except TypeError:
        # Fallback in case your local API differs
        focal_grid = hcipy.make_focal_grid(q=q, num_airy=20, pupil_diameter=APERTURE_DIAMETER)

    focal_wf = propagate_to_focus(wf, focal_grid)

    # Extract results
    intensity = focal_wf.intensity
    extent = focal_grid.x.max() / focal_length  # Angular extent [rad]
    return {
        'intensity': intensity,
        'grid': focal_grid,
        'extent': extent,
        'wavefront': focal_wf
    }


def main():
    """Main execution function"""
    print("\n" + "#"*70)
    print("#" + " "*68 + "#")
    print("#  FREE-SPACE OPTICAL COMMUNICATION (FSOC) SIMULATION".center(70) + "#")
    print("#  Atmospheric Turbulence Effects".center(70) + "#")
    print("#" + " "*68 + "#")
    print("#"*70)

    results = {}

    for wl in WAVELENGTHS:
        results[wl] = {}
        # Diffraction-limited case (no turbulence)
        results[wl]['ideal'] = run_simulation(wl, enable_turbulence_local=False)
        # With turbulence
        results[wl]['turbulent'] = run_simulation(wl, enable_turbulence_local=True)

    fig = plot_results(results, WAVELENGTHS)

    print("\n" + "="*70)
    print("SIMULATION COMPLETE")
    print("="*70)
    print("\nKey Findings:")
    print("- Shorter wavelengths are more affected by turbulence")
    print("- Longer wavelengths have larger diffraction-limited spots")
    print("- 1550 nm offers good balance for FSOC applications")
    print("\nClose the plot window to exit.")

    plt.show()

# ============================================================================
# RUN SIMULATION
# ============================================================================

if __name__ == "__main__":
    main()
