import hcipy
import numpy as np

def cn_squared_from_r0(r0, wavelength):
    """
    Calculates the integrated Cn^2 value from the Fried parameter r0.
    This is a convenience wrapper around hcipy's utility function.
    """
    return hcipy.Cn_squared_from_fried_parameter(r0, wavelength)

def build_site(pupil_grid, site_parameters, L0=40.0):
    """
    Builds a list of atmospheric layers for a custom site.

    Parameters
    ----------
    pupil_grid : Grid
        The pupil grid on which the layers are defined.
    site_parameters : list of dicts
        A list where each dictionary describes an atmospheric layer.
        Each dictionary should contain:
        - 'altitude' : float, altitude of the layer in meters.
        - 'Cn_squared' : float, the structure constant of the refractive index fluctuations.
        - 'L0' : float, the outer scale of the turbulence in meters.
        - 'velocity' : array-like, the velocity vector of the layer in m/s.

    Returns
    -------
    list
        A list of hcipy.InfiniteAtmosphericLayer objects.
    """
    layers = []
    for params in site_parameters:
        phase_screen = hcipy.InfiniteAtmosphericLayer(
            pupil_grid,
            Cn_squared=params['Cn_squared'],
            L0=params['outer_scale'],
            velocity=params['velocity']
        )
        # InfiniteAtmosphericLayer already encapsulates frozen-flow dynamics; set altitude directly.
        phase_screen.height = params['altitude']
        layers.append(phase_screen)
    return layers

def build_multilayer_screen(layers, scintillation=True):
    """
    Builds a multi-layer atmospheric screen from a list of layers.

    Parameters
    ----------
    layers : list
        A list of hcipy.AtmosphericLayer objects.
    scintillation : bool
        Whether to enable scintillation effects.

    Returns
    -------
    hcipy.MultiLayerAtmosphere
        The multi-layer atmosphere object.
    """
    return hcipy.MultiLayerAtmosphere(layers, scintillation=scintillation)
