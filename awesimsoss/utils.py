# -*- coding: utf-8 -*-
"""
A module to of utilities to assist awesim.py

Authors: Joe Filippazzo, Kevin Volk, Jonathan Fraine, Michael Wolfe
"""

import itertools
from pkg_resources import resource_filename

from astropy.io import fits
import astropy.units as q
import batman
from bokeh.palettes import Category20
import numpy as np


def color_gen():
    """Generator for a Bokeh color palette"""
    yield from itertools.cycle(Category20[20])


def planet_data():
    """
    Dummy data for time series simulations

    Returns
    -------
    sequence
        The wavelength and atmospheric transmission of the planet
    """
    planet_file = resource_filename('awesimsoss', '/files/WASP107b_pandexo_input_spectrum.dat')
    planet = np.genfromtxt(planet_file, unpack=True)
    planet1D = [planet[0]*q.um, planet[1]]

    return planet1D


def star_data():
    """
    Dummy data for time series simulations

    Returns
    -------
    sequence
        The wavelength and flux of the star
    """
    star_file = resource_filename('awesimsoss', 'files/scaled_spectrum.txt')
    star = np.genfromtxt(star_file, unpack=True)
    star1D = [star[0]*q.um, (star[1]*q.W/q.m**2/q.um).to(q.erg/q.s/q.cm**2/q.AA)]

    return star1D


def transit_params(time):
    """
    Dummy transit parameters for time series simulations

    Parameters
    ----------
    time: sequence
        The time axis of the transit observation

    Returns
    -------
    batman.transitmodel.TransitModel
        The transit model
    """
    params = batman.TransitParams()
    params.t0 = 0.                                # time of inferior conjunction
    params.per = 5.7214742                        # orbital period (days)
    params.a = 0.0558*q.AU.to(q.R_sun)*0.66      # semi-major axis (in units of stellar radii)
    params.inc = 89.8                             # orbital inclination (in degrees)
    params.ecc = 0.                               # eccentricity
    params.w = 90.                                # longitude of periastron (in degrees)
    params.limb_dark = 'quadratic'                # limb darkening profile to use
    params.u = [0.1, 0.1]                          # limb darkening coefficients
    params.rp = 0.                                # planet radius (placeholder)
    tmodel = batman.TransitModel(params, time)
    tmodel.teff = 3500                            # effective temperature of the host star
    tmodel.logg = 5                               # log surface gravity of the host star
    tmodel.feh = 0                                # metallicity of the host star

    return tmodel


COLORS = color_gen()
STAR_DATA = star_data()
PLANET_DATA = planet_data()

def subarray(subarr):
    """
    Get the pixel information for a NIRISS subarray

    The returned dictionary defines the extent ('x' and 'y'),
    the starting pixel ('xloc' and 'yloc'), and the number
    of reference pixels at each subarray edge ('x1', 'x2',
    'y1', 'y2) as defined by SSB/DMS coordinates shown below:
        ___________________________________
       |               y2                  |
       |                                   |
       |                                   |
       | x1                             x2 |
       |                                   |
       |               y1                  |
       |___________________________________|
    (1,1)

    Parameters
    ----------
    subarr: str
        The subarray name

    Returns
    -------
    dict
        The dictionary of the specified subarray
        or a nested dictionary of all subarrays
    """
    pix = {'FULL': {'xloc': 1, 'x': 2048, 'x1': 4, 'x2': 4, 'yloc': 1, 'y': 2048, 'y1': 4, 'y2': 4, 'tfrm': 10.737, 'tgrp': 10.737},
           'SUBSTRIP96': {'xloc': 1, 'x': 2048, 'x1': 4, 'x2': 4, 'yloc': 1803, 'y': 96, 'y1': 0, 'y2': 0, 'tfrm': 2.213, 'tgrp': 2.213},
           'SUBSTRIP256': {'xloc': 1, 'x': 2048, 'x1': 4, 'x2': 4, 'yloc': 1793, 'y': 256, 'y1': 0, 'y2': 4, 'tfrm': 5.491, 'tgrp': 5.491}}

    return pix[subarr]


def wave_solutions(subarr=None, order=None, directory=None):
    """
    Get the wavelength maps for SOSS orders 1, 2, and 3
    This will be obsolete once the apply_wcs step of the JWST pipeline
    is in place.

    Parameters
    ----------
    subarr: str
        The subarray to return, ['SUBSTRIP96', 'SUBSTRIP256', 'FULL']
    order: int (optional)
        The trace order, [1, 2, 3]
    directory: str
        The directory containing the wavelength FITS files

    Returns
    -------
    np.ndarray
        An array of the wavelength solutions for orders 1, 2, and 3
    """
    # Get the directory
    if directory is None:
        default = '/files/soss_wavelengths_fullframe.fits'
        directory = resource_filename('awesimsoss', default)

    # Trim to the correct subarray
    if subarr == 'SUBSTRIP256':
        idx = slice(0, 256)
    elif subarr == 'SUBSTRIP96':
        idx = slice(160, 256)
    else:
        idx = slice(0, 2048)

    # Select the right order
    if order in [1, 2]:
        order = int(order)-1
    else:
        order = slice(0, 3)

    # Get the data from file and trim
    wave = fits.getdata(directory).swapaxes(-2, -1)[order, idx, ::-1]

    return wave
