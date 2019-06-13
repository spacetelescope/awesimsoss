# -*- coding: utf-8 -*-
"""
A module to of utilities to assist awesim.py

Authors: Joe Filippazzo, Kevin Volk, Jonathan Fraine, Michael Wolfe
"""

import itertools
from pkg_resources import resource_filename

from astropy.io import fits
import astropy.units as q
from bokeh.palettes import Category20
import numpy as np


def color_gen():
    yield from itertools.cycle(Category20[20])


COLORS = color_gen()
STAR_DATA = star_data()
PLANET_DATA = planet_data()


def planet_data():
    """
    Dummy data for time series simulations
    """
    planet_file = resource_filename('awesimsoss', '/files/WASP107b_pandexo_input_spectrum.dat')
    planet1D = np.genfromtxt(planet_file, unpack=True)

    return planet1D


def star_data():
    """
    Dummy data for time series simulations
    """
    star = np.genfromtxt(resource_filename('awesimsoss', 'files/scaled_spectrum.txt'), unpack=True)
    star1D = [star[0]*q.um, (star[1]*q.W/q.m**2/q.um).to(q.erg/q.s/q.cm**2/q.AA)]

    return star1D


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
