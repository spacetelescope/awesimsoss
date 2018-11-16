#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for `awesim` module."""

from pkg_resources import resource_filename

import numpy as np
import astropy.units as q
import astropy.constants as ac
import batman

from awesimsoss import TSO


def test_awesim_tso():
    """Test that a TSO simulation runs"""
    # Get the star data
    star_file = resource_filename('awesimsoss', 'files/scaled_spectrum.txt')
    star = np.genfromtxt(star_file, unpack=True)
    star1D = [star[0]*q.um, (star[1]*q.W/q.m**2/q.um).to(q.erg/q.s/q.cm**2/q.AA)]

    # Make the TSO object
    tso = TSO(ngrps=3, nints=5, star=star1D)

    # Get the planet data
    planet_file = resource_filename('awesimsoss', '/files/WASP107b_pandexo_input_spectrum.dat')
    planet1D = np.genfromtxt(planet_file, unpack=True)

    # Set the orbital params
    # From https://www.cfa.harvard.edu/~lkreidberg/batman/quickstart.html
    params = batman.TransitParams()
    params.t0 = 0.                  # Time of inferior conjunction
    params.per = 1.                 # Orbital period
    params.rp = 0.1                 # Planet radius (in units of R*)
    params.a = 15.                  # Semi-major axis (in units of R*)
    params.inc = 87.                # Orbital inclination (in degrees)
    params.ecc = 0.                 # Eccentricity
    params.w = 90.                  # Longitude of periastron (in degrees) 
    params.u = [0.1, 0.1]           # Limb darkening coefficients [u1, u2]
    params.limb_dark = "quadratic"  # Limb darkening model

    # Make the transit model and add the stellar params
    day2sec = 86400
    tmodel = batman.TransitModel(params, tso.time/day2sec)
    tmodel.teff = 3500              # Effective temperature of the host star
    tmodel.logg = 5                 # log surface gravity of the host star
    tmodel.feh = 0                  # Metallicity of the host star

    # Run the simulation
    tso.run_simulation(planet=planet1D, tmodel=tmodel)

    # Plot the lightcurve
    tso.plot_lightcurve(column=range(10,2048,500))
