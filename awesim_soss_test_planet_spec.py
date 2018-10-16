import matplotlib
matplotlib.use('Qt5Agg')

import matplotlib.pyplot as plt

import numpy as np
from AWESim_SOSS.sim2D import awesim
import astropy.units as q
import astropy.constants as ac
import batman
from pkg_resources import resource_filename

day2sec = 86400

star = np.genfromtxt(resource_filename('AWESim_SOSS','files/scaled_spectrum.txt'), unpack=True)
star1D = [star[0]*q.um, (star[1]*q.W/q.m**2/q.um).to(q.erg/q.s/q.cm**2/q.AA)]


tso = awesim.TSO(ngrps=3, nints=5, star=star1D)

planet_file = resource_filename('AWESim_SOSS', '/files/WASP107b_pandexo_input_spectrum.dat')
planet1D = np.genfromtxt(planet_file, unpack=True)

params = batman.TransitParams()
params.t0 = 0.                                # time of inferior conjunction
params.per = 5.7214742                        # orbital period (days)
params.a = 0.0558*q.AU.to(ac.R_sun)*0.66      # semi-major axis (in units of stellar radii)
params.rp = 0.1                               # radius ratio for Jupiter orbiting the Sun
params.inc = 89.8                             # orbital inclination (in degrees)
params.ecc = 0.                               # eccentricity
params.w = 90.                                # longitude of periastron (in degrees)
params.limb_dark = 'quadratic'                # limb darkening profile to use
params.u = [0.1,0.1]                          # limb darkening coefficients
tmodel = batman.TransitModel(params, tso.time/day2sec)
tmodel.teff = 3500                            # effective temperature of the host star
tmodel.logg = 5                               # log surface gravity of the host star
tmodel.feh = 0                                # metallicity of the host star

tso.run_simulation(planet=planet1D, tmodel=tmodel, time_unit='seconds')
tso.plot_lightcurve(column=42)

plt.show()
