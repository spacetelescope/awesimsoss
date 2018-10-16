import matplotlib
matplotlib.use('Qt5Agg')

import matplotlib.pyplot as plt

from pylab import *;ion()

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

# From https://www.cfa.harvard.edu/~lkreidberg/batman/quickstart.html
params = batman.TransitParams()
params.t0 = 0.                       #time of inferior conjunction
params.per = 1.                      #orbital period
params.rp = 0.1                      #planet radius (in units of stellar radii)
params.a = 15.                       #semi-major axis (in units of stellar radii)
params.inc = 87.                     #orbital inclination (in degrees)
params.ecc = 0.                      #eccentricity
params.w = 90.                       #longitude of periastron (in degrees)

# Added here
params.u = [0.1, 0.3]                #limb darkening coefficients [u1, u2]
params.limb_dark = "quadratic"       #limb darkening model
params.u = [0.1,0.1]                          # limb darkening coefficients
tmodel = batman.TransitModel(params, tso.time/day2sec)
tmodel.teff = 3500                            # effective temperature of the host star
tmodel.logg = 5                               # log surface gravity of the host star
tmodel.feh = 0                                # metallicity of the host star

tso.run_simulation(planet=planet1D, tmodel=tmodel)
tso.plot_lightcurve(column=range(10,2048,500))

# plt.show()
