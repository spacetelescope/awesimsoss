==========
awesimsoss
==========


.. image:: https://img.shields.io/pypi/v/awesimsoss.svg
        :target: https://pypi.python.org/pypi/awesimsoss

.. image:: https://img.shields.io/travis/hover2pi/awesimsoss.svg
        :target: https://travis-ci.com/hover2pi/awesimsoss

.. image:: https://readthedocs.org/projects/awesimsoss/badge/?version=latest
        :target: https://awesimsoss.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status

.. image:: https://coveralls.io/repos/github/spacetelescope/awesimsoss/badge.svg?branch=master
        :target: https://coveralls.io/github/spacetelescope/awesimsoss?branch=master

.. image:: https://pyup.io/repos/github/hover2pi/awesimsoss/shield.svg
     :target: https://pyup.io/repos/github/hover2pi/awesimsoss/
     :alt: Updates



Advanced Webb Exposure SIMulator for SOSS
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Authors: Joe Filippazzo, Kevin Volk, Jonathan Fraine, Michael Wolfe

This pure Python 3.5+ package produces simulated TSO data for the Single
Object Slitless Spectroscopy (SOSS) mode of the NIRISS instrument
onboard the James Webb Space Telescope.

Additional resources:

- `Full documentation <https://awesimsoss.readthedocs.io/en/latest/>`_
- `Jupyter notebook <https://github.com/spacetelescope/awesimsoss/blob/master/notebooks/awesimsoss_demo.ipynb>`_
- `Build history <https://travis-ci.com/hover2pi/awesimsoss>`_

Simulating SOSS Observations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Given a 1D spectrum of a target, this module produces a 2D SOSS ramp
image with the given number of groups and integrations. For example, if
I want to produce 20 integrations of 5 groups each for a J=9 A0 star as
seen through SOSS, my code might look like:

.. code:: python

   # Imports
   import numpy as np
   from awesimsoss import TSO
   import astropy.units as q
   from pkg_resources import resource_filename
   star = np.genfromtxt(resource_filename('awesimsoss','files/scaled_spectrum.txt'), unpack=True)
   star1D = [star[0]*q.um, (star[1]*q.W/q.m**2/q.um).to(q.erg/q.s/q.cm**2/q.AA)]

   # Initialize simulation
   tso = TSO(ngrps=3, nints=5, star=star1D)
               
   # Run it and make a plot
   tso.simulate()
   tso.plot()

.. figure:: awesimsoss/img/2D_star.png
   :alt: The output trace

The SUBSTRIP256 subarray is the default but the SUBSTRIP96 subarray and
FULL frame configurations are also supported:

.. code:: python

   tso96 = TSO(ngrps=3, nints=5, star=star1D, subarray='SUBSTRIP96')
   tso2048 = TSO(ngrps=3, nints=5, star=star1D, subarray='FULL')

The default filter is CLEAR but you can also simulate observations with
the F277W filter like so:

.. code:: python

   tso = TSO(ngrps=3, nints=5, star=star1D, filt='F277W')

Simulated Planetary Transits
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The example above was for an isolated star though. To include a
planetary transit we must additionally provide a transmission spectrum
and the orbital parameters of the planet.

Here is a sample transmission spectrum generated with PANDEXO:

.. code:: python

   planet_file = resource_filename('awesimsoss', '/files/WASP107b_pandexo_input_spectrum.dat')
   planet1D = np.genfromtxt(planet_file, unpack=True)

.. figure:: awesimsoss/img/1D_planet.png
   :alt: The input transmission spectrum

And here are some parameters for our planetary system:

.. code:: python

   # Simulate star with transiting exoplanet by including transmission spectrum and orbital params
   import batman
   tso = TSO(ngrps=3, nints=5, star=star1D, run=False)
   params = batman.TransitParams()
   params.t0 = 0. # time of inferior conjunction
   params.per = 5.7214742 # orbital period (days)
   params.a = 0.0558\* q.AU.to(q.R_sun)\* 0.66 # semi-major axis (in units of stellar radii)
   params.rp = 0.1 # radius ratio for Jupiter orbiting the Sun
   params.inc = 89.8 # orbital inclination (in degrees)
   params.ecc = 0. # eccentricity
   params.w = 90. # longitude of periastron (in degrees) p
   params.limb_dark = 'quadratic' # limb darkening profile to use
   params.u = [0.1,0.1] # limb darkening coefficients
   tmodel = batman.TransitModel(params, tso.time)
   tmodel.teff = 3500 # effective temperature of the host star
   tmodel.logg = 5 # log surface gravity of the host star
   tmodel.feh = 0 # metallicity of the host star

Now the code to generate a simulated planetary transit around our star might look like:

.. code:: python

   tso.simulate(planet=planet1D, tmodel=tmodel, time_unit='seconds')
   tso.plot_lightcurve(column=42)

We can write this to a FITS file directly ingestible by the JWST pipeline with:

.. code:: python

   tso.to_fits('my_SOSS_simulation.fits')
