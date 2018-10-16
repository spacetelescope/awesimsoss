# AWESim SOSS

### Advanced Webb Exposure SIMulator for SOSS

Authors: Joe Filippazzo, Kevin Volk, Jonathan Fraine, Michael Wolfe

This pure Python module produces simulated data for the Single Object Slitless Spectroscopy (SOSS) mode of the NIRISS instrument onboard the James Webb Space Telescope.

### Dependencies
The following packages are needed to run `AWESim_SOSS`:
- numpy
- batman
- astropy

### Simulating SOSS Observations

Given a 1D spectrum of a target, this module produces a 2D SOSS ramp image with the given number of groups and integrations. For example, if I want to produce 20 integrations of 5 groups each for a J=9 A0 star as seen through SOSS, my code might look like:

```python
# Imports
import numpy as np
from AWESim_SOSS.sim2D import awesim
import astropy.units as q
import astropy.constants as ac
import batman
from pkg_resources import resource_filename
star = np.genfromtxt(resource_filename('AWESim_SOSS','files/scaled_spectrum.txt'), unpack=True)
star1D = [star[0]*q.um, (star[1]*q.W/q.m**2/q.um).to(q.erg/q.s/q.cm**2/q.AA)]

# Initialize simulation
tso = awesim.TSO(ngrps=3, nints=5, star=star1D)
            
# Run it and make a plot
tso.run_simulation()
tso.plot_frame()
```

![output](AWESim_SOSS/img/2D_star.png "The output trace")

The 96 subarray is also supported:

```python
tso = awesim.TSO(ngrps=3, nints=5, star=star1D, subarray='SUBSTRIP96')
```

The default filter is CLEAR but you can also simulate observations with the F277W filter like so:

```python
tso = awesim.TSO(ngrps=3, nints=5, star=star1D, filt='F277W')
```

### Simulated Planetary Transits

The example above was for an isolated star though. To include a planetary transit we must additionally provide a transmission spectrum and the orbital parameters of the planet.

Here is a sample transmission spectrum generated with PANDEXO:

```python
planet_file = resource_filename('AWESim_SOSS', '/files/WASP107b_pandexo_input_spectrum.dat')
planet1D = np.genfromtxt(planet_file, unpack=True)
````

![planet_input](AWESim_SOSS/img/1D_planet.png "The input transmission spectrum")

And here are some parameters for our planetary system:

```python
# Simulate star with transiting exoplanet by including transmission spectrum and orbital params
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
tmodel = batman.TransitModel(params, tso.time)
tmodel.teff = 3500                            # effective temperature of the host star
tmodel.logg = 5                               # log surface gravity of the host star
tmodel.feh = 0                                # metallicity of the host star
```

Now the code to generate a simulated planetary transit around our star might look like:

```python
tso.run_simulation(planet=planet1D, tmodel=tmodel, time_unit='seconds')
tso.plot_lightcurve(col=42)
```

We can write this to a FITS file directly ingestible by the JWST pipeline with:

```python
tso.to_fits('my_SOSS_simulation.fits')
```

<!--We can verify that the lightcurves are wavelength dependent by plotting a few different columns of the SOSS trace like so:

```python
TSO.plot_lightcurve([15,150,300])
```

![lightcurves](AWESim_SOSS/img/lc.png "lightcurves") -->
