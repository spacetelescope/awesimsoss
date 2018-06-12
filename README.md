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

First, some imports:

```python
import astropy.units as q
import astropy.constants as ac
import os
import numpy as np
import batman
import AWESim_SOSS
from AWESim_SOSS.sim2D import awesim
DIR_PATH = os.path.dirname(os.path.realpath(AWESim_SOSS.__file__))
```

Given a 1D spectrum of a target, this module produces a 2D SOSS ramp image with the given number of groups and integrations. For example, if I want to produce 20 integrations of 5 groups each for a J=9 A0 star as seen through SOSS, my code might look like:

```python
star1D = np.genfromtxt(DIR_PATH+'/files/scaled_spectrum.txt', unpack=True)
star1D = [star1D[0]*q.um, (star1D[1]*q.W/q.m**2/q.um).to(q.erg/q.s/q.cm**2/q.AA)]
tso = awesim.TSO(ngrps=5, nints=20, star=star1D)
tso.run_simulation()
tso.plot_frame()
```

![output](AWESim_SOSS/img/2D_star.png "The output trace")

The 96 subarray is also supported:

```python
tso = awesim.TSO(ngrps=5, nints=20, star=star1D, subarray='SUBSTRIP96')
```

The default filter is CLEAR but you can also simulate observations with the F277W filter like so:

```python
tso.run_simulation(filt='F277W')
```

### Simulated Planetary Transits

The example above was for an isolated star though. To include a planetary transit we must additionally provide a transmission spectrum and the orbital parameters of the planet.

Here is a sample transmission spectrum generated with PANDEXO:

```python
planet1D = np.genfromtxt(DIR_PATH+'/files/WASP107b_pandexo_input_spectrum.dat', unpack=True)
````

![planet_input](AWESim_SOSS/img/1D_planet.png "The input transmission spectrum")

And here are some parameters for our planetary system:

```python
params = batman.TransitParams()
params.t0 = 0.                                # time of inferior conjunction
params.per = 5.7214742                        # orbital period (days)
params.a = 0.0558*q.AU.to(ac.R_sun)*0.66      # semi-major axis (in units of stellar radii)
params.inc = 89.8                             # orbital inclination (in degrees)
params.ecc = 0.                               # eccentricity
params.w = 90.                                # longitude of periastron (in degrees)
params.teff = 3500                            # effective temperature of the host star
params.logg = 5                               # log surface gravity of the host star
params.feh = 0                                # metallicity of the host star
```

Now the code to generate a simulated planetary transit around our star might look like:

```python
tso.run_simulation(planet=planet1D, params=params)
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
