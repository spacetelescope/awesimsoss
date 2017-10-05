# AWESim SOSS

### Analyzing Webb Exoplanet Simulations with SOSS

Authors: Joe Filippazzo and Jonathan Fraine

This pure Python module produces simulated data for the Single Object Slitless Spectroscopy (SOSS) mode of the NIRISS instrument onboard the James Webb Space Telescope.

### Dependencies
The following packages are needed to run `AWESim_SOSS`:
- numpy
- batman
- astropy

### Usage

Given a time axis and a 1D spectrum of a target, this module produces a 2D SOSS slope image for each point on the time axis. For example, if I want to produce 200 integrations of WASP-107 (no planet) as seen through SOSS, my code might look like:

```
import numpy as np
from AWESim_SOSS import awesim
t = np.linspace(-0.2, 0.2, 200)
WASP107 = np.genfromtxt('AWESim_SOSS/data/WASP107.txt', unpack=True)
TSO = awesim.TSO(t, WASP107)
TSO.plot_frame()
```

Here is the input spectrum and one frame of the output data cube:

![input](AWESim_SOSS/img/1D_spec.png "The input spectrum")

![output](AWESim_SOSS/img/2D_spec.png "The output trace")

The example above was for an isolated star though. To include a planetary transit we must additionally provide:

- A transmission spectrum of the planet
- A map of the limb darkening coefficients at each pixel
- The orbital parameters of the planet

Here is a sample transmission spectrum generated with PANDEXO:

```
WASP107b = np.genfromtxt('AWESim_SOSS/data/WASP107b_pandexo_input_spectrum.dat', unpack=True)
````

![planet](AWESim_SOSS/img/1D_planet.png "Planet")

And here are some orbital parameters for WASP-107:

```
import batman
params = batman.TransitParams()
params.t0 = 0.                                # time of inferior conjunction
params.per = 5.7214742                        # orbital period
params.a = 0.0558*q.AU.to(ac.R_sun)*0.66      # semi-major axis (in units of stellar radii)
params.inc = 89.8                             # orbital inclination (in degrees)
params.ecc = 0.                               # eccentricity
params.w = 90.                                # longitude of periastron (in degrees)
```

Now the code to create 200 integrations which contain a planetary transit might look like:

```
TSO = awesim.TSO(t, WASP107, WASP107b, params, ld_coeffs)
```

We can verify that the lightcurves are wavelength dependent by plotting a few different columns of the SOSS trace like so:

```
TSO.plot_lightcurve([15,150,300])
```

![lightcurves](AWESim_SOSS/img/lc.png "lightcurves")
