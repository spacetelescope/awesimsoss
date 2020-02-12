# -*- coding: utf-8 -*-
"""
A module to generate simulated 2D time-series SOSS data

Authors: Joe Filippazzo, Kevin Volk, Jonathan Fraine, Michael Wolfe
"""
from copy import copy
import datetime
from functools import partial, wraps
from multiprocessing.pool import ThreadPool
from multiprocessing import cpu_count
import os
from pkg_resources import resource_filename
import time
import warnings

import astropy.units as q
import astropy.constants as ac
from astropy.io import fits
from astropy.modeling.models import BlackBody1D, Voigt1D, Gaussian1D, Lorentz1D
from astropy.modeling.blackbody import FLAM
import astropy.table as at
from astropy.coordinates import SkyCoord
from astroquery.simbad import Simbad
import batman
from bokeh.plotting import figure, show
from bokeh.models import HoverTool, LogColorMapper, LogTicker, LinearColorMapper, ColorBar, Span
from bokeh.layouts import column
from hotsoss import utils, plotting, locate_trace
import numpy as np

try:
    from jwst.datamodels import RampModel
except ImportError:
    print("Could not import `jwst` package. Functionality limited.")

from . import generate_darks as gd
from . import make_trace as mt

# Some extra imports for PHOENIX model downloads:
import zipfile
import shutil
import glob
import urllib.request as request
from contextlib import closing

# Interpolation tools:
from scipy import interpolate,ndimage


warnings.simplefilter('ignore')


def check_psf_files():
    """Function to run on import to verify that the PSF files have been precomputed"""
    if not os.path.isfile(resource_filename('awesimsoss', 'files/SOSS_CLEAR_PSF_order1_1.npy')):
        print("Looks like you haven't generated the SOSS PSFs yet, which are required to produce simulations.")
        print("This takes about 10 minutes but you will only need to do it this one time.")
        compute = input("Would you like to do it now? [y] ")

        if compute is None or compute.lower() in ['y', 'yes']:
            mt.nuke_psfs()


def run_required(func):
    """A wrapper to check that the simulation has been run before a method can be executed"""
    @wraps(func)
    def _run_required(*args, **kwargs):
        """Check that the 'tso' attribute is not None"""
        if args[0].tso is None:
            print("No simulation found! Please run the 'simulate' method first.")

        else:
            return func(*args, **kwargs)

    return _run_required


check_psf_files()


class TSO(object):
    """
    Generate NIRISS SOSS time series observations
    """
    def __init__(self, ngrps, nints, star=None, planet=None, tmodel=None, snr=700,
                 filter='CLEAR', subarray='SUBSTRIP256', orders=[1, 2], t0=0, nresets=0,
                 target='New Target', title=None, verbose=True):
        """
        Initialize the TSO object and do all pre-calculations

        Parameters
        ----------
        ngrps: int
            The number of groups per integration
        nints: int
            The number of integrations for the exposure
        star: sequence
            The wavelength and flux of the star
        planet: sequence
            The wavelength and transmission of the planet
        snr: float
            The signal-to-noise
        filter: str
            The name of the filter to use, ['CLEAR', 'F277W']
        subarray: str
            The name of the subarray to use, ['SUBSTRIP256', 'SUBSTRIP96', 'FULL']
        orders: int, list
            The orders to simulate, [1], [1, 2], [1, 2, 3]
        t0: float
            The start time of the exposure [days]
        nresets: int
            The number of resets before each integration
        target: str (optional)
            The name of the target
        title: str (optionl)
            A title for the simulation
        verbose: bool
            Print status updates throughout calculation

        Example
        -------
        # Imports
        import numpy as np
        from awesimsoss import TSO, STAR_DATA
        import astropy.units as q
        from pkg_resources import resource_filename
        star = np.genfromtxt(resource_filename('awesimsoss', 'files/scaled_spectrum.txt'), unpack=True)
        star1D = [star[0]*q.um, (star[1]*q.W/q.m**2/q.um).to(q.erg/q.s/q.cm**2/q.AA)]

        # Initialize simulation
        tso = TSO(ngrps=3, nints=10, star=star1D)
        """
        # Metadata
        self.verbose = verbose
        self.target = target
        self.title = title or '{} Simulation'.format(self.target)

        # Set static values
        self.gain = 1.61
        self._star = None

        # Set instance attributes for the exposure
        self.t0 = t0
        self.ngrps = ngrps
        self.nints = nints
        self.nresets = nresets
        self.nframes = (self.nresets+self.ngrps)*self.nints
        self.obs_date = datetime.datetime.now().strftime("%Y-%m-%d")
        self.obs_time = datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]
        self.orders = orders
        self.filter = filter
        self.header = ''
        self.snr = snr
        self.model_grid = None
        self.subarray = subarray

        # Set instance attributes for the target
        self.lines = at.Table(names=('name', 'profile', 'x_0', 'amp', 'fwhm', 'flux'), dtype=('S20', 'S20', float, float, 'O', 'O'))
        self.star = star
        self.tmodel = tmodel
        self.ld_coeffs = np.zeros((3, 2048, 2))
        self.ld_profile = 'quadratic'
        self.planet = planet

        # Reset data based on subarray and observation settings
        self._reset_data()
        self._reset_time()

    def add_line(self, x_0, amplitude, fwhm, profile='lorentz', name='Line I'):
        """
        Add an emission or absorption line to the spectrum

        Parameters
        ----------
        x_0: astropy.units.quantity.Quantity
            The rest wavelength of the line
        amplitude: astropy.units.quantity.Quantity
            The amplitude of the line relative to the continuum,
            with negative value for absorption and positive for emission
        fwhm: astropy.units.quantity.Quantity, sequence
            The full-width-half-max(s) of the line ('voigt' requires 2)
        profile: str
            The profile to use, ['voigt', 'lorentz', 'gaussian']
        name: str
            A name for the line
        """
        # Check the profile
        profiles = {'voigt': Voigt1D, 'gaussian': Gaussian1D, 'lorentz': Lorentz1D}
        if profile not in profiles:
            raise ValueError("'{}' profile not supported. Please select from {}".format(profile, list(profiles.keys())))

        # Select the profile
        prof = profiles[profile]

        # Convert to match star units and remove units
        x_0 = x_0.to(self.star[0].unit).value
        amplitude = amplitude.to(self.star[1].unit).value

        # Generate the line function
        if profile == 'voigt':
            if len(fwhm) != 2:
                raise TypeError("fwhm must be sequence of two values for Voigt profile.")
            else:
                fwhm_L, fwhm_G = [fw.to(self.star[0].unit).value for fw in fwhm]
            func = prof(amplitude_L=amplitude, x_0=x_0, fwhm_L=fwhm_L, fwhm_G=fwhm_G)

        elif profile == 'lorentz':
            fwhm = fwhm.to(self.star[0].unit).value
            func = prof(amplitude=amplitude, x_0=x_0, fwhm=fwhm)

        elif profile == 'gaussian':
            fwhm = fwhm.to(self.star[0].unit).value
            func = prof(amplitude=amplitude, mean=x_0, stddev=fwhm/2.355)

        # Evaluate the profile
        line = func(self.star[0].value)*self.star[1].unit

        # Add the line to the line list
        self.lines.add_row([name, profile, x_0, amplitude, fwhm, line])

        # Reset the psfs
        self._reset_psfs()

    @run_required
    def add_noise(self, zodi_scale=1., offset=500):
        """
        Generate ramp and background noise

        Parameters
        ----------
        zodi_scale: float
            The scale factor of the zodiacal background
        offset: int
            The dark current offset
        """
        print('Adding noise to TSO...')
        start = time.time()

        # Get the separated orders
        orders = [getattr(self, 'tso_order{}_ideal'.format(i)) for i in self.orders]

        # Put into 3D
        orders = np.array([order.reshape(self.dims3) for order in orders])
        tso_ideal = self.tso_ideal.reshape(self.dims3)

        # Load the reference files
        pca0_file = resource_filename('awesimsoss', 'files/niriss_pca0.fits')
        nonlinearity = fits.getdata(resource_filename('awesimsoss', 'files/forward_coefficients_dms.fits'))
        pedestal = fits.getdata(resource_filename('awesimsoss', 'files/pedestaldms.fits'))
        photon_yield = fits.getdata(resource_filename('awesimsoss', 'files/photonyieldfullframe.fits'))
        zodi = fits.getdata(resource_filename('awesimsoss', 'files/background_detectorfield_normalized.fits'))
        darksignal = fits.getdata(resource_filename('awesimsoss', 'files/signaldms.fits'))*self.gain

        # Slice of FULL frame reference files
        slc = slice(1792, 1888) if self.subarray == 'SUBSTRIP96' else slice(1792, 2048) if self.subarray == 'SUBSTRIP256' else slice(0, 2048)

        # Trim FULL frame reference files
        pedestal = pedestal[slc, :]
        nonlinearity = nonlinearity[:, slc, :]
        zodi = zodi[slc, :]
        darksignal = darksignal[slc, :]
        photon_yield = photon_yield[:, slc, :]

        # Generate the photon yield factor values
        pyf = gd.make_photon_yield(photon_yield, np.mean(orders, axis=1))

        # Remove negatives from the dark ramp
        darksignal[np.where(darksignal < 0.)] = 0.

        # Make the exposure
        RAMP = gd.make_exposure(1, self.ngrps, darksignal, self.gain, pca0_file=pca0_file, offset=offset)

        # Iterate over integrations
        tso = copy(tso_ideal)
        for n in range(self.nints):

            # Add in the SOSS signal
            ramp = gd.add_signal(tso_ideal[self.ngrps*n:self.ngrps*n+self.ngrps], RAMP.copy(), pyf, self.frame_time, self.gain, zodi, zodi_scale, photon_yield=False)

            # Apply the non-linearity function
            ramp = gd.non_linearity(ramp, nonlinearity, offset=offset)

            # Add the pedestal to each frame in the integration
            ramp = gd.add_pedestal(ramp, pedestal, offset=offset)

            # Update the TSO with one containing noise
            tso[self.ngrps*n:self.ngrps*n+self.ngrps] = ramp

        # Put into 4D
        self.tso = tso.reshape(self.dims)

        # Memory cleanup
        del RAMP, tso, tso_ideal, ramp, pyf, photon_yield, darksignal, zodi, nonlinearity, pedestal, orders

        print('Noise model finished:', round(time.time()-start, 3), 's')

    @run_required
    def add_refpix(self, counts=0):
        """Add reference pixels to detector edges

        Parameters
        ----------
        counts: int
            The number of counts or the reference pixels
        """
        # Left, right (all subarrays)
        self.tso[:, :, :, :4] = counts
        self.tso[:, :, :, -4:] = counts

        # Top (excluding SUBSTRIP96)
        if self.subarray != 'SUBSTRIP96':
            self.tso[:, :, -4:, :] = counts

        # Bottom (Only FULL frame)
        if self.subarray == 'FULL':
            self.tso[:, :, :4, :] = counts

    @run_required
    def export(self, outfile, all_data=False):
        """
        Export the simulated data to a JWST pipeline ingestible FITS file

        Parameters
        ----------
        outfile: str
            The path of the output file
        """
        # Make a RampModel
        data = self.tso
        mod = RampModel(data=data, groupdq=np.zeros_like(data), pixeldq=np.zeros((self.nrows, self.ncols)), err=np.zeros_like(data))
        pix = utils.subarray_specs(self.subarray)

        # Set meta data values for header keywords
        mod.meta.telescope = 'JWST'
        mod.meta.instrument.name = 'NIRISS'
        mod.meta.instrument.detector = 'NIS'
        mod.meta.instrument.filter = self.filter
        mod.meta.instrument.pupil = 'GR700XD'
        mod.meta.exposure.type = 'NIS_SOSS'
        mod.meta.exposure.nints = self.nints
        mod.meta.exposure.ngroups = self.ngrps
        mod.meta.exposure.nframes = self.nframes
        mod.meta.exposure.readpatt = 'NISRAPID'
        mod.meta.exposure.groupgap = 0
        mod.meta.exposure.frame_time = self.frame_time
        mod.meta.exposure.group_time = self.group_time
        mod.meta.exposure.duration = self.time[-1]-self.time[0]
        mod.meta.exposure.nresets_at_start = 1
        mod.meta.exposure.nresets_between_ints = 1
        mod.meta.subarray.name = self.subarray
        mod.meta.subarray.xsize = data.shape[3]
        mod.meta.subarray.ysize = data.shape[2]
        mod.meta.subarray.xstart = pix.get('xloc', 1)
        mod.meta.subarray.ystart = pix.get('yloc', 1)
        mod.meta.subarray.fastaxis = -2
        mod.meta.subarray.slowaxis = -1
        mod.meta.observation.date = self.obs_date
        mod.meta.observation.time = self.obs_time
        mod.meta.target.ra = self.ra
        mod.meta.target.dec = self.dec
        mod.meta.target.source_type = 'POINT'

        # Save the file
        mod.save(outfile, overwrite=True)

        # Save input data
        with fits.open(outfile) as hdul:

            # Save input star data
            hdul.append(fits.ImageHDU(data=np.array([i.value for i in self.star], dtype=np.float64), name='STAR'))
            hdul['STAR'].header.set('FUNITS', str(self.star[1].unit))
            hdul['STAR'].header.set('WUNITS', str(self.star[0].unit))

            # Save input planet data
            if self.planet is not None:
                hdul.append(fits.ImageHDU(data=np.asarray(self.planet, dtype=np.float64), name='PLANET'))
                for param, val in self.tmodel.__dict__.items():
                    if isinstance(val, (float, int, str)):
                        hdul['PLANET'].header.set(param.upper()[:8], val)
                    elif isinstance(val, np.ndarray) and len(val) == 1:
                        hdul['PLANET'].header.set(param.upper(), val[0])
                    elif isinstance(val, type(None)):
                        hdul['PLANET'].header.set(param.upper(), '')
                    elif param == 'u':
                        for n, v in enumerate(val):
                            hdul['PLANET'].header.set('U{}'.format(n+1), v)
                    else:
                        print(param, val, type(val))

            # Write to file
            hdul.writeto(outfile, overwrite=True)

        print('File saved as', outfile)

    @property
    def filter(self):
        """Getter for the filter"""
        return self._filter

    @filter.setter
    def filter(self, filt):
        """Setter for the filter

        Properties
        ----------
        filt: str
            The name of the filter to use,
            ['CLEAR', 'F277W']
        """
        # Valid filters
        filts = ['CLEAR', 'F277W']

        # Check the value
        if not isinstance(filt, str) or filt.upper() not in filts:
            raise ValueError("'{}' not a supported filter. Try {}".format(filt, filts))

        # Set it
        filt = filt.upper()
        self._filter = filt

        # If F277W, set orders to 1 to speed up calculation
        if filt == 'F277W':
            self.orders = [1]

        # Get absolute calibration reference file
        calfile = resource_filename('awesimsoss', 'files/niriss_ref_photom.fits')
        caldata = fits.getdata(calfile)
        self.photom = caldata[(caldata['pupil'] == 'GR700XD') & (caldata['filter'] == filt)]

        # Update the results
        self._reset_data()

        # Reset relative response function
        self._reset_psfs()

    @property
    def info(self):
        """Summary table for the observation settings"""
        # Pull out relevant attributes
        track = ['_ncols', '_nrows', '_nints', '_ngrps', '_nresets', '_subarray', '_filter', '_t0', '_orders', 'ld_profile', '_target', 'title', 'ra', 'dec']
        settings = {key.strip('_'): val for key, val in self.__dict__.items() if key in track}
        return settings

    @property
    def ld_coeffs(self):
        """Get the limb darkening coefficients"""
        return self._ld_coeffs

    @ld_coeffs.setter
    def ld_coeffs(self, coeffs):
        """Set the limb darkening coefficients

        Parameters
        ----------
        coeffs: str, sequence
            The limb darkening coefficients or 'update'
        """
        # Default message
        msg = "Limb darkening coefficients must be an array of 3 dimensions"

        # Update the coeffs based on the transit model parameters
        if coeffs == 'update':

            # Check the transit model
            if self.tmodel is None:
                msg = "Please set a transit model with the 'tmodel' attribute to update the limb darkening coefficients"

            # Check the model grid
            elif self.model_grid is None:
                msg = "Please set a stellar intensity model grid with the 'model_grid' attribute to update the limb darkening coefficients"

            # Generate the coefficients
            else:
                coeffs = [mt.generate_SOSS_ldcs(self.avg_wave[order-1], self.tmodel.limb_dark, [getattr(self.tmodel, p) for p in ['teff', 'logg', 'feh']], model_grid=self.model_grid) for order in self.orders]

        # Check the coefficient type
        if not isinstance(coeffs, np.ndarray) or not coeffs.ndim == 3:
            if self.verbose:
                print(msg)

        else:
            self._ld_coeffs = coeffs

    @property
    def ncols(self):
        """Getter for the number of columns"""
        return self._ncols

    @ncols.setter
    def ncols(self, err):
        """Error when trying to change the number of columns
        """
        raise TypeError("The number of columns is fixed by setting the 'subarray' attribute.")

    @property
    def ngrps(self):
        """Getter for the number of groups"""
        return self._ngrps

    @ngrps.setter
    def ngrps(self, ngrp_val):
        """Setter for the number of groups

        Properties
        ----------
        ngrp_val: int
            The number of groups
        """
        # Check the value
        if not isinstance(ngrp_val, int):
            raise TypeError("The number of groups must be an integer")

        # Set it
        self._ngrps = ngrp_val

        # Update the results
        self._reset_data()
        self._reset_time()

    @property
    def nints(self):
        """Getter for the number of integrations"""
        return self._nints

    @nints.setter
    def nints(self, nint_val):
        """Setter for the number of integrations

        Properties
        ----------
        nint_val: int
            The number of integrations
        """
        # Check the value
        if not isinstance(nint_val, int):
            raise TypeError("The number of integrations must be an integer")

        # Set it
        self._nints = nint_val

        # Update the results
        self._reset_data()
        self._reset_time()

    @property
    def nresets(self):
        """Getter for the number of resets"""
        return self._nresets

    @nresets.setter
    def nresets(self, nreset_val):
        """Setter for the number of resets

        Properties
        ----------
        nreset_val: int
            The number of resets
        """
        # Check the value
        if not isinstance(nreset_val, int):
            raise TypeError("The number of resets must be an integer")

        # Set it
        self._nresets = nreset_val

        # Update the time (data shape doesn't change)
        self._reset_time()

    @property
    def nrows(self):
        """Getter for the number of rows"""
        return self._nrows

    @nrows.setter
    def nrows(self, err):
        """Error when trying to change the number of rows
        """
        raise TypeError("The number of rows is fixed by setting the 'subarray' attribute.")

    @property
    def orders(self):
        """Getter for the orders"""
        return self._orders

    @orders.setter
    def orders(self, ords):
        """Setter for the orders

        Properties
        ----------
        ords: list
            The orders to simulate, [1, 2, 3]
        """
        # Valid order lists
        orderlist = [[1], [1, 2], [1, 2, 3]]

        # Check the value
        # Set single order to list
        if isinstance(ords, int):
            ords = [ords]
        if not all([o in [1, 2, 3] for o in ords]):
            raise ValueError("'{}' is not a valid list of orders. Try {}".format(ords, orderlist))

        # Set it
        self._orders = ords

        # Update the results
        self._reset_data()

    @property
    def planet(self):
        """Getter for the stellar data"""
        return self._planet

    @planet.setter
    def planet(self, spectrum):
        """Setter for the planetary data

        Parameters
        ----------
        spectrum: sequence
            The [W, F] or [W, F, E] of the planet to simulate
        """
        # Check if the planet has been set
        if spectrum is None:
            self._planet = None

        else:

            # Check planet is a sequence of length 2 or 3
            if not isinstance(spectrum, (list, tuple)) or not len(spectrum) in [2, 3]:
                raise ValueError(type(spectrum), ': Planet input must be a sequence of [W, F] or [W, F, E]')

            # Check the units
            if not spectrum[0].unit.is_equivalent(q.um):
                raise ValueError(spectrum[0].unit, ': Wavelength must be in units of distance')

            # Check the transmission spectrum is less than 1
            if not all(spectrum[1] < 1):
                raise ValueError('{} - {}: Transmission must be between 0 and 1'.format(min(spectrum[1]), max(spectrum[1])))

            # Check the wavelength range
            spec_min = np.nanmin(spectrum[0][spectrum[0] > 0.])
            spec_max = np.nanmax(spectrum[0][spectrum[0] > 0.])
            sim_min = np.nanmin(self.wave[self.wave > 0.])*q.um
            sim_max = np.nanmax(self.wave[self.wave > 0.])*q.um
            if spec_min > sim_min or spec_max < sim_max:
                print("Wavelength range of input spectrum ({} - {} um) does not cover the {} - {} um range needed for a complete simulation. Interpolation will be used at the edges.".format(spec_min, spec_max, sim_min, sim_max))

            # Good to go
            self._planet = spectrum

    @run_required
    def plot(self, idx=0, scale='linear', order=None, noise=True, traces=False, saturation=0.8, draw=True):
        """
        Plot a TSO frame

        Parameters
        ----------
        idx: int
            The frame index to plot
        scale: str
            Plot scale, ['linear', 'log']
        order: int (optional)
            The order to isolate
        noise: bool
            Plot with the noise model
        traces: bool
            Plot the traces used to generate the frame
        saturation: float
            The fraction of full well defined as saturation
        draw: bool
            Render the figure instead of returning it
        """
        # Get the data cube
        tso = self._select_data(order, noise)

        # Set the plot args
        wavecal = self.wave
        title = '{} - Frame {}'.format(self.title, idx)
        coeffs = locate_trace.trace_polynomial() if traces else None

        # Plot the frame
        fig = plotting.plot_frames(data=tso, idx=idx, scale=scale, trace_coeffs=coeffs, saturation=saturation, title=title, wavecal=wavecal)

        if draw:
            show(fig)
        else:
            return fig

    @run_required
    def plot_ramp(self, order=None, noise=True, draw=True):
        """
        Plot the total flux on each frame to display the ramp

        Parameters
        ----------
        order: sequence
            The order to isolate
        noise: bool
            Plot with the noise model
        draw: bool
            Render the figure instead of returning it
        """
        # Get the data cube
        tso = self._select_data(order, noise)

        # Make the figure
        fig = plotting.plot_ramp(tso)

        if draw:
            show(fig)
        else:
            return fig

    @run_required
    def plot_lightcurve(self, column, time_unit='s', resolution_mult=20, draw=True):
        """
        Plot a lightcurve for each column index given

        Parameters
        ----------
        column: int, float, sequence
            The integer column index(es) or float wavelength(s) in microns
            to plot as a light curve
        time_unit: string
            The string indicator for the units that the self.time array is in
            ['s', 'min', 'h', 'd' (default)]
        resolution_mult: int
            The number of theoretical points to plot for each data point
        draw: bool
            Render the figure instead of returning it
        """
        # Check time_units
        if time_unit not in ['s', 'min', 'h', 'd']:
            raise ValueError("time_unit must be 's', 'min', 'h' or 'd']")

        # Get the scaled flux in each column for the last group in
        # each integration
        flux_cols = np.nansum(self.tso_ideal.reshape(self.dims3)[self.ngrps-1::self.ngrps], axis=1)
        flux_cols = flux_cols/np.nanmax(flux_cols, axis=0)[None, :]

        # Make it into an array
        if isinstance(column, (int, float)):
            column = [column]

        # Make the figure
        lc = figure()

        for kcol, col in enumerate(column):

            color = next(utils.COLORS)

            # If it is an index
            if isinstance(col, (int, np.integer)):
                lightcurve = flux_cols[:, col]
                label = 'Column {}'.format(col)

            # Or assumed to be a wavelength in microns
            elif isinstance(col, float):
                waves = np.mean(self.wave[0], axis=0)
                lightcurve = [np.interp(col, waves, flux_col) for flux_col in flux_cols]
                label = '{} um'.format(col)

            else:
                print('Please enter an index, astropy quantity, or array thereof.')
                return

            # Plot the theoretical light curve
            if str(type(self.tmodel)) == "<class 'batman.transitmodel.TransitModel'>":

                # Make time axis and convert to desired units
                time = np.linspace(min(self.time), max(self.time), self.ngrps*self.nints*resolution_mult)
                time = time*q.s.to('d')

                tmodel = batman.TransitModel(self.tmodel, time)
                tmodel.rp = self.rp[col]
                theory = tmodel.light_curve(tmodel)
                theory *= max(lightcurve)/max(theory)

                time = time*q.d.to(time_unit)

                lc.line(time, theory, legend=label+' model', color=color, alpha=0.8)

            # Convert datatime
            data_time = self.time[self.ngrps-1::self.ngrps].copy()
            data_time = data_time*q.s.to(time_unit)

            # Plot the lightcurve
            lc.circle(data_time, lightcurve, legend=label, color=color)

        lc.xaxis.axis_label = 'Time [{}]'.format(time_unit)
        lc.yaxis.axis_label = 'Transit Depth'

        if draw:
            show(lc)
        else:
            return lc

    @run_required
    def plot_spectrum(self, frame=0, order=None, noise=False, scale='log', draw=True):
        """
        Parameters
        ----------
        frame: int
            The frame number to plot
        order: sequence
            The order to isolate
        noise: bool
            Plot with the noise model
        scale: str
            Plot scale, ['linear', 'log']
        draw: bool
            Render the figure instead of returning it
        """
        # Get the data cube
        tso = self._select_data(order, noise)

        # Get extracted spectrum (Column sum for now)
        wave = np.mean(self.wave[0], axis=0)
        flux_out = np.sum(tso[frame].data, axis=0)
        response = 1./self.order1_response

        # Convert response in [mJy/ADU/s] to [Flam/ADU/s] then invert so
        # that we can convert the flux at each wavelegth into [ADU/s]
        flux_out *= response/self.time[np.mod(self.ngrps, frame)]

        # Trim wacky extracted edges
        flux_out[0] = flux_out[-1] = np.nan

        # Plot it along with input spectrum
        flux_in = np.interp(wave, self.star[0], self.star[1])

        # Make the spectrum plot
        spec = figure(x_axis_type=scale, y_axis_type=scale, width=1024, height=500)
        spec.step(wave, flux_out, mode='center', legend='Extracted', color='red')
        spec.step(wave, flux_in, mode='center', legend='Injected', alpha=0.5)
        spec.yaxis.axis_label = 'Flux Density [{}]'.format(self.star[1].unit)

        # Get the residuals
        res = figure(x_axis_type=scale, x_range=spec.x_range, width=1024, height=150)
        res.step(wave, flux_out-flux_in, mode='center')
        res.xaxis.axis_label = 'Wavelength [{}]'.format(self.star[0].unit)
        res.yaxis.axis_label = 'Residuals'

        if draw:
            show(column(spec, res))
        else:
            return column(spec, res)

    def _reset_data(self):
        """Reset the results to all zeros"""
        # Check that all the appropriate values have been initialized
        if all([i in self.info for i in ['nints', 'ngrps', 'nrows', 'ncols']]):

            # Update the dimensions
            self.dims = (self.nints, self.ngrps, self.nrows, self.ncols)
            self.dims3 = (self.nints*self.ngrps, self.nrows, self.ncols)

            # Reset the results
            for arr in ['tso', 'tso_ideal']+['tso_order{}_ideal'.format(n) for n in self.orders]:
                setattr(self, arr, None)

    def _reset_time(self):
        """Reset the time axis based on the observation settings"""
        # Check that all the appropriate values have been initialized
        if all([i in self.info for i in ['subarray', 'nints', 'ngrps', 't0', 'nresets']]):

            # Get frame time based on the subarray
            self.frame_time = self.subarray_specs.get('tfrm')
            self.group_time = self.subarray_specs.get('tgrp')

            # Generate the time axis, removing reset frames
            time_axis = []
            t = self.t0+self.frame_time
            for _ in range(self.nints):
                times = t+np.arange(self.nresets+self.ngrps)*self.frame_time
                t = times[-1]+self.frame_time
                time_axis.append(times[self.nresets:])

            self.time = np.concatenate(time_axis)
            self.inttime = np.tile(self.time[:self.ngrps], self.nints)

    def _reset_psfs(self):
        """Scale the psf for each detector column to the flux from the 1D spectrum"""
        # Check that all the appropriate values have been initialized
        if all([i in self.info for i in ['filter', 'subarray']]) and self.star is not None:

            for order in self.orders:

                # Get the wavelength map
                wave = self.avg_wave[order-1]

                # Get relative spectral response for the order
                throughput = self.photom[self.photom['order'] == order]
                ph_wave = throughput.wavelength[throughput.wavelength > 0][1:-2]
                ph_resp = throughput.relresponse[throughput.wavelength > 0][1:-2]
                response = np.interp(wave, ph_wave, ph_resp)

                # Add spectral lines if necessary
                for line in self.lines:
                    self.star[1] += line['flux']

                # Convert response in [mJy/ADU/s] to [Flam/ADU/s] then invert so
                # that we can convert the flux at each wavelegth into [ADU/s]
                response = self.frame_time/(response*q.mJy*ac.c/(wave*q.um)**2).to(self.star[1].unit).value
                flux = np.interp(self.avg_wave[order-1], self.star[0], self.star[1], left=0, right=0)*response
                cube = mt.SOSS_psf_cube(filt=self.filter, order=order, subarray=self.subarray)*flux[:, None, None]
                setattr(self, 'order{}_response'.format(order), response)
                setattr(self, 'order{}_psfs'.format(order), cube)

    @run_required
    def _select_data(self, order, noise, reshape=True):
        """
        Select the data given the order and noise args

        Parameters
        ----------
        order: int (optional)
            The order to use, [1, 2, 3]
        noise: bool
            Include noise model
        reshape: bool
            Reshape to 3 dimensions

        Returns
        -------
        np.ndarray
            The selected data
        """
        if order in [1, 2]:
            tso = getattr(self, 'tso_order{}_ideal'.format(order))
        else:
            if noise:
                tso = self.tso
            else:
                tso = self.tso_ideal

        # Reshape data
        if reshape:
            tso.shape = self.dims3

        return tso

    def simulate(self, ld_coeffs=None, noise=True, model_grid=None, n_jobs=-1, **kwargs):
        """
        Generate the simulated 4D ramp data given the initialized TSO object

        Parameters
        ----------
        ld_coeffs: array-like (optional)
            A 3D array that assigns limb darkening coefficients to each pixel, i.e. wavelength
        ld_profile: str (optional)
            The limb darkening profile to use
        noise: bool
            Add noise model
        model_grid: ExoCTK.modelgrid.ModelGrid (optional)
            The model atmosphere grid to calculate LDCs
        n_jobs: int
            The number of cores to use in multiprocessing

        Example
        -------
        # Run simulation of star only
        tso.simulate()

        # Simulate star with transiting exoplanet by including transmission spectrum and orbital params
        import batman
        from hotsoss import PLANET_DATA
        params = batman.TransitParams()
        params.t0 = 0.                                # time of inferior conjunction
        params.per = 5.7214742                        # orbital period (days)
        params.a = 3.5                                # semi-major axis (in units of stellar radii)
        params.inc = 89.8                             # orbital inclination (in degrees)
        params.ecc = 0.                               # eccentricity
        params.w = 90.                                # longitude of periastron (in degrees)
        params.limb_dark = 'quadratic'                # limb darkening profile to use
        params.u = [0.1, 0.1]                         # limb darkening coefficients
        tmodel = batman.TransitModel(params, tso.time)
        tmodel.teff = 3500                            # effective temperature of the host star
        tmodel.logg = 5                               # log surface gravity of the host star
        tmodel.feh = 0                                # metallicity of the host star
        tso.simulate(planet=PLANET_DATA, tmodel=tmodel)
        """
        # Check that there is star data
        if self.star is None:
            print("No star to simulate! Please set the self.star attribute!")
            return

        # Check kwargs for updated attrs
        for key, val in kwargs.items():
            setattr(self, key, val)

        if self.verbose:
            begin = time.time()

        # Set the number of cores for multiprocessing
        max_cores = cpu_count()
        if n_jobs == -1 or n_jobs > max_cores:
            n_jobs = max_cores

        # Clear previous results
        self._reset_data()

        # Generate simulation for each order
        for order in self.orders:

            # Get the wavelength map
            wave = self.avg_wave[order-1]

            # Get the psf cube and filter response function
            psfs = getattr(self, 'order{}_psfs'.format(order))

            # Get limb darkening coeffs and make into a list
            ld_coeffs = self.ld_coeffs[order-1]
            ld_coeffs = list(map(list, ld_coeffs))

            # Set the radius at the given wavelength from the transmission
            # spectrum (Rp/R*)**2... or an array of ones
            if self.planet is not None:
                tdepth = np.interp(wave, self.planet[0], self.planet[1])
            else:
                tdepth = np.ones_like(wave)
            self.rp = np.sqrt(tdepth)

            # Run multiprocessing to generate lightcurves
            if self.verbose:
                print('Calculating order {} light curves...'.format(order))
                start = time.time()

            # Generate the lightcurves at each wavelength
            pool = ThreadPool(n_jobs)
            func = partial(mt.psf_lightcurve, time=self.time, tmodel=self.tmodel)
            data = list(zip(psfs, ld_coeffs, self.rp))
            lightcurves = np.asarray(pool.starmap(func, data), dtype=np.float64)
            pool.close()
            pool.join()
            del pool

            # Reshape to make frames
            lightcurves = lightcurves.swapaxes(0, 1)

            # Multiply by the integration time to convert to [ADU]
            lightcurves *= self.inttime[:, None, None, None]

            # Generate TSO frames
            if self.verbose:
                print('Lightcurves finished:', round(time.time()-start, 3), 's')
                print('Constructing order {} traces...'.format(order))
                start = time.time()

            # Make the 2048*N lightcurves into N frames
            pool = ThreadPool(n_jobs)
            frames = np.asarray(pool.map(mt.make_frame, lightcurves))
            pool.close()
            pool.join()
            del pool

            if self.verbose:
                # print('Total flux after warp:', np.nansum(all_frames[0]))
                print('Order {} traces finished:'.format(order), round(time.time()-start, 3), 's')

            # Add it to the individual order
            setattr(self, 'tso_order{}_ideal'.format(order), frames)

            # Clear memory
            del frames, lightcurves, psfs, wave

        # Add to the master TSO
        self.tso_ideal = np.sum([getattr(self, 'tso_order{}_ideal'.format(order)) for order in self.orders], axis=0)
        self.tso = self.tso_ideal.copy()

        # Trim SUBSTRIP256 array if SUBSTRIP96
        if self.subarray == 'SUBSTRIP96':
            for arr in ['tso', 'tso_ideal']+['tso_order{}_ideal'.format(n) for n in self.orders]:
                setattr(self, arr, getattr(self, arr)[:, :self.nrows, :])

        # Expand SUBSTRIP256 array if FULL frame
        if self.subarray == 'FULL':
            for arr in ['tso', 'tso_ideal']+['tso_order{}_ideal'.format(n) for n in self.orders]:
                full = np.zeros(self.dims3)
                full[:, -256:, :] = getattr(self, arr)
                setattr(self, arr, full)
                del full

        # Reshape into (nints, ngrps, y, x)
        for arr in ['tso', 'tso_ideal']+['tso_order{}_ideal'.format(n) for n in self.orders]:
            setattr(self, arr, getattr(self, arr).reshape(self.dims))

        # Make ramps and add noise to the observations using Kevin Volk's
        # dark ramp simulator
        if noise:
            self.add_noise()

        # Simulate reference pixels
        self.add_refpix()

        if self.verbose:
            print('\nTotal time:', round(time.time()-begin, 3), 's')

    @property
    def star(self):
        """Getter for the stellar data"""
        return self._star

    @star.setter
    def star(self, spectrum):
        """Setter for the stellar data

        Parameters
        ----------
        spectrum: sequence
            The [W, F] or [W, F, E] of the star to simulate
        """
        # Check if the star has been set
        if spectrum is None:
            if self.verbose:
                print("No star to simulate! Please set the self.star attribute!")
            self._star = None

        else:

            # Check star is a sequence of length 2 or 3
            if not isinstance(spectrum, (list, tuple)) or not len(spectrum) in [2, 3]:
                raise ValueError(type(spectrum), ': Star input must be a sequence of [W, F] or [W, F, E]')

            # Check star has units
            if not all([isinstance(i, q.quantity.Quantity) for i in spectrum]):
                types = ', '.join([str(type(i)) for i in spectrum])
                raise ValueError('[{}]: Spectrum must be in astropy units'.format(types))

            # Check the units
            if not spectrum[0].unit.is_equivalent(q.um):
                raise ValueError(spectrum[0].unit, ': Wavelength must be in units of distance')

            if not all([i.unit.is_equivalent(q.erg/q.s/q.cm**2/q.AA) for i in spectrum[1:]]):
                raise ValueError(spectrum[1].unit, ': Flux density must be in units of F_lambda')

            # Check the wavelength range
            spec_min = np.nanmin(spectrum[0][spectrum[0] > 0.])
            spec_max = np.nanmax(spectrum[0][spectrum[0] > 0.])
            sim_min = np.nanmin(self.wave[self.wave > 0.])*q.um
            sim_max = np.nanmax(self.wave[self.wave > 0.])*q.um
            if spec_min > sim_min or spec_max < sim_max:
                print("Wavelength range of input spectrum ({} - {} um) does not cover the {} - {} um range needed for a complete simulation. Interpolation will be used at the edges.".format(spec_min, spec_max, sim_min, sim_max))

            # Good to go
            self._star = spectrum

            # Reset the psfs
            self._reset_psfs()

    @property
    def subarray(self):
        """Getter for the subarray"""
        return self._subarray

    @subarray.setter
    def subarray(self, subarr):
        """Setter for the subarray

        Properties
        ----------
        subarr: str
            The name of the subarray to use,
            ['SUBSTRIP256', 'SUBSTRIP96', 'FULL']
        """
        subs = ['SUBSTRIP256', 'SUBSTRIP96', 'FULL']

        # Check the value
        if subarr not in subs:
            raise ValueError("'{}' not a supported subarray. Try {}".format(subarr, subs))

        # Set the subarray
        self._subarray = subarr
        self.subarray_specs = utils.subarray_specs(subarr)

        # Set the dependent quantities
        self._ncols = 2048
        self._nrows = self.subarray_specs.get('y')
        self.wave = utils.wave_solutions(subarr)
        self.avg_wave = np.mean(self.wave, axis=1)
        self.coeffs = locate_trace.trace_polynomial(subarray=subarr)

        # Reset the data and time arrays
        self._reset_data()
        self._reset_time()

        # Reset the psfs
        self._reset_psfs()

    @property
    def t0(self):
        """Getter for transit midpoint"""
        return self._t0

    @t0.setter
    def t0(self, tmid):
        """Setter for transit midpoint

        Properties
        ----------
        tmid: str
            The transit midpoint
        """
        # Check the value
        if not isinstance(tmid, (float, int)):
            raise ValueError("'{}' not a supported transit midpoint. Try a float or integer value.".format(tmid))

        # Set the transit midpoint
        self._t0 = tmid

        # Reset the data and time arrays
        self._reset_data()
        self._reset_time()

    @property
    def target(self):
        """Getter for target name"""
        return self._target

    @target.setter
    def target(self, name):
        """Setter for target name and coordinates

        Properties
        ----------
        tmid: str
            The transit midpoint
        """
        # Check the name
        if not isinstance(name, str):
            raise TypeError("Target name must be a string.")

        # Set the subarray
        self._target = name
        self.ra = 1.23456
        self.dec = 2.34567

        # Query Simbad for target RA and Dec
        if self.target != 'New Target':

            try:
                rec = Simbad.query_object(self.target)
                coords = SkyCoord(ra=rec[0]['RA'], dec=rec[0]['DEC'], unit=(q.hour, q.degree), frame='icrs')
                self.ra = coords.ra.degree
                self.dec = coords.dec.degree
                if self.verbose:
                    print("Coordinates {} {} for '{}' found in Simbad!".format(self.ra, self.dec, self.target))
            except TypeError:
                if self.verbose:
                    print("Could not resolve target '{}' in Simbad. Using ra={}, dec={}.".format(self.target, self.ra, self.dec))
                    print("Set coordinates manually by updating 'ra' and 'dec' attributes.")

    @property
    def tmodel(self):
        """Getter for the transit model"""
        return self._tmodel

    @tmodel.setter
    def tmodel(self, model, time_unit='days'):
        """Setter for the transit model

        Parameters
        ----------
        model: batman.transitmodel.TransitModel
            The transit model
        time_unit: string
            The units of model.t, ['seconds', 'minutes', 'hours', 'days']
        """
        # Check if the transit model has been set
        if model is None:
            self._tmodel = None

        else:

            # Check transit model type
            mod_type = str(type(model))
            if not mod_type == "<class 'batman.transitmodel.TransitModel'>":
                raise TypeError("{}: Transit model must be of type batman.transitmodel.TransitModel".format(mod_type))

            # Check time units
            time_units = {'seconds': 86400., 'minutes': 1440., 'hours': 24., 'days': 1.}
            if time_unit not in time_units:
                raise ValueError("{}: time_unit must be {}".format(time_unit, time_units.keys()))

            # Check if the stellar params have changed
            plist = ['teff', 'logg', 'feh', 'limb_dark']
            old_params = [getattr(self.tmodel, p, None) for p in plist]
            new_params = [getattr(model, p) for p in plist]

            # Update the LD profile
            self.ld_profile = model.limb_dark

            # Convert seconds to days in order to match the Period and T0 parameters
            model.t /= time_units[time_unit]

            # Update the transit model
            self._tmodel = model

            # Update ld_coeffs if necessary
            if new_params != old_params:
                self.ld_coeffs = 'update'


class TestTSO(TSO):
    """Generate a test object for quick access"""
    def __init__(self, ngrps=2, nints=2, filter='CLEAR', subarray='SUBSTRIP256', run=True, add_planet=False, **kwargs):
        """Get the test data and load the object

        Parameters
        ----------
        ngrps: int
            The number of groups per integration
        nints: int
            The number of integrations for the exposure
        filter: str
            The name of the filter to use, ['CLEAR', 'F277W']
        subarray: str
            The name of the subarray to use, ['SUBSTRIP256', 'SUBSTRIP96', 'FULL']
        run: bool
            Run the simulation after initialization
        add_planet: bool
            Add a transiting exoplanet
        """
        # Initialize base class
        super().__init__(ngrps=ngrps, nints=nints, star=utils.STAR_DATA, subarray=subarray, filter=filter, **kwargs)

        # Add planet
        if add_planet:
            self.planet = utils.PLANET_DATA
            self.tmodel = utils.transit_params(self.time)

        # Run the simulation
        if run:
            self.simulate()


class BlackbodyTSO(TSO):
    """Generate a test object with a blackbody spectrum"""
    def __init__(self, ngrps=2, nints=2, teff=1800, filter='CLEAR', subarray='SUBSTRIP256', run=True, add_planet=False, scale=1., **kwargs):
        """Get the test data and load the object

        Parmeters
        ---------
        ngrps: int
            The number of groups per integration
        nints: int
            The number of integrations for the exposure
        teff: int
            The effective temperature of the test source
        filter: str
            The name of the filter to use, ['CLEAR', 'F277W']
        subarray: str
            The name of the subarray to use, ['SUBSTRIP256', 'SUBSTRIP96', 'FULL']
        run: bool
            Run the simulation after initialization
        add_planet: bool
            Add a transiting exoplanet
        scale: int, float
            Scale the flux by the given factor
        """
        # Generate a blackbody at the given temperature
        bb = BlackBody1D(temperature=teff*q.K)
        wav = np.linspace(0.5, 2.9, 1000) * q.um
        flux = bb(wav).to(FLAM, q.spectral_density(wav))*1E-8*scale

        # Initialize base class
        super().__init__(ngrps=ngrps, nints=nints, star=[wav, flux], subarray=subarray, filter=filter, **kwargs)

        # Add planet
        if add_planet:
            self.planet = utils.PLANET_DATA
            self.tmodel = utils.transit_params(self.time)

        # Run the simulation
        if run:
            self.simulate()

class ModelTSO(TSO):
    def closest_value(self, input_value, possible_values):
        """
        This function calculates, given an input_value and an array of possible_values, 
        the closest value to input_value in the array.
        """
        distance = np.abs(possible_values - input_value)
        idx = np.where(distance == np.min(distance))[0]
        return possible_values[idx[0]]

    def get_atlas_folder(self, feh):
        """
        Given input metallicity, this function defines the first part of the URL that will define what
        file to download from the STScI website.
        """
        # Define closest possible metallicity from ATLAS models:
        model_metallicity = self.closest_value(feh, np.array([-2.5,-2.0,-1.5,-1.0,-0.5,0.0,0.2,0.5]))
        met_sign = 'm'
        # Define the sign before the filename, obtain absolute value if needed:
        if model_metallicity >= 0.0:
            met_sign = 'p'
        else:
            model_metallicity = np.abs(model_metallicity)
        model_metallicity = ''.join(str(model_metallicity).split('.'))
        fname = 'https://archive.stsci.edu/hlsps/reference-atlases/cdbs/grid/ck04models/ck{0:}{1:}/'.format(met_sign,model_metallicity)
        return fname

    def get_phoenix_folder(self, feh, alpha):
        """
        Given input metallicity and alpha-enhancement, this function defines the first part of the URL that will define what 
        file to download from the PHOENIX site.
        """
        # Define closest possible metallicity from PHOENIX models:
        model_metallicity = self.closest_value(feh, np.array([-4.0,-3.0,-2.0,-1.5,-1.0,-0.5,0.0,0.5,1.0]))
        # Same for alpha-enhancement:
        model_alpha = self.closest_value(alpha, np.array([-0.2,0.0,0.2,0.4,0.6,0.8,1.0,1.20]))
        met_sign, alpha_sign = '-', '-'
        # Define the sign before the filename, obtain absolute value if needed:
        if model_metallicity > 0.0:
            met_sign = '+'
        else:
            model_metallicity = np.abs(model_metallicity)
        if model_alpha > 0.0:
            alpha_sign = '+'
        else:
            model_alpha = np.abs(model_alpha)
        # Create the folder name
        if alpha == 0.0:
            fname = 'ftp://phoenix.astro.physik.uni-goettingen.de/HiResFITS/PHOENIX-ACES-AGSS-COND-2011/Z{0:}{1:.1f}/'.format(met_sign,model_metallicity)
        else:
            fname = 'ftp://phoenix.astro.physik.uni-goettingen.de/HiResFITS/PHOENIX-ACES-AGSS-COND-2011/Z{0:}{1:.1f}.Alpha={2:}{3:.2f}/'.format(met_sign,model_metallicity,alpha_sign,model_alpha)
        return fname

    def download(self, url, fname):
        """
        Download files from ftp server at url in filename fname. Obtained/modified from jfs here: https://stackoverflow.com/questions/11768214/python-download-a-file-over-an-ftp-server
        """
        with closing(request.urlopen(url)) as r:
            with open(fname, 'wb') as f:
                shutil.copyfileobj(r, f)

    def get_vega(self):
        """
        This functions reads in the spectrum of Vega (Alpha Lyr) from CALSPEC
        """
        data = fits.getdata(resource_filename('awesimsoss', 'files/alpha_lyr_stis_009.fits'),header=False)
        # Wavelength is in Angstroms, convert to microns to match the get_phoenix_model function. 
        # Flux is in Flambda (same as Phoenix; i.e., erg/s/cm2/A):
        return (data['WAVELENGTH']*q.angstrom).to(q.um),data['FLUX']*(q.erg/q.s/q.cm**2/q.AA) 

    def read_phoenix_list(self, phoenix_model_list):
        fin = open(phoenix_model_list,'r')
        fnames = np.array([])
        teffs = np.array([])
        loggs = np.array([])
        while True:
            line = fin.readline()
            if line!='':
                fname = line.split()[-1]
                teff, logg = fname.split('-')[:2]
                fnames = np.append(fnames, fname)
                teffs = np.append(teffs, np.double(teff[3:]))
                loggs = np.append(loggs, np.double(logg))
            else:
                break
        return fnames, teffs, loggs
    
    def get_phoenix_model(self, feh, alpha, teff, logg):
        """
        This function gets you the closest PHOENIX high-resolution model to the input stellar parameters 
        from the Goettingen website (ftp://phoenix.astro.physik.uni-goettingen.de). Outputs are two arrays, one 
        containing the wavelength in um and the other containing the (surface) flux (in f-lambda) of the 
        star in units of erg/s/cm**2/angstroms.
        """
        # First get grid corresponding to input Fe/H and alpha:
        url_folder = self.get_phoenix_folder(feh, alpha)
        # Now define details for filenames and folders. First, extract metallicity and alpha-enhancement in 
        # the PHOENIX filename format (and get rid of the "Z" in, e.g., "Z-1.0.Alpha=-0.20"):
        phoenix_met_and_alpha = url_folder.split('/')[-2][1:]
        # Define folders where we will save (1) all stellar model data and (2) all phoenix models:
        stellarmodels_folder_path = resource_filename('awesimsoss', 'files/stellarmodels/')
        phoenix_folder_path = resource_filename('awesimsoss', 'files/stellarmodels/phoenix/')
        model_folder_path = resource_filename('awesimsoss', 'files/stellarmodels/phoenix/'+phoenix_met_and_alpha+'/')

        # Check if we even have stellarmodels folder created. Create it if not:
        if not os.path.exists(stellarmodels_folder_path):
            os.mkdir(stellarmodels_folder_path)
        # Same for phoenix folder:
        if not os.path.exists(phoenix_folder_path):
            os.mkdir(phoenix_folder_path)
        # Check if the current metallicity-alpha folder exists as well:
        if not os.path.exists(model_folder_path):
            os.mkdir(model_folder_path)

        # Check if we have the PHOENIX wavelength solution. If not, download it:
        if not os.path.exists(phoenix_folder_path+'wavsol.fits'):
            self.download('ftp://phoenix.astro.physik.uni-goettingen.de/HiResFITS/WAVE_PHOENIX-ACES-AGSS-COND-2011.fits',phoenix_folder_path+'wavsol.fits')

        # Extract wavelength solution:
        wavelengths = fits.getdata(phoenix_folder_path+'wavsol.fits')

        # Now, figure out the closest model to the input stellar parameters. For this, first figure out the range of teff and logg 
        # for the current metallicity and alpha. For this, either retrieve from the system or download the full list of PHOENIX models 
        # for the current metallicity and alpha. If not already here, save it on the system:
        phoenix_model_list = model_folder_path+'model_list.txt'
        if not os.path.exists(phoenix_model_list):
            self.download(url_folder, phoenix_model_list)
        # Extract information from this list:
        model_names, possible_teffs, possible_loggs = self.read_phoenix_list(model_folder_path+'model_list.txt')

        # Search the closest to the input teff:
        phoenix_teff = self.closest_value(teff, possible_teffs)

        # Raise a warning in case the found teff is outside the PHOENIX model range, give some 
        # guidance on how to proceed:
        if np.abs(phoenix_teff-teff)>200.:
            print('\t Warning: the input stellar effective temperature is outside the {0:}-{1:} K model range of PHOENIX models for {2:}.'.format(np.min(possible_teffs),\
                    np.max(possible_teffs),phoenix_met_and_alpha))
            if 'Alpha' in phoenix_met_and_alpha:
                print('\t Modelling using a {0:} K model. Using models without alpha-enhancement (alpha = 0.0), which range '+\
                                'from 2300 to 12000 K would perhaps help find more suitable temperature models.'.format(phoenix_teff))
            else:
                print('\t Modelling using a {0:} K model.'.format(phoenix_teff))

        # Same excercise for logg, given the teffs:
        idx_logg = np.where(np.abs(phoenix_teff-possible_teffs) == 0.)[0]
        phoenix_logg = self.closest_value(logg, possible_loggs[idx_logg])
        
        # Select final model:
        idx = np.where((np.abs(phoenix_teff-possible_teffs) == 0.)&(np.abs(possible_loggs == phoenix_logg)))[0]
        phoenix_model, phoenix_logg = model_names[idx][0], possible_loggs[idx][0]
        
        # Raise warning for logg as well:
        if np.abs(phoenix_logg - logg)>0.5:
            print('\t Warning: the input stellar log-gravity is outside the {0:}-{1:} model range of PHOENIX models for {2:} and Teff {3:}.'.format(\
                   np.min(possible_loggs[idx_logg]),np.max(possible_loggs[idx_logg]),phoenix_met_and_alpha,phoenix_teff))

        # Check if we already have the downloaded model. If not, download the corresponding file:
        if not os.path.exists(model_folder_path+phoenix_model):
            print('\t PHOENIX stellar models for {0:} not found in {1:}. Downloading...'.format(phoenix_met_and_alpha,model_folder_path))
            self.download(url_folder+phoenix_model,model_folder_path+phoenix_model)

        # Once we have the file, simply extract the data:
        print('\t Using the {0:} PHOENIX model (Teff {1:}, logg {2:}).'.format(phoenix_model, phoenix_teff, phoenix_logg))
        flux = fits.getdata(model_folder_path+phoenix_model,header=False)

        # Change units in order to match what is expected by the TSO modules:
        wav = (wavelengths * q.angstrom).to(q.um)
        flux = (flux * (q.erg/q.s/q.cm**2/q.cm)).to(q.erg/q.s/q.cm**2/q.AA)
        return wav, flux

    def get_atlas_model(self, feh, vturb, teff, logg):
        """
        This function gets you the closest ATLAS9 Castelli and Kurucz model to the input stellar parameters
        from the STScI website (http://www.stsci.edu/hst/instrumentation/reference-data-for-calibration-and-tools/astronomical-catalogs/castelli-and-kurucz-atlas). 
        Outputs are two arrays, one containing the wavelength in um and the other containing the (surface) flux (in f-lambda) of the
        star in units of erg/s/cm**2/angstroms.
        """
        # First get grid corresponding to input Fe/H:
        url_folder = self.get_atlas_folder(feh)
        # Now define details for filenames and folders. Extract foldername with the metallicity info from the url_folder:
        atlas_met = url_folder.split('/')[-2]
        # Define folders where we will save (1) all stellar model data and (2) all atlas models:
        stellarmodels_folder_path = resource_filename('awesimsoss', 'files/stellarmodels/')
        atlas_folder_path = resource_filename('awesimsoss', 'files/stellarmodels/atlas/')
        model_folder_path = resource_filename('awesimsoss', 'files/stellarmodels/atlas/'+atlas_met+'/')

        # Check if we even have stellarmodels folder created. Create it if not:
        if not os.path.exists(stellarmodels_folder_path):
            os.mkdir(stellarmodels_folder_path)
        # Same for phoenix folder:
        if not os.path.exists(atlas_folder_path):
            os.mkdir(atlas_folder_path)
        # Check if the current metallicity-alpha folder exists as well:
        if not os.path.exists(model_folder_path):
            os.mkdir(model_folder_path)

        # Define possible teff and loggs (thankfully, this is easier for ATLAS models):
        possible_teffs, possible_loggs = np.append(np.arange(3500,13250,250),np.arange(14000,51000,1000)), np.arange(0.0,5.5,0.5)

        # Check the closest teff and logg to input ones:
        atlas_teff = self.closest_value(teff, possible_teffs)
        atlas_logg = self.closest_value(logg, possible_loggs)

        # Raise a warning in case the found teff is outside the ATLAS model range, give some
        # guidance on how to proceed:
        if np.abs(atlas_teff-teff)>200.:
            print('\t Warning: the input stellar effective temperature is outside the {0:}-{1:} K model range of ATLAS models for {2:}.'.format(np.min(possible_teffs),\
                    np.max(possible_teffs),atlas_met))
            print('\t Modelling using a {0:} K model.'.format(atlas_teff))

        # Now, if not already in the system, download the model corresponding to the chosen teff:
        atlas_fname = model_folder_path+atlas_met+'_{0:}.fits'.format(atlas_teff)
        if not os.path.exists(atlas_fname):
            self.download(url_folder+atlas_met+'_{0:}.fits'.format(atlas_teff), atlas_fname)

        # Read the file:
        d = fits.getdata(atlas_fname)

        # This variable will save non-zero logg at the given temperatures. Only useful to report back to the user and/or input logg 
        # doesn't have data:
        real_possible_loggs = np.array([])
        # Check if the closest requested logg has any data. If not, check all possible loggs for non-zero data, and select the closest 
        # to the input logg that has data:
        s_logg = 'g'+''.join('{0:.1f}'.format(atlas_logg).split('.'))
        if np.count_nonzero(d[s_logg]) != 0:
            w,f = d['WAVELENGTH'],d[s_logg]
        else:
            real_possible_loggs = np.array([])
            for loggs in possible_loggs:
                s_logg = 'g'+''.join('{0:.1f}'.format(loggs).split('.'))
                if np.count_nonzero(d[s_logg]) != 0:
                    real_possible_loggs = np.append(real_possible_loggs, loggs)
            atlas_logg = self.closest_value(logg, real_possible_loggs)
            s_logg = 'g'+''.join('{0:.1f}'.format(atlas_logg).split('.'))
            w,f = d['WAVELENGTH'],d[s_logg]

        # Raise warning for logg as well:
        if np.abs(atlas_logg - logg)>0.5:
            # If real_possible_loggs is empty, calculate it:
            if len(real_possible_loggs) == 0:
                for loggs in possible_loggs:
                    s_logg = 'g'+''.join('{0:.1f}'.format(loggs).split('.'))
                    if np.count_nonzero(d[s_logg]) != 0:
                        real_possible_loggs = np.append(real_possible_loggs, loggs)
            print('\t Warning: the input stellar log-gravity is outside the {0:}-{1:} model range of ATLAS models for {2:} and Teff {3:}.'.format(\
                   np.min(real_possible_loggs),np.max(real_possible_loggs),atlas_met,atlas_teff))

        # Change units in order to match what is expected by the TSO modules:
        wav = (w * q.angstrom).to(q.um)
        flux = f * q.erg/q.s/q.cm**2/q.AA
        return wav, flux

    def get_resolution(self, w, f):
        eff_wav = np.sum(w*f)/np.sum(f)
        delta_wav = np.median(np.abs(np.diff(w)))
        return eff_wav/delta_wav

    def spec_integral(self, input_w, input_f, wT, TT):
        """
        This function computes the integral of lambda*f*T divided by the integral of lambda*T, where 
        lambda is the wavelength, f the flux (in f-lambda) and T the transmission function. The input 
        stellar spectrum is given by wavelength w and flux f. The input filter response wavelengths 
        are given by wT and transmission curve by TT. It is assumed both w and wT are in the same wavelength 
        units.
        """

        # If resolution of input spectra in the wavelength range of the response function 
        # is higher than it, degrade it to match the transmission function resolution. First, 
        # check that resolution of input spectra is indeed higher than the one of the 
        # transmisssion. Resolution of input transmission first:
        min_wav,max_wav = np.min(wT),np.max(wT)
        resT = self.get_resolution(wT, TT) 
        # Resolution of input spectra in the same wavelength range:
        idx = np.where((input_w>=min_wav-10)&(input_w<=max_wav+10))[0]
        res = self.get_resolution(input_w[idx],input_f[idx])
        # If input spetrum resolution is larger, degrade:
        if res>resT:
            # This can be way quicker if we just take the gaussian weight *at* the evaluated 
            # points in the interpolation. TODO: make faster.
            f = ndimage.gaussian_filter(input_f[idx],int(np.double(len(idx))/np.double(len(wT))))
            w = input_w[idx]
        else:
            w, f = input_w, input_f
        interp_spectra = interpolate.interp1d(w,f)
        numerator = np.trapz(wT*interp_spectra(wT)*TT, x = wT)
        denominator = np.trapz(wT*TT, x = wT)
        return numerator/denominator

    def scale_spectrum(self, w, f, jmag):
        """
        Function expects input wavelength (w) in um, flux (f) in erg/s/cm2/A. To scale the spectra, we use equation (8) in 
        Casagrande et al. (2014, MNRAS, 444, 392).
        """
        # Get filter response (note wT is in microns):
        wT,TT = np.loadtxt(resource_filename('awesimsoss', 'files/jband_transmission.dat'),unpack=True,usecols=(0,1))
        # Get spectrum of vega:
        w_vega,f_vega = self.get_vega()
        # Use those two to get the absolute flux calibration for Vega (left-most term in equation (9) in Casagrande et al., 2014).
        # Multiply wavelengths by 1e4 as they are in microns (i.e., transform back to angstroms both wavelength ranges):
        vega_weighted_flux = self.spec_integral(np.array(w_vega.to(q.AA)), np.array(f_vega), wT*1e4, TT)
        # J-band zero-point is thus (maginutde of Vega, m_*, obtained from Table 1 in Casagrande et al, 2014):
        ZP = -0.001 + 2.5*np.log10(vega_weighted_flux)
        # Now compute (inverse?) bolometric correction for target star. For this, compute same integral as for vega, but for target:
        target_weighted_flux = self.spec_integral(np.array(w)*1e4, f, np.array(wT)*1e4, TT)
        # Get scaling factor for target spectrum (this ommits any extinction):
        scaling_factor = 10**(-((jmag + 2.5*np.log10(target_weighted_flux) - ZP)/2.5))
        # Return scaled spectrum:
        return f*scaling_factor

    """Generate a test object with a blackbody spectrum"""
    def __init__(self, ngrps=2, nints=2, teff=5700.0, logg = 4.0, feh = 0.0, vturb = 2.0, alpha = 0.0, jmag = 9.0, stellar_model = 'PHOENIX', filter='CLEAR', subarray='SUBSTRIP256', run=True, add_planet=False, scale=1., **kwargs):
        """Get the test data and load the object

        Parmeters
        ---------
        ngrps: int
            The number of groups per integration
        nints: int
            The number of integrations for the exposure
        teff: double
            The effective temperature in kelvins of the stellar source
        logg: double
            The log-gravity of the stellar source
        feh: double
            The [Fe/H] of the stellar source
        vturb: double
            The microturbulent velocity of the stellar source in km/s
        alpha: double
            The alpha enhancement of the stellar source
        jmag: double
            The J magnitude of the source
        stellar_model: str
            The stellar model grid to use. Can either be 'ATLAS' or 'PHOENIX'. Default is 'PHOENIX'
        filter: str
            The name of the filter to use, ['CLEAR', 'F277W']
        subarray: str
            The name of the subarray to use, ['SUBSTRIP256', 'SUBSTRIP96', 'FULL']
        run: bool
            Run the simulation after initialization
        add_planet: bool
            Add a transiting exoplanet
        scale: int, float
            Scale the flux by the given factor
        """
        # Retrieve PHOENIX or ATLAS stellar models:
        if stellar_model.lower() == 'phoenix':
            w,f = self.get_phoenix_model(feh, alpha, teff, logg)
        elif stellar_model.lower() == 'atlas':
            w,f = self.get_atlas_model(feh, vturb, teff, logg)

        # Now scale model spectrum to user-input J-band:
        f = self.scale_spectrum(w,f,jmag)

        self.stellar_spectrum_wav = w
        self.stellar_spectrum_flux = f
        # Initialize base class
        super().__init__(ngrps=ngrps, nints=nints, star=[w, f], subarray=subarray, filter=filter, **kwargs)

        # Add planet
        if add_planet:
            self.planet = utils.PLANET_DATA
            self.tmodel = utils.transit_params(self.time)

        # Run the simulation
        if run: 
            self.simulate()
