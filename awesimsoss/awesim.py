# -*- coding: utf-8 -*-
"""
A module to generate simulated 2D time-series SOSS data

Authors: Joe Filippazzo, Kevin Volk, Jonathan Fraine, Michael Wolfe
"""
import time
import warnings
import datetime
from functools import partial
from pkg_resources import resource_filename
from multiprocessing.dummy import Pool as ThreadPool
from multiprocessing import cpu_count

import numpy as np
from bokeh.plotting import figure, show
from bokeh.models import LogColorMapper, LogTicker, LinearColorMapper, ColorBar
import batman
import astropy.units as q
import astropy.constants as ac
from astropy.io import fits
from exoctk import modelgrid as mg

from . import generate_darks as gd
from . import make_trace as mt

try:
    # Use a progress bar if one is available
    from tqdm import tqdm
except:
    print('`pip install tqdm` to make this procedure prettier')
    tqdm = lambda iterable, total=None: iterable

warnings.simplefilter('ignore')


class TSO(object):
    """
    Generate NIRISS SOSS time series observations
    """
    def __init__(self, ngrps, nints, star, snr=700, filt='CLEAR',
                 subarray='SUBSTRIP256', orders=[1, 2], t0=0,
                 target='Simulated Target', verbose=True):
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
        snr: float
            The signal-to-noise
        subarray: str
            The subarray name, i.e. 'SUBSTRIP256', 'SUBSTRIP96', or 'FULL'
        t0: float
            The start time of the exposure [days]
        target: str (optional)
            The name of the target

        Example
        -------
        # Imports
        import numpy as np
        from awesimsoss import TSO
        import astropy.units as q
        from pkg_resources import resource_filename
        star = np.genfromtxt(resource_filename('awesimsoss', 'files/scaled_spectrum.txt'), unpack=True)
        star1D = [star[0]*q.um, (star[1]*q.W/q.m**2/q.um).to(q.erg/q.s/q.cm**2/q.AA)]

        # Initialize simulation
        tso = TSO(ngrps=3, nints=10, star=star1D)
        """
        # Set instance attributes for the exposure
        self.subarray = subarray
        self.nrows = mt.SUBARRAY_Y[subarray]
        self.ncols = 2048
        self.ngrps = ngrps
        self.nints = nints
        self.nresets = 1
        self.frame_time = mt.FRAME_TIMES[subarray]
        self.time = mt.get_frame_times(subarray, ngrps, nints, t0, self.nresets)
        self.nframes = len(self.time)
        self.target = target
        self.obs_date = '2016-01-04'
        self.obs_time = '23:37:52.226'
        self.filter = filt
        self.header = ''
        self.gain = 1.61
        self.snr = snr
        self.model_grid = None

        # Set instance attributes for the target
        self.star = star
        self.wave = mt.wave_solutions(subarray)
        self.avg_wave = np.mean(self.wave, axis=1)
        self._ld_coeffs = np.zeros((3, 2048, 2))
        self.planet = None
        self.tmodel = None

        # Set single order to list
        if isinstance(orders, int):
            orders = [orders]
        if not all([o in [1, 2] for o in orders]):
            raise TypeError('Order must be either an int, float, or list thereof; i.e. [1, 2]')
        self.orders = list(set(orders))

        # Check if it's F277W to speed up calculation
        if self.filter == 'F277W':
            self.orders = [1]

        # Scale the psf for each detector column to the flux from
        # the 1D spectrum
        for order in self.orders:
            # Get the 1D flux in
            flux = np.interp(self.avg_wave[order-1], self.star[0], self.star[1], left=0, right=0)[:, np.newaxis, np.newaxis]
            cube = mt.SOSS_psf_cube(filt=self.filter, order=order)*flux
            setattr(self, 'order{}_psfs'.format(order), cube)

        # Get absolute calibration reference file
        calfile = resource_filename('awesimsoss', 'files/jwst_niriss_photom_0028.fits')
        caldata = fits.getdata(calfile)
        self.photom = caldata[caldata['pupil'] == 'GR700XD']

        # Create the empty exposure
        self.dims = (self.nframes, self.nrows, self.ncols)
        self.tso = np.zeros(self.dims)
        self.tso_ideal = np.zeros(self.dims)
        self.tso_order1_ideal = np.zeros(self.dims)
        self.tso_order2_ideal = np.zeros(self.dims)

    def run_simulation(self, planet=None, tmodel=None, ld_coeffs=None, time_unit='days', 
                       ld_profile='quadratic', model_grid=None, n_jobs=1, verbose=True):
        """
        Generate the simulated 2D data given the initialized TSO object

        Parameters
        ----------
        filt: str
            The element from the filter wheel to use, i.e. 'CLEAR' or 'F277W'
        planet: sequence (optional)
            The wavelength and Rp/R* of the planet at t=0
        tmodel: batman.transitmodel.TransitModel (optional)
            The transit model of the planet
        ld_coeffs: array-like (optional)
            A 3D array that assigns limb darkening coefficients to each pixel, i.e. wavelength
        ld_profile: str (optional)
            The limb darkening profile to use
        time_unit: string
            The string indicator for the units that the tmodel.t array is in
            options: 'seconds', 'minutes', 'hours', 'days' (default)
        orders: sequence
            The list of orders to imulate
        model_grid: ExoCTK.modelgrid.ModelGrid (optional)
            The model atmosphere grid to calculate LDCs
        n_jobs: int
            The number of cores to use in multiprocessing
        verbose: bool
            Print helpful stuff

        Example
        -------
        # Run simulation of star only
        tso.run_simulation()

        # Simulate star with transiting exoplanet by including transmission spectrum and orbital params
        import batman
        import astropy.constants as ac
        planet1D = np.genfromtxt(resource_filename('awesimsoss', '/files/WASP107b_pandexo_input_spectrum.dat'), unpack=True)
        params = batman.TransitParams()
        params.t0 = 0.                                # time of inferior conjunction
        params.per = 5.7214742                        # orbital period (days)
        params.a = 0.0558*q.AU.to(ac.R_sun)*0.66      # semi-major axis (in units of stellar radii)
        params.inc = 89.8                             # orbital inclination (in degrees)
        params.ecc = 0.                               # eccentricity
        params.w = 90.                                # longitude of periastron (in degrees)
        params.limb_dark = 'quadratic'                # limb darkening profile to use
        params.u = [0.1, 0.1]                          # limb darkening coefficients
        tmodel = batman.TransitModel(params, tso.time)
        tmodel.teff = 3500                            # effective temperature of the host star
        tmodel.logg = 5                               # log surface gravity of the host star
        tmodel.feh = 0                                # metallicity of the host star
        tso.run_simulation(planet=planet1D, tmodel=tmodel)
        """
        if verbose:
            begin = time.time()

        max_cores = cpu_count()
        if n_jobs == -1 or n_jobs > max_cores:
            n_jobs = max_cores

        # Clear previous results
        self.tso = np.zeros(self.dims)
        self.tso_ideal = np.zeros(self.dims)
        self.tso_order1_ideal = np.zeros(self.dims)
        self.tso_order2_ideal = np.zeros(self.dims)

        # If there is a planet transmission spectrum but no LDCs generate them
        is_tmodel = isinstance(tmodel, batman.transitmodel.TransitModel)
        if planet is not None and is_tmodel:

            if time_unit not in ['seconds', 'minutes', 'hours', 'days']:
                raise ValueError("time_unit must be either 'seconds', 'hours', or 'days']")

            # Check if the stellar params are the same
            plist = ['teff', 'logg', 'feh', 'limb_dark']
            old_params = [getattr(self.tmodel, p, None) for p in plist]

            # Store planet details
            self.planet = planet
            self.tmodel = tmodel

            if self.tmodel.limb_dark is None:
                self.tmodel.limb_dark = ld_profile

            # Set time of inferior conjunction
            if self.tmodel.t0 is None or self.time[0] > self.tmodel.t0 > self.time[-1]:
                self.tmodel.t0 = self.time[self.nframes//2]

            # Convert seconds to days, in order to match the Period and T0 parameters
            days_to_seconds = 86400.
            if time_unit == 'seconds':
                self.tmodel.t /= days_to_seconds
            if time_unit == 'minutes':
                self.tmodel.t /= days_to_seconds / 60
            if time_unit == 'hours':
                self.tmodel.t /= days_to_seconds / 3600

            # Set the ld_coeffs if provided
            stellar_params = [getattr(tmodel, p) for p in plist]
            changed = stellar_params != old_params
            if ld_coeffs is not None:
                self.ld_coeffs = ld_coeffs

            # Update the limb darkning coeffs if the stellar params or
            # ld profile have changed
            elif isinstance(model_grid, mg.ModelGrid) and changed:

                # Try to set the model grid
                self.model_grid = model_grid
                self.ld_coeffs = tmodel

            else:
                pass

        # Generate simulation for each order
        for order in self.orders:

            # Get the wavelength map
            wave = self.avg_wave[order-1]

            # Get the psf cube
            cube = getattr(self, 'order{}_psfs'.format(order))

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

            # Get relative spectral response for the order (from
            # /grp/crds/jwst/references/jwst/jwst_niriss_photom_0028.fits)
            throughput = self.photom[(self.photom['order'] == order)&(self.photom['filter'] == self.filter)]
            ph_wave = throughput.wavelength[throughput.wavelength>0][1:-2]
            ph_resp = throughput.relresponse[throughput.wavelength>0][1:-2]
            response = np.interp(wave, ph_wave, ph_resp)

            # Convert response in [mJy/ADU/s] to [Flam/ADU/s] then invert so
            # that we can convert the flux at each wavelegth into [ADU/s]
            response = self.frame_time/(response*q.mJy*ac.c/(wave*q.um)**2).to(self.star[1].unit).value
            setattr(self, 'photom_order{}'.format(order), response)

            # Run multiprocessing to generate lightcurves
            if verbose:
                print('Calculating order {} light curves...'.format(order))
                start = time.time()

            # Generate the lightcurves at each wavelength
            pool = ThreadPool(n_jobs)
            func = partial(mt.psf_lightcurve, time=self.time, tmodel=self.tmodel)
            data = list(zip(wave, cube, response, ld_coeffs, self.rp))
            psfs = np.asarray(pool.starmap(func, data))
            pool.close()
            pool.join()

            # Reshape into frames
            psfs = psfs.swapaxes(0, 1)

            # Multiply by the frame time to convert to [ADU]
            ft = np.tile(self.time[:self.ngrps], self.nints)
            psfs *= ft[:, None, None, None]

            # Generate TSO frames
            if verbose:
                print('Lightcurves finished:', time.time()-start)
                print('Constructing order {} traces...'.format(order))
                start = time.time()

            # Make the 2048*N lightcurves into N frames
            pool = ThreadPool(n_jobs)
            # func = partial(mt.make_frame, subarray=self.subarray)
            psfs = np.asarray(pool.map(mt.make_frame, psfs))
            pool.close()
            pool.join()

            if verbose:
                # print('Total flux after warp:', np.nansum(all_frames[0]))
                print('Order {} traces finished:'.format(order), time.time()-start)

            # Add it to the individual order
            setattr(self, 'tso_order{}_ideal'.format(order), np.array(psfs))

        # Add to the master TSO
        self.tso = np.sum([getattr(self, 'tso_order{}_ideal'.format(order)) for order in self.orders], axis=0)

        # Make ramps and add noise to the observations using Kevin Volk's
        # dark ramp simulator
        self.tso_ideal = self.tso.copy()
        self.add_noise()

        # Trim if SUBSTRIP96
        if self.subarray == 'SUBSTRIP96':
            self.tso = self.tso[:, 256-self.nrows:, :]
            self.tso_ideal = self.tso_ideal[:, 256-self.nrows:, :]

        if verbose:
            print('\nTotal time:', time.time()-begin)

    @property
    def ld_coeffs(self):
        """Get the limb darkening coefficients"""
        return self._ld_coeffs

    @ld_coeffs.setter
    def ld_coeffs(self, coeffs=None):
        """Set the limb darkening coefficients

        Parameters
        ----------
        coeffs: sequence
            The limb darkening coefficients
        teff: float, int
            The effective temperature of the star
        logg: int, float
            The surface gravity of the star
        feh: float, int
            The logarithm of the star metallicity/solar metallicity
        """
        # Use input ld coeff array
        if isinstance(coeffs, np.ndarray) and len(coeffs.shape) == 3:
            self._ld_coeffs = coeffs

        # Or generate them if the stellar parameters have changed
        elif isinstance(coeffs, batman.transitmodel.TransitModel) and isinstance(self.model_grid, mg.ModelGrid):
            self.ld_coeffs = [mt.generate_SOSS_ldcs(self.avg_wave[order-1], coeffs.limb_dark, [getattr(coeffs, p) for p in ['teff', 'logg', 'feh']], model_grid=self.model_grid) for order in self.orders]

        else:
            raise ValueError('Please set ld_coeffs with a 3D array or batman.transitmodel.TransitModel.')

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
        orders = np.asarray([self.tso_order1_ideal, self.tso_order2_ideal])

        # Load all the reference files
        photon_yield = fits.getdata(resource_filename('awesimsoss', 'files/photon_yield_dms.fits'))
        pca0_file = resource_filename('awesimsoss', 'files/niriss_pca0.fits')
        zodi = fits.getdata(resource_filename('awesimsoss', 'files/soss_zodiacal_background_scaled.fits'))
        nonlinearity = fits.getdata(resource_filename('awesimsoss', 'files/substrip256_forward_coefficients_dms.fits'))
        pedestal = fits.getdata(resource_filename('awesimsoss', 'files/substrip256pedestaldms.fits'))
        darksignal = fits.getdata(resource_filename('awesimsoss', 'files/substrip256signaldms.fits'))*self.gain

        # Generate the photon yield factor values
        pyf = gd.make_photon_yield(photon_yield, np.mean(orders, axis=1))

        # Remove negatives from the dark ramp
        darksignal[np.where(darksignal < 0.)] = 0.

        # Make the exposure
        RAMP = gd.make_exposure(1, self.ngrps, darksignal, self.gain, pca0_file=pca0_file, offset=offset)

        # Iterate over integrations
        for n in range(self.nints):

            # Add in the SOSS signal
            ramp = gd.add_signal(self.tso_ideal[self.ngrps*n:self.ngrps*n+self.ngrps], RAMP.copy(), pyf, self.frame_time, self.gain, zodi, zodi_scale, photon_yield=False)

            # Apply the non-linearity function
            ramp = gd.non_linearity(ramp, nonlinearity, offset=offset)

            # Add the pedestal to each frame in the integration
            ramp = gd.add_pedestal(ramp, pedestal, offset=offset)

            # Update the TSO with one containing noise
            self.tso[self.ngrps*n:self.ngrps*n+self.ngrps] = ramp

        print('Noise model finished:', time.time()-start)

    def plot(self, ptype='data', idx=0, scale='linear', order=None, noise=True,
             traces=False, saturation=0.8, draw=True):
        """
        Plot a TSO frame

        Parameters
        ----------
        ptype: str
            The type of plot, ['data', 'snr', 'saturation']
        idx: int
            The frame index to plot
        scale: str
            Plot in linear or log scale
        orders: sequence
            The order to isolate
        noise: bool
            Plot with the noise model
        traces: bool
            Plot the traces used to generate the frame
        saturation: float
            The fraction of full well defined as saturation
        """
        if order:
            tso = getattr(self, 'tso_order{}_ideal'.format(order))
        else:
            if noise:
                tso = self.tso
            else:
                tso = self.tso_ideal

        # Get data for plotting
        vmax = int(np.nanmax(tso[tso < np.inf]))
        frame = np.array(tso[idx].data)

        # Modify the data
        if ptype == 'snr':
            frame = np.sqrt(frame.data)

        elif ptype == 'saturation':
            fullWell = 65536.0
            frame = frame > saturation * fullWell
            frame = frame.astype(int)

        else:
            pass

        # Make the figure
        height = 180 if self.subarray == 'SUBSTRIP96' else 225
        fig = figure(x_range=(0, frame.shape[1]), y_range=(0, frame.shape[0]),
                     tooltips=[("x", "$x"), ("y", "$y"), ("value", "@image")],
                     width=int(frame.shape[1]/2), height=height,
                     title='{}: Frame {}'.format(self.target, idx),
                     toolbar_location='above', toolbar_sticky=True)

        # Plot the frame
        if scale == 'log':
            frame[frame < 1.] = 1.
            color_mapper = LogColorMapper(palette="Viridis256", low=frame.min(), high=frame.max())
            fig.image(image=[frame], x=0, y=0, dw=frame.shape[1],
                      dh=frame.shape[0], color_mapper=color_mapper)
            color_bar = ColorBar(color_mapper=color_mapper, ticker=LogTicker(),
                                 orientation="horizontal", label_standoff=12,
                                 border_line_color=None, location=(0,0))

        else:
            color_mapper = LinearColorMapper(palette="Viridis256", low=frame.min(), high=frame.max())
            fig.image(image=[frame], x=0, y=0, dw=frame.shape[1],
                      dh=frame.shape[0], palette='Viridis256')
            color_bar = ColorBar(color_mapper=color_mapper,
                                 orientation="horizontal", label_standoff=12,
                                 border_line_color=None, location=(0,0))

        # Add color bar
        if ptype != 'saturation':
            fig.add_layout(color_bar, 'below')

        # Plot the polynomial too
        if traces:
            coeffs = trace_polynomials(subarray=self.subarray)
            X = np.linspace(0, 2048, 2048)

            # Order 1
            Y = np.polyval(coeffs[0], X)
            fig.line(X, Y, color='r')

            # Order 2
            Y = np.polyval(coeffs[1], X)
            fig.line(X, Y, color='r')

        if draw:
            show(fig)
        else:
            return fig

    def plot_slice(self, column, trace='tso', frame=0, order=None, **kwargs):
        """
        Plot a column of a frame to see the PSF in the cross dispersion direction

        Parameters
        ----------
        column: int, sequence
            The column index(es) to plot a light curve for
        trace: str
            The attribute name to plot
        frame: int
            The frame number to plot
        """
        if order is not None:
            tso = getattr(self, 'tso_order{}_ideal'.format(order))
        else:
            tso = self.tso

        # Transpose data
        flux = tso[frame].T

        # Turn one column into a list
        if isinstance(column, int):
            column = [column]

        # Make the figure
        fig = figure()
        for col in column:
            fig.line(np.arange(flux[col].size), flux[col], legend='Column {}'.format(col), **kwargs)

        show(fig)

    # def plot_ramp(self):
    #     """
    #     Plot the total flux on each frame to display the ramp
    #     """
    #     plt.figure()
    #     plt.plot(np.sum(self.tso, axis=(-1, -2)), ls='--', marker='o')
    #     plt.xlabel('Group')
    #     plt.ylabel('Count Rate [ADU/s]')
    #     plt.grid()
    #
    # def plot_lightcurve(self, column=None, time_unit='seconds', 
    #                     cmap=plt.cm.coolwarm, resolution_mult=20, 
    #                     theory_alpha=0.1):
    #     """
    #     Plot a lightcurve for each column index given
    #
    #     Parameters
    #     ----------
    #     column: int, float, sequence
    #         The integer column index(es) or float wavelength(s) in microns
    #         to plot as a light curve
    #     time_unit: string
    #         The string indicator for the units that the self.time array is in
    #         options: 'seconds', 'minutes', 'hours', 'days' (default)
    #     cmap: matplotlib.pyplot.cm entry
    #         A selection from the matplotlib.pyplot.cm color maps library
    #     resolution_mult: int
    #         The number of theoretical points to plot for each data point plotted here
    #     """
    #     # Get the scaled flux in each column for the last group in
    #     # each integration
    #     flux_cols = np.nansum(self.tso_ideal[self.ngrps-1::self.ngrps], axis=1)
    #     flux_cols = flux_cols/np.nanmax(flux_cols, axis=1)[:, None]
    #
    #     # Make it into an array
    #     if isinstance(column, (int, float)): column = [column]
    #
    #     if column is None: column = list(range(self.tso.shape[-1]))
    #
    #     n_colors = len(column)
    #     color_cycle = cmap(np.linspace(0, cmap.N, n_colors, dtype=int))
    #
    #     for kcol, col in tqdm(enumerate(column), total=len(column)):
    #
    #         # If it is an index
    #         if isinstance(col, int):
    #             lightcurve = flux_cols[:, col]
    #             label = 'Column {}'.format(col)
    #
    #         # Or assumed to be a wavelength in microns
    #         elif isinstance(col, float):
    #             waves = np.mean(self.wave[0], axis=0)
    #             lightcurve = [np.interp(col, waves, flux_col) for flux_col in flux_cols]
    #             label = '{} um'.format(col)
    #
    #         else:
    #             print('Please enter an index, astropy quantity, or array thereof.')
    #             return
    #
    #         # Plot the theoretical light curve
    #         # if self.rp is not None:
    #         #     if time_unit not in ['seconds', 'minutes', 'hours', 'days']:
    #         #         raise ValueError("time_unit must be either 'seconds', 'hours', or 'days']")
    #         #
    #         #     time = np.linspace(min(self.time), max(self.time), self.ngrps*self.nints*resolution_mult)
    #         #
    #         #     days_to_seconds = 86400.
    #         #     if time_unit == 'seconds':
    #         #         time /= days_to_seconds
    #         #     if time_unit == 'minutes':
    #         #         time /= days_to_seconds / 60
    #         #     if time_unit == 'hours':
    #         #         time /= days_to_seconds / 3600
    #         #
    #         #     tmodel = batman.TransitModel(self.tmodel, time)
    #         #     tmodel.rp = self.rp[col]
    #         #     theory = tmodel.light_curve(tmodel)
    #         #     theory *= max(lightcurve)/max(theory)
    #         #
    #         #     plt.plot(time, theory, label=label+' model', marker='.', ls='--', color=color_cycle[kcol%n_colors], alpha=theory_alpha)
    #
    #         data_time = self.time[self.ngrps-1::self.ngrps].copy()
    #
    #         if time_unit == 'seconds':
    #             data_time /= days_to_seconds
    #         if time_unit == 'minutes':
    #             data_time /= days_to_seconds / 60
    #         if time_unit == 'hours':
    #             data_time /= days_to_seconds / 3600
    #
    #         plt.plot(data_time, lightcurve, label=label, marker='o', ls='None', color=color_cycle[kcol%n_colors])
    #
    #     plt.legend(loc=0, frameon=False)
    #
    # def plot_spectrum(self, frame=0, order=None):
    #     """
    #     Parameters
    #     ----------
    #     frame: int
    #         The frame number to plot
    #     """
    #     if order is not None:
    #         tso = getattr(self, 'tso_order{}_ideal'.format(order))
    #     else:
    #         tso = self.tso
    #
    #     # Get extracted spectrum (Column sum for now)
    #     wave = np.mean(self.wave[0], axis=0)
    #     flux = np.sum(tso[frame].data, axis=0)
    #     response = 1./self.photom_order1
    #
    #     # Convert response in [mJy/ADU/s] to [Flam/ADU/s] then invert so
    #     # that we can convert the flux at each wavelegth into [ADU/s]
    #     flux *= response/self.time[np.mod(self.ngrps, frame)]
    #
    #     # Plot it along with input spectrum
    #     plt.figure(figsize=(13, 5))
    #     plt.loglog(wave, flux, label='Extracted')
    #     plt.loglog(*self.star, label='Injected')
    #     plt.xlim(wave[0]*0.95, wave[-1]*1.05)
    #     plt.ylim(np.min(flux)*0.9, np.max(flux)*1.1)
    #     plt.legend()

    def to_fits(self, outfile):
        """
        Save the data to a JWST pipeline ingestible FITS file

        Parameters
        ----------
        outfile: str
            The path of the output file
        """
        # Make the cards
        cards = [('DATE', datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S"), 'Date file created yyyy-mm-ddThh:mm:ss, UTC'), 
                ('FILENAME', outfile, 'Name of the file'), 
                ('DATAMODL', 'RampModel', 'Type of data model'), 
                ('ORIGIN', 'STScI', 'Institution responsible for creating FITS file'), 
                ('TIMESYS', 'UTC', 'principal time system for time-related keywords'), 
                ('FILETYPE', 'uncalibrated', 'Type of data in the file'), 
                ('SDP_VER', '2016_1', 'data processing software version number'), 
                ('PRD_VER', 'PRDDEVSOC-D-012', 'S&OC PRD version number used in data processing'), 
                ('TELESCOP', 'JWST', 'Telescope used to acquire data'), 
                ('RADESYS', 'ICRS', 'Name of the coordinate reference frame'), 
                ('', '', ''), 
                ('COMMENT', '/ Program information', ''), 
                ('TITLE', 'UNKNOWN', 'Proposal title'), 
                ('PI_NAME', 'N/A', 'Principal investigator name'), 
                ('CATEGORY', 'UNKNOWN', 'Program category'), 
                ('SUBCAT', '', 'Program sub-category'), 
                ('SCICAT', '', 'Science category assigned during TAC process'), 
                ('CONT_ID', 0, 'Continuation of previous program'), 
                ('', '', ''), 
                ('COMMENT', '/ Observation identifiers', ''), 
                ('DATE-OBS', self.obs_date, 'UT date at start of exposure'), 
                ('TIME-OBS', self.obs_time, 'UT time at the start of exposure'), 
                ('OBS_ID', 'V87600007001P0000000002102', 'Programmatic observation identifier'), 
                ('VISIT_ID', '87600007001', 'Visit identifier'), 
                ('PROGRAM', '87600', 'Program number'), 
                ('OBSERVTN', '001', 'Observation number'), 
                ('VISIT', '001', 'Visit number'), 
                ('VISITGRP', '02', 'Visit group identifier'), 
                ('SEQ_ID', '1', 'Parallel sequence identifier'), 
                ('ACT_ID', '02', 'Activity identifier'), 
                ('EXPOSURE', '1', 'Exposure request number'), 
                ('', '', ''), 
                ('COMMENT', '/ Visit information', ''), 
                ('TEMPLATE', 'NIRISS SOSS', 'Proposal instruction template used'), 
                ('OBSLABEL', 'Observation label', 'Proposer label for the observation'), 
                ('VISITYPE', '', 'Visit type'), 
                ('VSTSTART', self.obs_date, 'UTC visit start time'), 
                ('WFSVISIT', '', 'Wavefront sensing and control visit indicator'), 
                ('VISITSTA', 'SUCCESSFUL', 'Status of a visit'), 
                ('NEXPOSUR', 1, 'Total number of planned exposures in visit'), 
                ('INTARGET', False, 'At least one exposure in visit is internal'), 
                ('TARGOOPP', False, 'Visit scheduled as target of opportunity'), 
                ('', '', ''), 
                ('COMMENT', '/ Target information', ''), 
                ('TARGPROP', '', "Proposer's name for the target"), 
                ('TARGNAME', self.target, 'Standard astronomical catalog name for tar'), 
                ('TARGTYPE', 'FIXED', 'Type of target (fixed, moving, generic)'), 
                ('TARG_RA', 175.5546225, 'Target RA at mid time of exposure'), 
                ('TARG_DEC', 26.7065694, 'Target Dec at mid time of exposure'), 
                ('TARGURA', 0.01, 'Target RA uncertainty'), 
                ('TARGUDEC', 0.01, 'Target Dec uncertainty'), 
                ('PROP_RA', 175.5546225, 'Proposer specified RA for the target'), 
                ('PROP_DEC', 26.7065694, 'Proposer specified Dec for the target'), 
                ('PROPEPOC', '2000-01-01 00:00:00', 'Proposer specified epoch for RA and Dec'), 
                ('', '', ''), 
                ('COMMENT', '/ Exposure parameters', ''), 
                ('INSTRUME', 'NIRISS', 'Identifier for niriss used to acquire data'), 
                ('DETECTOR', 'NIS', 'ASCII Mnemonic corresponding to the SCA_ID'), 
                ('LAMP', 'NULL', 'Internal lamp state'), 
                ('FILTER', self.filter, 'Name of the filter element used'), 
                ('PUPIL', 'GR700XD', 'Name of the pupil element used'), 
                ('FOCUSPOS', 0.0, 'Focus position'), 
                ('', '', ''), 
                ('COMMENT', '/ Exposure information', ''), 
                ('PNTG_SEQ', 2, 'Pointing sequence number'), 
                ('EXPCOUNT', 0, 'Running count of exposures in visit'), 
                ('EXP_TYPE', 'NIS_SOSS', 'Type of data in the exposure'), 
                ('', '', ''), 
                ('COMMENT', '/ Exposure times', ''), 
                ('EXPSTART', self.time[0], 'UTC exposure start time'), 
                ('EXPMID', self.time[len(self.time)//2], 'UTC exposure mid time'), 
                ('EXPEND', self.time[-1], 'UTC exposure end time'), 
                ('READPATT', 'NISRAPID', 'Readout pattern'), 
                ('NINTS', self.nints, 'Number of integrations in exposure'), 
                ('NGROUPS', self.ngrps, 'Number of groups in integration'), 
                ('NFRAMES', self.nframes, 'Number of frames per group'), 
                ('GROUPGAP', 0, 'Number of frames dropped between groups'), 
                ('NSAMPLES', 1, 'Number of A/D samples per pixel'), 
                ('TSAMPLE', 10.0, 'Time between samples (microsec)'), 
                ('TFRAME', mt.FRAME_TIMES[self.subarray], 'Time in seconds between frames'), 
                ('TGROUP', mt.FRAME_TIMES[self.subarray], 'Delta time between groups (s)'), 
                ('EFFINTTM', 15.8826, 'Effective integration time (sec)'), 
                ('EFFEXPTM', 15.8826, 'Effective exposure time (sec)'), 
                ('CHRGTIME', 0.0, 'Charge accumulation time per integration (sec)'), 
                ('DURATION', self.time[-1]-self.time[0], 'Total duration of exposure (sec)'), 
                ('NRSTSTRT', self.nresets, 'Number of resets at start of exposure'), 
                ('NRESETS', self.nresets, 'Number of resets between integrations'), 
                ('FWCPOS', float(75.02400207519531), ''), 
                ('PWCPOS', float(245.6344451904297), ''), 
                ('ZEROFRAM', False, 'Zero frame was downlinkws separately'), 
                ('DATAPROB', False, 'Science telemetry indicated a problem'), 
                ('SCA_NUM', 496, 'Sensor Chip Assembly number'), 
                ('DATAMODE', 91, 'post-processing method used in FPAP'), 
                ('COMPRSSD', False, 'data compressed on-board (T/F)'), 
                ('SUBARRAY', True, 'Subarray pattern name'), 
                # ('SUBARRAY', self.subarray, 'Subarray pattern name'), 
                ('SUBSTRT1', 1, 'Starting pixel in axis 1 direction'), 
                ('SUBSTRT2', 1793, 'Starting pixel in axis 2 direction'), 
                ('SUBSIZE1', self.ncols, 'Number of pixels in axis 1 direction'), 
                ('SUBSIZE2', self.nrows, 'Number of pixels in axis 2 direction'), 
                ('FASTAXIS', -2, 'Fast readout axis direction'), 
                ('SLOWAXIS', -1, 'Slow readout axis direction'), 
                ('COORDSYS', '', 'Ephemeris coordinate system'), 
                ('EPH_TIME', 57403, 'UTC time from ephemeris start time (sec)'), 
                ('JWST_X', 1462376.39634336, 'X spatial coordinate of JWST (km)'), 
                ('JWST_Y', -178969.457007469, 'Y spatial coordinate of JWST (km)'), 
                ('JWST_Z', -44183.7683640854, 'Z spatial coordinate of JWST (km)'), 
                ('JWST_DX', 0.147851665036734, 'X component of JWST velocity (km/sec)'), 
                ('JWST_DY', 0.352194454527743, 'Y component of JWST velocity (km/sec)'), 
                ('JWST_DZ', 0.032553742839182, 'Z component of JWST velocity (km/sec)'), 
                ('APERNAME', 'NIS-CEN', 'PRD science aperture used'), 
                ('PA_APER', -290.1, 'Position angle of aperture used (deg)'), 
                ('SCA_APER', -697.500000000082, 'SCA for intended target'), 
                ('DVA_RA', 0.0, 'Velocity aberration correction RA offset (rad)'), 
                ('DVA_DEC', 0.0, 'Velocity aberration correction Dec offset (rad)'), 
                ('VA_SCALE', 0.0, 'Velocity aberration scale factor'), 
                ('BARTDELT', 0.0, 'Barycentric time correction'), 
                ('BSTRTIME', 0.0, 'Barycentric exposure start time'), 
                ('BENDTIME', 0.0, 'Barycentric exposure end time'), 
                ('BMIDTIME', 0.0, 'Barycentric exposure mid time'), 
                ('HELIDELT', 0.0, 'Heliocentric time correction'), 
                ('HSTRTIME', 0.0, 'Heliocentric exposure start time'), 
                ('HENDTIME', 0.0, 'Heliocentric exposure end time'), 
                ('HMIDTIME', 0.0, 'Heliocentric exposure mid time'), 
                ('WCSAXES', 2, 'Number of WCS axes'), 
                ('CRPIX1', 1955.0, 'Axis 1 coordinate of the reference pixel in the'), 
                ('CRPIX2', 1199.0, 'Axis 2 coordinate of the reference pixel in the'), 
                ('CRVAL1', 175.5546225, 'First axis value at the reference pixel (RA in'), 
                ('CRVAL2', 26.7065694, 'Second axis value at the reference pixel (RA in'), 
                ('CTYPE1', 'RA---TAN', 'First axis coordinate type'), 
                ('CTYPE2', 'DEC--TAN', 'Second axis coordinate type'), 
                ('CUNIT1', 'deg', 'units for first axis'), 
                ('CUNIT2', 'deg', 'units for second axis'), 
                ('CDELT1', 0.065398, 'first axis increment per pixel, increasing east'), 
                ('CDELT2', 0.065893, 'Second axis increment per pixel, increasing nor'), 
                ('PC1_1', -0.5446390350150271, 'linear transformation matrix element cos(theta)'), 
                ('PC1_2', 0.8386705679454239, 'linear transformation matrix element -sin(theta'), 
                ('PC2_1', 0.8386705679454239, 'linear transformation matrix element sin(theta)'), 
                ('PC2_2', -0.5446390350150271, 'linear transformation matrix element cos(theta)'), 
                ('S_REGION', '', 'spatial extent of the observation, footprint'), 
                ('GS_ORDER', 0, 'index of guide star within listed of selected g'), 
                ('GSSTRTTM', '1999-01-01 00:00:00', 'UTC time when guide star activity started'), 
                ('GSENDTIM', '1999-01-01 00:00:00', 'UTC time when guide star activity completed'), 
                ('GDSTARID', '', 'guide star identifier'), 
                ('GS_RA', 0.0, 'guide star right ascension'), 
                ('GS_DEC', 0.0, 'guide star declination'), 
                ('GS_URA', 0.0, 'guide star right ascension uncertainty'), 
                ('GS_UDEC', 0.0, 'guide star declination uncertainty'), 
                ('GS_MAG', 0.0, 'guide star magnitude in FGS detector'), 
                ('GS_UMAG', 0.0, 'guide star magnitude uncertainty'), 
                ('PCS_MODE', 'COARSE', 'Pointing Control System mode'), 
                ('GSCENTX', 0.0, 'guide star centroid x postion in the FGS ideal'), 
                ('GSCENTY', 0.0, 'guide star centroid x postion in the FGS ideal'), 
                ('JITTERMS', 0.0, 'RMS jitter over the exposure (arcsec).'), 
                ('VISITEND', '2017-03-02 15:58:45.36', 'Observatory UTC time when the visit st'), 
                ('WFSCFLAG', '', 'Wavefront sensing and control visit indicator'), 
                ('BSCALE', 1, ''), 
                ('BZERO', 32768, ''), 
                ('NCOLS', float(self.nrows-1), ''), 
                ('NROWS', float(self.ncols-1), '')]

        # Make the header
        prihdr = fits.Header()
        for card in cards:
            prihdr.append(card, end=True)

        # Store the header in the object too
        self.header = prihdr

        # Put data into detector coordinates
        data = np.swapaxes(self.tso, 1, 2)[:, :, ::-1]

        # Make the HDUList
        prihdu = fits.PrimaryHDU(data=data, header=prihdr)

        # Write the file
        prihdu.writeto(outfile, overwrite=True)

        print('File saved as', outfile)


class TestTSO(TSO):
    """Generate a test object for quick access"""
    def __init__(self, subarray='SUBSTRIP256', filt='CLEAR'):
        """Get the test data and load the object"""
        file = resource_filename('awesimsoss', 'files/scaled_spectrum.txt')
        star = np.genfromtxt(file, unpack=True)
        star1D = [star[0]*q.um, (star[1]*q.W/q.m**2/q.um).to(q.erg/q.s/q.cm**2/q.AA)]
        super().__init__(ngrps=2, nints=2, star=star1D, subarray=subarray, filt=filt)
        self.run_simulation()
