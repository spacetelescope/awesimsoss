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

from astroquery.simbad import Simbad
import numpy as np
from bokeh.plotting import figure, show
from bokeh.models import LogColorMapper, LogTicker, LinearColorMapper, ColorBar, Span
from bokeh.layouts import column
import astropy.units as q
import astropy.constants as ac
from astropy.io import fits
from astropy.modeling.models import BlackBody1D
from astropy.modeling.blackbody import FLAM
from astropy.coordinates import SkyCoord

try:
    from jwst.datamodels import RampModel
except ImportError:
    print("Could not import `jwst` package. Functionality limited.")

try:
    import batman
except ImportError:
    print("Could not import `batman` package. Functionality limited.")

from . import generate_darks as gd
from . import make_trace as mt
from . import utils


warnings.simplefilter('ignore')


class TSO(object):
    """
    Generate NIRISS SOSS time series observations
    """
    def __init__(self, ngrps, nints, star, snr=700, filter='CLEAR',
                 subarray='SUBSTRIP256', orders=[1, 2], t0=0, nresets=0,
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
        from awesimsoss import TSO
        import astropy.units as q
        from pkg_resources import resource_filename
        star = np.genfromtxt(resource_filename('awesimsoss', 'files/scaled_spectrum.txt'), unpack=True)
        star1D = [star[0]*q.um, (star[1]*q.W/q.m**2/q.um).to(q.erg/q.s/q.cm**2/q.AA)]

        # Initialize simulation
        tso = TSO(ngrps=3, nints=10, star=star1D)
        """
        self.verbose = verbose

        # Check the star units
        self._check_star(star)

        # Set static values
        self.gain = 1.61

        # Set instance attributes for the exposure
        self.t0 = t0
        self.ngrps = ngrps
        self.nints = nints
        self.nresets = nresets
        self.nframes = (self.nresets+self.ngrps)*self.nints
        self.obs_date = str(datetime.datetime.now())
        self.obs_time = str(datetime.datetime.now())
        self.orders = orders
        self.filter = filter
        self.header = ''
        self.snr = snr
        self.model_grid = None
        self.subarray = subarray

        # Reset data based on subarray and observation settings
        self._reset_data()
        self._reset_time()

        # Meta data for the target
        self.target = target
        self.title = title or '{} Simulation'.format(self.target)

        # Set instance attributes for the target
        self._ld_coeffs = np.zeros((3, 2048, 2))
        self.planet = None
        self.tmodel = None

        # Generate the psfs
        self._reset_psfs()

        # Generate the response function
        self._reset_response()

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

        # Load the reference files
        photon_yield = fits.getdata(resource_filename('awesimsoss', 'files/photon_yield_dms.fits'))
        pedestal = fits.getdata(resource_filename('awesimsoss', 'files/substrip256pedestaldms.fits'))
        pca0_file = resource_filename('awesimsoss', 'files/niriss_pca0.fits')
        nonlinearity = fits.getdata(resource_filename('awesimsoss', 'files/substrip256_forward_coefficients_dms.fits'))
        zodi = fits.getdata(resource_filename('awesimsoss', 'files/soss_zodiacal_background_scaled.fits'))
        darksignal = fits.getdata(resource_filename('awesimsoss', 'files/substrip256signaldms.fits'))*self.gain

        # Updates if SUBSTRIP96
        if self.subarray == 'SUBSTRIP96':

            # Make slice from FULL frame
            slc = slice(160, 256)

            # Trim SUBSTRIP256 photon yield
            photon_yield = photon_yield[:, :96, :]

        # Updates if SUBSTRIP256
        elif self.subarray == 'SUBSTRIP256':

            # Make slice from FULL frame
            slc = slice(0, 256)

        # Updates if FULL
        else:

            # Make slice from FULL frame
            slc = slice(0, 2048)

            # Pad SUBSTRIP256 photon yield with ones since there is no wavelength information in those pixels
            full_py = np.ones(self.dims)
            full_py[:, :256, :] = photon_yield
            photon_yield = full_py

        # Trim FULL frame reference files
        pedestal = pedestal[slc, :]
        nonlinearity = nonlinearity[:, slc, :]
        zodi = zodi[slc, :]
        darksignal = darksignal[slc, :]

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

        print('Noise model finished:', round(time.time()-start, 3), 's')

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

    def _check_star(self, star):
        """Make sure the input star has units

        Parameters
        ----------
        star: sequence
            The [W, F] or [W, F, E] of the star to simulate

        Returns
        -------
        bool
            True or False
        """
        # Check star is a sequence of length 2 or 3
        if not isinstance(star, (list, tuple)) or not len(star) in [2, 3]:
            raise ValueError(type(star), ': Star input must be a sequence of [W, F] or [W, F, E]')

        # Check star has units
        if not all([isinstance(i, q.quantity.Quantity) for i in star]):
            types = ', '.join([type(i) for i in star])
            raise ValueError('[{}]: Spectrum must be in astropy units'.format(types))

        # Check the units
        if not star[0].unit.is_equivalent(q.um):
            raise ValueError(star[0].unit, ': Wavelength must be in units of distance')

        if not all([i.unit.is_equivalent(q.erg/q.s/q.cm**2/q.AA) for i in star[1:]]):
            raise ValueError(star[1].unit, ': Flux density must be in units of F_lambda')

        # Good to go
        self.star = star

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
        self._reset_response()

    @property
    def info(self):
        """Summary table for the observation settings"""
        # Pull out relevant attributes
        track = ['_ncols', '_nrows', '_nints', '_ngrps', '_nresets', '_subarray', '_filter', '_t0', '_orders']
        settings = {key[1:]: val for key, val in self.__dict__.items() if key in track}
        return settings

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
        elif str(type(tmodel)) == "<class 'batman.transitmodel.TransitModel'>" and str(type(self.model_grid)) == "<class 'exoctk.modelgrid.ModelGrid'>":
            self.ld_coeffs = [mt.generate_SOSS_ldcs(self.avg_wave[order-1], coeffs.limb_dark, [getattr(coeffs, p) for p in ['teff', 'logg', 'feh']], model_grid=self.model_grid) for order in self.orders]

        else:
            raise ValueError('Please set ld_coeffs with a 3D array or batman.transitmodel.TransitModel.')

    @property
    def ncols(self):
        """Getter for the number of columns"""
        return self._ncols

    @ncols.setter
    def ncols(self, err):
        """Error when trying to change the number of columns
        """
        raise ValueError("The number of columns is fixed by setting the 'subarray' attribute.")

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
        raise ValueError("The number of rows is fixed by setting the 'subarray' attribute.")

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
            Plot scale, ['linear', 'log']
        order: sequence
            The order to isolate
        noise: bool
            Plot with the noise model
        traces: bool
            Plot the traces used to generate the frame
        saturation: float
            The fraction of full well defined as saturation
        """
        if order in [1, 2]:
            tso = getattr(self, 'tso_order{}_ideal'.format(order))
        else:
            if noise:
                tso = self.tso
            else:
                tso = self.tso_ideal

        # Get data for plotting
        vmax = int(np.nanmax(tso[tso < np.inf]))
        frame = np.array(tso.reshape(self.dims3)[idx].data)

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
        height = 180 if self.subarray == 'SUBSTRIP96' else 800 if self.subarray == 'FULL' else 225
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
                                 border_line_color=None, location=(0, 0))

        else:
            color_mapper = LinearColorMapper(palette="Viridis256", low=frame.min(), high=frame.max())
            fig.image(image=[frame], x=0, y=0, dw=frame.shape[1],
                      dh=frame.shape[0], palette='Viridis256')
            color_bar = ColorBar(color_mapper=color_mapper,
                                 orientation="horizontal", label_standoff=12,
                                 border_line_color=None, location=(0, 0))

        # Add color bar
        if ptype != 'saturation':
            fig.add_layout(color_bar, 'below')

        # Plot the polynomial too
        if traces:
            X = np.linspace(0, 2048, 2048)

            # Order 1
            Y = np.polyval(self.coeffs[0], X)
            fig.line(X, Y, color='red')

            # Order 2
            Y = np.polyval(self.coeffs[1], X)
            fig.line(X, Y, color='red')

        if draw:
            show(fig)
        else:
            return fig

    def plot_slice(self, col, idx=0, order=None, noise=False, **kwargs):
        """
        Plot a column of a frame to see the PSF in the cross dispersion direction

        Parameters
        ----------
        col: int, sequence
            The column index(es) to plot
        idx: int
            The frame index to plot
        order: sequence
            The order to isolate
        noise: bool
            Plot with the noise model
        """
        if order in [1, 2]:
            tso = getattr(self, 'tso_order{}_ideal'.format(order))
        else:
            if noise:
                tso = self.tso
            else:
                tso = self.tso_ideal

        # Transpose data
        flux = tso.reshape(self.dims3)[idx].T

        # Turn one column into a list
        if isinstance(col, int):
            col = [col]

        # Get the data
        dfig = self.plot(ptype='data', idx=idx, order=order, draw=False, noise=noise, **kwargs)

        # Make the figure
        fig = figure(width=1024, height=500)
        fig.xaxis.axis_label = 'Row'
        fig.yaxis.axis_label = 'Count Rate [ADU/s]'
        fig.legend.click_policy = 'mute'
        for c in col:
            color = next(utils.COLORS)
            fig.line(np.arange(flux[c, :].size), flux[c, :], color=color, legend='Column {}'.format(c))
            vline = Span(location=c, dimension='height', line_color=color, line_width=3)
            dfig.add_layout(vline)

        show(column(fig, dfig))

    def plot_ramp(self):
        """
        Plot the total flux on each frame to display the ramp
        """
        ramp = figure()
        x = range(self.dims3[0])
        y = np.sum(self.tso.reshape(self.dims3), axis=(-1, -2))
        ramp.circle(x, y, size=12)
        ramp.xaxis.axis_label = 'Group'
        ramp.yaxis.axis_label = 'Count Rate [ADU/s]'

        show(ramp)

    def plot_lightcurve(self, column=None, time_unit='s', resolution_mult=20):
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
        """
        # Check time_units
        if time_unit not in ['s', 'min', 'h', 'd']:
            raise ValueError("time_unit must be 's', 'min', 'h' or 'd']")

        # Get the scaled flux in each column for the last group in
        # each integration
        flux_cols = np.nansum(self.tso_ideal.reshape(self.dims3)[self.ngrps-1::self.ngrps], axis=1)
        flux_cols = flux_cols/np.nanmax(flux_cols, axis=1)[:, None]

        # Make it into an array
        if isinstance(column, (int, float)):
            column = [column]

        if column is None:
            column = list(range(self.tso.shape[-1]))

        # Make the figure
        lc = figure()

        for kcol, col in enumerate(column):

            color = next(utils.COLORS)

            # If it is an index
            if isinstance(col, int):
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
                time = time*q.d.to(time_unit)

                tmodel = batman.TransitModel(self.tmodel, time)
                tmodel.rp = self.rp[col]
                theory = tmodel.light_curve(tmodel)
                theory *= max(lightcurve)/max(theory)

                lc.line(time, theory, legend=label+' model', color=color, alpha=0.1)

            # Convert datetime
            data_time = self.time[self.ngrps-1::self.ngrps].copy()
            data_time*q.d.to(time_unit)

            # Plot the lightcurve
            lc.circle(data_time, lightcurve, legend=label, color=color)

        lc.xaxis.axis_label = 'Time [{}]'.format(time_unit)
        lc.yaxis.axis_label = 'Transit Depth'
        show(lc)

    def plot_spectrum(self, frame=0, order=None, noise=False, scale='log'):
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
        """
        if order in [1, 2]:
            tso = getattr(self, 'tso_order{}_ideal'.format(order))
        else:
            if noise:
                tso = self.tso
            else:
                tso = self.tso_ideal

        # Get extracted spectrum (Column sum for now)
        wave = np.mean(self.wave[0], axis=0)
        flux_out = np.sum(tso.reshape(self.dims3)[frame].data, axis=0)
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

        show(column(spec, res))

    def _reset_data(self):
        """Reset the results to all zeros"""
        # Check that all the appropriate values have been initialized
        if all([i in self.info for i in ['nints', 'ngrps', 'nrows', 'ncols']]):

            # Update the dimensions
            self.dims = (self.nints, self.ngrps, self.nrows, self.ncols)
            self.dims3 = (self.nints*self.ngrps, self.nrows, self.ncols)

            # Reset the results
            self.tso = None
            self.tso_ideal = None
            self.tso_order1_ideal = None
            self.tso_order2_ideal = None

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

    def _reset_response(self):
        """Generate the relative response function for each order"""
        # Check that all the appropriate values have been initialized
        if all([i in self.info for i in ['filter', 'subarray']]):

            for order in self.orders:

                # Get the wavelength map
                wave = self.avg_wave[order-1]

                # Get relative spectral response for the order (from
                # /grp/crds/jwst/references/jwst/jwst_niriss_photom_0028.fits)
                throughput = self.photom[self.photom['order'] == order]
                ph_wave = throughput.wavelength[throughput.wavelength > 0][1:-2]
                ph_resp = throughput.relresponse[throughput.wavelength > 0][1:-2]
                response = np.interp(wave, ph_wave, ph_resp)

                # Convert response in [mJy/ADU/s] to [Flam/ADU/s] then invert so
                # that we can convert the flux at each wavelegth into [ADU/s]
                response = self.frame_time/(response*q.mJy*ac.c/(wave*q.um)**2).to(self.star[1].unit).value
                setattr(self, 'order{}_response'.format(order), response)

    def _reset_psfs(self):
        """Scale the psf for each detector column to the flux from the 1D spectrum"""
        # Check that all the appropriate values have been initialized
        if all([i in self.info for i in ['filter', 'subarray']]):

            for order in self.orders:

                flux = np.interp(self.avg_wave[order-1], self.star[0], self.star[1], left=0, right=0)
                cube = mt.SOSS_psf_cube(filt=self.filter, order=order, subarray=self.subarray)*flux[:, None, None]
                setattr(self, 'order{}_flux'.format(order), flux)
                setattr(self, 'order{}_psfs'.format(order), cube)

    def run_simulation(self, planet=None, tmodel=None, ld_coeffs=None, time_unit='days', 
                       ld_profile='quadratic', model_grid=None, n_jobs=-1, verbose=True):
        """
        Generate the simulated 4D ramp data given the initialized TSO object

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
        self._reset_data()

        # If there is a planet transmission spectrum but no LDCs generate them
        is_tmodel = str(type(tmodel)) == "<class 'batman.transitmodel.TransitModel'>"
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

            # Convert seconds to days, in order to match the Period and
            # T0 parameters
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
            elif str(type(model_grid)) == "<class 'exoctk.modelgrid.ModelGrid'>" and changed:

                # Try to set the model grid
                self.model_grid = model_grid
                self.ld_coeffs = tmodel

            else:
                pass

        # Generate simulation for each order
        for order in self.orders:

            # Get the wavelength map
            wave = self.avg_wave[order-1]

            # Get the psf cube and filter response function
            psfs = getattr(self, 'order{}_psfs'.format(order))
            response = getattr(self, 'order{}_response'.format(order))

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
            if verbose:
                print('Calculating order {} light curves...'.format(order))
                start = time.time()

            # Generate the lightcurves at each wavelength
            pool = ThreadPool(n_jobs)
            func = partial(mt.psf_lightcurve, time=self.time, tmodel=self.tmodel)
            data = list(zip(wave, psfs, response, ld_coeffs, self.rp))
            lightcurves = np.asarray(pool.starmap(func, data), dtype=np.float64)
            pool.close()
            pool.join()

            # Reshape to make frames
            lightcurves = lightcurves.swapaxes(0, 1)

            # Multiply by the frame time to convert to [ADU]
            ft = np.tile(self.time[:self.ngrps], self.nints)
            lightcurves *= ft[:, None, None, None]

            # Generate TSO frames
            if verbose:
                print('Lightcurves finished:', round(time.time()-start, 3), 's')
                print('Constructing order {} traces...'.format(order))
                start = time.time()

            # Make the 2048*N lightcurves into N frames
            pool = ThreadPool(n_jobs)
            frames = np.asarray(pool.map(mt.make_frame, lightcurves))
            pool.close()
            pool.join()

            if verbose:
                # print('Total flux after warp:', np.nansum(all_frames[0]))
                print('Order {} traces finished:'.format(order), round(time.time()-start, 3), 's')

            # Add it to the individual order
            setattr(self, 'tso_order{}_ideal'.format(order), np.array(frames))

        # Add to the master TSO
        self.tso_ideal = np.sum([getattr(self, 'tso_order{}_ideal'.format(order)) for order in self.orders], axis=0)

        # Trim SUBSTRIP256 array if SUBSTRIP96
        if self.subarray == 'SUBSTRIP96':
            for arr in ['tso', 'tso_ideal', 'tso_order1_ideal', 'tso_order2_ideal']:
                setattr(self, arr, getattr(self, arr)[:, :self.nrows, :])

        # Expand SUBSTRIP256 array if FULL frame
        if self.subarray == 'FULL':
            for arr in ['tso', 'tso_ideal', 'tso_order1_ideal', 'tso_order2_ideal']:
                full = np.zeros(self.dims3)
                full[:, :256, :] = getattr(self, arr)
                setattr(self, arr, full)

        # Make ramps and add noise to the observations using Kevin Volk's
        # dark ramp simulator
        self.tso = self.tso_ideal.copy()
        self.add_noise()

        # Reshape into (nints, ngrps, y, x)
        self.tso = self.tso.reshape(self.dims)
        self.tso_ideal = self.tso_ideal.reshape(self.dims)
        self.tso_order1_ideal = self.tso_order1_ideal.reshape(self.dims)
        self.tso_order2_ideal = self.tso_order2_ideal.reshape(self.dims)

        # Simulate reference pixels
        self.add_refpix()

        if verbose:
            print('\nTotal time:', round(time.time()-begin, 3), 's')

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
        self.subarray_specs = utils.subarray(subarr)

        # Set the dependent quantities
        self._ncols = 2048
        self._nrows = self.subarray_specs.get('y')
        self.wave = utils.wave_solutions(subarr)
        self.avg_wave = np.mean(self.wave, axis=1)
        self.coeffs = mt.trace_polynomials(subarray=subarr)

        # Reset the data and time arrays
        self._reset_data()
        self._reset_time()

        # Reset the relative response function and the psfs
        self._reset_response()
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

        # Set the subarray
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
                    print("Coordinates for '{}' found in Simbad!".format(self.target))
            except TypeError:
                if self.verbose:
                    print("Could not resolve target '{}' in Simbad. Using ra={}, dec={}.".format(self.target, self.ra, self.dec))
                    print("Set coordinates manually by updating 'ra' and 'dec' attributes.")


    def to_fits(self, outfile, all_data=False):
        """
        Save the data to a JWST pipeline ingestible FITS file

        Parameters
        ----------
        outfile: str
            The path of the output file
        """
        try:

            # Make a RampModel
            data = self.tso
            mod = RampModel(data=data, groupdq=np.zeros_like(data), pixeldq=np.zeros((self.nrows, self.ncols)), err=np.zeros_like(data))
            pix = utils.subarray(self.subarray)

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

            # Save the file
            mod.save(outfile, overwrite=True)

            print('File saved as', outfile)

        except:
            print("Sorry, I could not save this simulation to file. Check that you have the `jwst` pipeline installed.")


class TestTSO(TSO):
    """Generate a test object for quick access"""
    def __init__(self, ngrps=2, nints=2, filter='CLEAR', subarray='SUBSTRIP256', run=True, **kwargs):
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
        """
        # Get stored data
        file = resource_filename('awesimsoss', 'files/scaled_spectrum.txt')
        star = np.genfromtxt(file, unpack=True)
        star1D = [star[0]*q.um, (star[1]*q.W/q.m**2/q.um).to(q.erg/q.s/q.cm**2/q.AA)]

        # Initialize base class
        super().__init__(ngrps=ngrps, nints=nints, star=star1D, subarray=subarray, filter=filter, **kwargs)

        # Run the simulation
        if run:
            self.run_simulation()


class BlackbodyTSO(TSO):
    """Generate a test object with a blackbody spectrum"""
    def __init__(self, ngrps=2, nints=2, teff=1800, filter='CLEAR', subarray='SUBSTRIP256', run=True, **kwargs):
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
        """
        # Generate a blackbody at the given temperature
        bb = BlackBody1D(temperature=teff*q.K)
        wav = np.linspace(0.5, 2.9, 1000) * q.um
        flux = bb(wav).to(FLAM, q.spectral_density(wav))*1E-8

        # Initialize base class
        super().__init__(ngrps=ngrps, nints=nints, star=[wav, flux], subarray=subarray, filter=filter, **kwargs)

        # Run the simulation
        if run:
            self.run_simulation()
