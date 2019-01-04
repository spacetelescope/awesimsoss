"""
A module to generate simulated 2D time-series SOSS data

Authors: Joe Filippazzo
"""

import os
from pkg_resources import resource_filename
import multiprocessing
import time
from functools import partial
import warnings
import copy

import numpy as np
import bokeh
from astropy.io import fits
import webbpsf
from svo_filters import svo
from scipy.interpolate import interp1d
from scipy.ndimage.interpolation import rotate
from scipy.interpolate import interp2d, RectBivariateSpline

try:
    import batman
except ImportError:
    print("Could not import `batman` package. Functionality limited.")

warnings.simplefilter('ignore')

FRAME_TIMES = {'SUBSTRIP96':2.213, 'SUBSTRIP256':5.491, 'FULL':10.737}
SUBARRAY_Y = {'SUBSTRIP96':96, 'SUBSTRIP256':256, 'FULL':2048}


def make_frame(psfs):
    """
    Generate a frame from an array of psfs

    Parameters
    ----------
    psfs: sequence
        An array of psfs of shape (2048, 76, 76)

    Returns
    -------
    np.ndarray
        An array of the SOSS psf at 2048 wavelengths for each order
    """
    # Empty frame
    frame = np.zeros((256, 2124))

    # Add each psf
    for n, psf in enumerate(psfs):
        frame[:, n:n+76] += psf

    return frame[:, 38:-38]

def get_angle(pf, p0=np.array([0, 0]), pi=None):
    """Compute angle (in degrees) for pf-p0-pi corner

    Parameters
    ----------
    pf: sequence
        The coordinates of a point on the rotated vector
    p0: sequence
        The coordinates of the pivot
    pi: sequence
        The coordinates of the fixed vector

    Returns
    -------
    float
        The angle in degrees
    """
    if pi is None:
        pi = p0 + np.array([0, 1])
    v0 = np.array(pf) - np.array(p0)
    v1 = np.array(pi) - np.array(p0)

    angle = np.math.atan2(np.linalg.det([v0, v1]), np.dot(v0, v1))
    angle = np.degrees(angle)

    return angle

def psf_tilts(order):
    """
    Get the psf tilts for the given order

    Parameters
    ----------
    order: int
        The order to use, [1, 2]

    Returns
    -------
    np.ndarray
        The angle from the vertical of the psf in each of the 2048 columns
    """
    if order not in [1, 2]:
        raise ValueError('Only orders 1 and 2 are supported.')

    # Get the file
    path = 'files/SOSS_PSF_tilt_order{}.npy'.format(order)
    psf_file = resource_filename('awesimsoss', path)

    if not os.path.exists(psf_file):
        calculate_psf_tilts()

    return np.load(psf_file)

def calculate_psf_tilts():
    """
    Calculate the tilt of the psf at the center of each column
    using all binned pixels in the given wavelength calibration file
    for both orders and save to file
    """
    for order in [1, 2]:

        # Get the file
        path = 'files/SOSS_PSF_tilt_order{}.npy'.format(order)
        psf_file = resource_filename('awesimsoss', path)

        # Dimensions
        subarray = 'SUBSTRIP256'
        X = range(2048)
        Y = range(SUBARRAY_Y.get(subarray))

        # Get the wave map
        wave_map = wave_solutions(subarray, order).astype(float)

        # Get the y-coordinate of the trace polynomial in this column
        # (center of the trace)
        coeffs = trace_polynomials(subarray=subarray, order=order)
        trace = np.polyval(coeffs, X)

        # Interpolate to get the wavelength value at the center
        wave = interp2d(X, Y, wave_map)

        # Get the wavelength of the trace center in each column
        trace_wave = []
        for x, y in zip(X, trace):
            trace_wave.append(wave(x, y)[0])

        # For each column wavelength (defined by the wavelength at
        # the trace center) define an isowavelength contour
        angles = []
        for n, x in enumerate(X):

            w = trace_wave[x]

            # Edge cases
            try:
                w0 = trace_wave[x-1]
            except IndexError:
                w0 = 0

            try:
                w1 = trace_wave[x+1]
            except IndexError:
                w1 = 10

            # Define the width of the wavelength bin as half-way
            # between neighboring points
            dw0 = np.mean([w0, w])
            dw1 = np.mean([w1, w])

            # Get the coordinates of all the pixels in that range
            yy, xx = np.where(np.logical_and(wave_map >= dw0, wave_map < dw1))

            # Find the angle between the vertical and the tilted wavelength bin
            if len(xx) >=1:
                angle = get_angle([xx[-1], yy[-1]], [x, trace[x]])
            else:
                angle = 0

            # Don't flip them upside down
            angle = angle % 180

            # Add to the array
            angles.append(angle)

        # Save the file
        np.save(psf_file, np.array(angles))
        print('Angles saved to', psf_file)

def put_psf_on_subarray(psf, y, frame_height=256):
    """Make a 2D SOSS trace from a sequence of psfs and trace center locations

    Parameters
    ----------
    psf: sequence
        The 2D psf
    y: float
        The grid y value to place the center of the psf
    grid: sequence
        The [x, y] grid ranges

    Returns
    -------
    np.ndarray
        The 2D frame with the interpolated psf
    """
    # Create spline generator
    dim = psf.shape[0]
    mid = (dim - 1.0) / 2.0
    l = np.arange(dim, dtype=np.float)
    spline = RectBivariateSpline(l, l, psf.T, kx=3, ky=3, s=0)

    # Create output frame, shifted as necessary
    yg, xg = np.indices((frame_height, dim), dtype=np.float64)
    yg += mid-y

    # Resample onto the subarray
    frame = spline.ev(xg, yg)

    # Fill resampled points with zeros
    extrapol = (((xg < -0.5) | (xg >= dim - 0.5)) | ((yg < -0.5) | (yg >= dim - 0.5)))
    frame[extrapol] = 0

    return frame

def generate_SOSS_ldcs(wavelengths, ld_profile, grid_point, model_grid='', subarray='SUBSTRIP256', n_bins=100, plot=False, save=''):
    """
    Generate a lookup table of limb darkening coefficients for full
    SOSS wavelength range

    Parameters
    ----------
    wavelengths: sequence
        The wavelengths at which to calculate the LDCs
    ld_profile: str
        A limb darkening profile name supported by
        `ExoCTK.ldc.ldcfit.ld_profile()`
    grid_point: dict, sequence
        The stellar parameters [Teff, logg, FeH] or stellar model
        dictionary from `ExoCTK.modelgrid.ModelGrid.get()`
    n_bins: int
        The number of bins to break up the grism into
    save: str
        The path to save to file to

    Example
    -------
    from awesimsoss.sim2D import awesim
    lookup = awesim.soss_ldc('quadratic', [3300, 4.5, 0])
    """
    try:
        from exoctk import modelgrid
        from exoctk.limb_darkening import limb_darkening_fit as lf
    except ImportError:
        return
        
    # Get the model grid
    if not isinstance(model_grid, modelgrid.ModelGrid):
        model_grid = modelgrid.ModelGrid(os.environ['MODELGRID_DIR'], resolution=700)

    # Load the model grid
    model_grid = modelgrid.ModelGrid(os.environ['MODELGRID_DIR'], resolution=700, 
                                wave_rng=(0.6, 2.8))

    # Get the grid point
    if isinstance(grid_point, (list, tuple, np.ndarray)):
        grid_point = model_grid.get(*grid_point)

    # Abort if no stellar dict
    if not isinstance(grid_point, dict):
        print('Please provide the grid_point argument as [Teff, logg, FeH] or ExoCTK.modelgrid.ModelGrid.get(Teff, logg, FeH).')
        return

    # Break the bandpass up into n_bins pieces
    bandpass = svo.Filter('NIRISS.GR700XD', n_bins=n_bins, verbose=False)

    # Calculate the LDCs
    ldc_results = lf.ldc(None, None, None, model_grid, [ld_profile], 
                         bandpass=bandpass, grid_point=grid_point.copy(), 
                         mu_min=0.08, verbose=False)

    # Interpolate the LDCs to the desired wavelengths
    coeff_table = ldc_results[ld_profile]['coeffs']
    coeff_cols = [c for c in coeff_table.colnames if c.startswith('c')]
    coeffs = [np.interp(wavelengths, coeff_table['wavelength'], coeff_table[c]) for c in coeff_cols]

    # # Compare
    # if plot:
    #     plt.figure()
    #     plt.scatter(coeff_table['c1'], coeff_table['c2'], c=coeff_table['wavelength'], marker='x')
    #     plt.scatter(coeffs[0], coeffs[1], c=wavelengths, marker='o')

    return np.array(coeffs).T

def generate_SOSS_psfs(filt):
    """
    Gnerate a cube of the psf at 100 wavelengths from the min to the max wavelength

    Parameters
    ----------
    filt: str
        The filter to use, ['CLEAR', 'F277W']
    """
    # Get the file
    file = resource_filename('awesimsoss', 'files/SOSS_{}_PSF.fits'.format(filt))

    # Get the NIRISS class from webbpsf and set the filter
    ns = webbpsf.NIRISS()
    ns.filter = filt
    ns.pupil_mask = 'GR700XD'

    # Get the min and max wavelengths
    wavelengths = wave_solutions(256).flatten()
    wave_min = np.max([ns.SHORT_WAVELENGTH_MIN*1E6, np.min(wavelengths[wavelengths>0])])
    wave_max = np.min([ns.LONG_WAVELENGTH_MAX*1E6, np.max(wavelengths[wavelengths>0])])

    # webbpsf.calc_datacube can only handle 100 but that's sufficient
    W = np.linspace(wave_min, wave_max, 100)*1E-6

    # Calculate the psfs
    print("Generating SOSS psfs. This takes about 8 minutes...")
    start = time.time()
    PSF = ns.calc_datacube(W, oversample=1)[0].data
    print("Finished in", time.time()-start)

    # Make the HDUList
    psfhdu = fits.PrimaryHDU(data=PSF)
    wavhdu = fits.ImageHDU(data=W*1E6, name='WAV')
    hdulist = fits.HDUList([psfhdu, wavhdu])

    # Write the file
    hdulist.writeto(file, overwrite=True)
    hdulist.close()

def generate_all_SOSS_psf_cubes():
    """Convenience function to generate all the psf cubes"""
    for filt in ['CLEAR', 'F277W']:
        # generate_SOSS_psfs(filt)
        SOSS_psf_cube(filt=filt, generate=True)

def SOSS_psf_cube(filt='CLEAR', order=1, generate=False):
    """
    Generate/retrieve a data cube of shape (3, 2048, 76, 76)

    Parameters
    ----------
    filt: str
        The filter to use, ['CLEAR', 'F277W']
    order: int
        The trace order
    generate: bool
        Generate a new cube

    Returns
    -------
    np.ndarray
        An array of the SOSS psf at 2048 wavelengths for each order
    """
    if generate:

        print('Coffee time! This takes about 5 minutes.')

        # Get the wavelengths
        wavelengths = np.mean(wave_solutions(256), axis=1)[:2 if filt == 'CLEAR' else 1]
        coeffs = trace_polynomials('SUBSTRIP256')

        # Get the file
        psf_path = 'files/SOSS_{}_PSF.fits'.format(filt)
        psf_file = resource_filename('awesimsoss', psf_path)

        # Load the SOSS psf cube
        cube = fits.getdata(psf_file).swapaxes(-1, -2)
        wave = fits.getdata(psf_file, ext=1)

        # Initilize interpolator
        psfs = interp1d(wave, cube, axis=0, kind=3)
        trace_cols = np.arange(2048)

        # Run datacube
        for n, wavelength in enumerate(wavelengths):

            # Evaluate the trace polynomial in each column to get the y-position
            # of the trace center
            trace_centers = np.polyval(coeffs[n], trace_cols)

            # Don't calculate order2 for F277W or order 3 for either
            if (n == 1 and filt.lower() == 'f277w') or n == 2:
                pass

            else:

                # Get the psf for each column
                print('Calculating order {} SOSS psfs for {} filter...'.format(n+1, filt))
                start = time.time()
                pool = multiprocessing.Pool(8)
                func = partial(get_SOSS_psf, filt=filt, psfs=psfs)
                raw_psfs = np.array(pool.map(func, wavelength))
                pool.close()
                pool.join()
                print('Finished in {} seconds.'.format(time.time()-start))

                # Get the PSF tilt at each column
                # angles = psf_tilts(order)

                # Rotate the psfs
                # print('Rotating order {} SOSS psfs for {} filter...'.format(n+1, filt))
                # start = time.time()
                # pool = multiprocessing.Pool(8)
                # func = partial(rotate, reshape=False)
                # rotated_psfs = np.array(pool.starmap(func, zip(raw_psfs, angles)))
                # pool.close()
                # pool.join()
                # print('Finished in {} seconds.'.format(time.time()-start))
                rotated_psfs = copy.copy(raw_psfs)

                # Scale psfs to 1
                rotated_psfs = np.abs(rotated_psfs)
                scale = np.nansum(rotated_psfs, axis=(1, 2))[:, None, None]
                rotated_psfs = rotated_psfs/scale

                # Split it into 4 chunks to be below Github file size limit
                chunks = rotated_psfs.reshape(4, 512, 76, 76)
                for N, chunk in enumerate(chunks):

                    idx0 = N*512
                    idx1 = idx0+512
                    centers = trace_centers[idx0:idx1]

                    # Interpolate the psfs onto the subarray
                    print('Interpolating chunk {}/4 for order {} SOSS psfs for {} filter onto subarray...'.format(N+1, n+1, filt))
                    start = time.time()
                    pool = multiprocessing.Pool(8)
                    data = zip(chunk, centers)
                    subarray_psfs = pool.starmap(put_psf_on_subarray, data)
                    pool.close()
                    pool.join()
                    print('Finished in {} seconds.'.format(time.time()-start))

                    # Get the filepath
                    filename = 'files/SOSS_{}_PSF_order{}_{}.npy'.format(filt, n+1, N+1)
                    file = resource_filename('awesimsoss', filename)

                    # Delete the file if it exists
                    if os.path.isfile(file):
                        os.system('rm {}'.format(file))

                    # Write the data
                    np.save(file, np.array(subarray_psfs))

                    print('Data saved to', file)

    else:

        # Get the chunked data and concatenate
        full_data = []
        for chunk in [1, 2, 3, 4]:
            path = 'files/SOSS_{}_PSF_order{}_{}.npy'.format(filt, order, chunk)
            file = resource_filename('awesimsoss', path)
            full_data.append(np.load(file))

        return np.concatenate(full_data, axis=0)

def get_SOSS_psf(wavelength, filt='CLEAR', psfs=None, cutoff=0.005):
    """
    Retrieve the SOSS psf for the given wavelength

    Parameters
    ----------
    wavelength: float
        The wavelength to retrieve [um]
    filt: str
        The filter to use, ['CLEAR', 'F277W']
    psfs: numpy.interp1d object (optional)
        The interpolator

    Returns
    -------
    np.ndarray
        The 2D psf for the input wavelength
    """
    if psfs is None:

        # Get the file
        file = resource_filename('awesimsoss', 'files/SOSS_{}_PSF.fits'.format(filt))

        # Load the SOSS psf cube
        cube = fits.getdata(file).swapaxes(-1, -2)
        wave = fits.getdata(file, ext=1)

        # Initilize interpolator
        psfs = interp1d(wave, cube, axis=0, kind=3)

    # Check the wavelength
    if wavelength < psfs.x[0]:
        wavelength = psfs.x[0]

    if wavelength > psfs.x[-1]:
        wavelength = psfs.x[-1]

    # Interpolate and scale psf
    psf = psfs(wavelength)
    psf *= 1./np.nansum(psf)

    # Remove background
    psf[psf < cutoff] = 0

    return psf

def psf_lightcurve(wavelength, psf, response, ld_coeffs, rp, time, tmodel, plot=False):
    """
    Generate a lightcurve for a given wavelength

    Parameters
    ----------
    wavelength: float
        The wavelength value in microns
    psf: sequencs
        The flux-scaled psf for the given wavelength
    response: float
        The spectral response of the detector at the given wavelength
    ld_coeffs: sequence
        The limb darkening coefficients to use
    rp: float
        The planet radius
    time: sequence
        The time axis for the TSO
    tmodel: batman.transitmodel.TransitModel
        The transit model of the planet
    plot: bool
        Plot the lightcurve

    Returns
    -------
    sequence
        A 1D array of the lightcurve with the same length as *t*

    Example 1
    ---------
    # No planet
    from awesimsoss.sim2D import awesim
    psf = np.ones((76, 76))
    time = np.linspace(-0.2, 0.2, 200)
    lc = awesim.psf_lightcurve(0.97, psf, 1, None, None, time, None, plot=True)

    Example 2
    ---------
    # With a planet
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
    params.limb_dark = 'quadratic'                # limb darkening profile to use
    params.u = [1, 1]                              # limb darkening coefficients
    tmodel = batman.TransitModel(params, time)
    lc = awesim.psf_lightcurve(0.97, psf, 1, [0.1, 0.1], 0.05, time, tmodel, plot=True)
    """
    # Expand to shape of time axis
    flux = np.tile(psf, (len(time), 1, 1))

    # If there is a transiting planet...
    if ld_coeffs is not None and rp is not None and str(type(tmodel)) == "<class 'batman.transitmodel.TransitModel'>":

        # Set the wavelength dependent orbital parameters
        tmodel.u = ld_coeffs
        tmodel.rp = rp

        # Generate the light curve for this pixel
        lightcurve = tmodel.light_curve(tmodel)

        # Scale the flux with the lightcurve
        flux *= lightcurve[:, None, None]

    # Apply the filter response to convert to [ADU/s]
    flux *= response

    return flux

def wave_solutions(subarr=None, order=None, directory=None):
    """
    Get the wavelength maps for SOSS orders 1, 2, and 3
    This will be obsolete once the apply_wcs step of the JWST pipeline
    is in place.

    Parameters
     ==  ==  ==  ==  ==
    subarr: str
        The subarray to return, ['SUBSTRIP96', 'SUBSTRIP256', or 'full']
    order: int (optional)
        The trace order, [1, 2, 3]
    directory: str
        The directory containing the wavelength FITS files

    Returns
     ==  ==  == =
    np.ndarray
        An array of the wavelength solutions for orders 1, 2, and 3
    """
    # Get the directory
    if directory is None:
        default = '/files/soss_wavelengths_fullframe.fits'
        directory = resource_filename('awesimsoss', default)

    # Trim to the correct subarray
    if subarr == 'SUBSTRIP256' or subarr == 256:
        idx = slice(0, 256)
    elif subarr == 'SUBSTRIP96' or subarr == 96:
        idx = slice(160, 256)
    else:
        idx = slice(0, 2048)

    # Select the right order
    if order in [1, 2]:
        order = int(order)-1
    else:
        order = slice(0, 3)

    wave = fits.getdata(directory).swapaxes(-2, -1)[order, idx, ::-1]

    return wave

def get_frame_times(subarray, ngrps, nints, t0, nresets=1):
    """
    Calculate a time axis for the exposure in the given SOSS subarray

    Parameters
    ----------
    subarray: str
        The subarray name, i.e. 'SUBSTRIP256', 'SUBSTRIP96', or 'FULL'
    ngrps: int
        The number of groups per integration
    nints: int
        The number of integrations for the exposure
    t0: float
        The start time of the exposure
    nresets: int
        The number of reset frames per integration

    Returns
    -------
    sequence
        The time of each frame
    """
    # Check the subarray
    if subarray not in ['SUBSTRIP256', 'SUBSTRIP96', 'FULL']:
        subarray = 'SUBSTRIP256'
        print("I do not understand subarray '{}'. Using 'SUBSTRIP256' instead.".format(subarray))

    # Get the appropriate frame time
    ft = FRAME_TIMES[subarray]

    # Generate the time axis, removing reset frames
    time_axis = []
    t = t0
    for _ in range(nints):
        times = t+np.arange(nresets+ngrps)*ft
        t = times[-1]+ft
        time_axis.append(times[nresets:])

    time_axis = np.concatenate(time_axis)

    return time_axis

def trace_polynomials(subarray='SUBSTRIP256', order=None, poly_order=4, generate=False):
    """
    Determine the polynomial coefficients of the SOSS traces from the IDT's values

    Parameters
    ----------
    subarray: str
        The name of the subarray
    order: int (optional)
        The trace order, [1, 2]
    poly_order: int
        The order polynomail to fit
    generate: bool
        Generate new coefficients

    Returns
    -------
    sequence
        The list of polynomial coefficients for orders 1  and 2
    """
    if generate:

        # Get the data
        file = resource_filename('awesimsoss', 'files/soss_wavelength_trace_table1.txt')
        x1, y1, w1, x2, y2, w2 = np.genfromtxt(file, unpack=True)

        # File says to do this but doesn't seem necessary on trace simulation
        # # Subarray 96
        # if subarray == 'SUBSTRIP96':
        #     y1 -= 10
        #     y2 -= 10

        # Fit the polynomails
        fit1 = np.polyfit(x1, y1, poly_order)
        fit2 = np.polyfit(x2, y2, poly_order)

        return fit1, fit2

    else:

        # Select the right order
        if order in [1, 2]:
            order = int(order)-1
        else:
            order = slice(0, 3)

        # if subarray == 'SUBSTRIP96':
        #     coeffs = [[1.71164994e-11, -4.72119272e-08, 5.10276801e-05, -5.91535309e-02, 7.30680347e+01], [2.35792131e-13, 2.42999478e-08, 1.03641247e-05, -3.63088657e-02, 8.96766537e+01]]
        # else:
        #     coeffs = [[1.71164994e-11, -4.72119272e-08, 5.10276801e-05, -5.91535309e-02, 8.30680347e+01], [2.35792131e-13, 2.42999478e-08, 1.03641247e-05, -3.63088657e-02, 9.96766537e+01]]

        coeffs = [[1.71164994e-11, -4.72119272e-08, 5.10276801e-05, -5.91535309e-02, 8.30680347e+01], [2.35792131e-13, 2.42999478e-08, 1.03641247e-05, -3.63088657e-02, 9.96766537e+01]]

        return coeffs[order]