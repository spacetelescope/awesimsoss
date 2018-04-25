"""
A module to generate simulated 2D time-series SOSS data

Authors: Joe Filippazzo, Kevin Volk, Jonathan Fraine, Michael Wolfe
"""

import os
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import batman
import astropy.units as q
import astropy.constants as ac
# import multiprocessing
import time
import AWESim_SOSS
import inspect
import warnings
import datetime
import webbpsf
import pkg_resources
from svo_filters import svo
from multiprocessing.dummy import Pool as ThreadPool 
from . import generate_darks as gd
from ExoCTK import core
from ExoCTK.ldc import ldcfit as lf
from astropy.io import fits
from scipy.optimize import curve_fit
from scipy.ndimage.interpolation import zoom
from scipy.interpolate import interp1d, _fitpack, splrep, splev
from functools import partial
from sklearn.externals import joblib
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from scipy.ndimage import map_coordinates
from numpy.core.multiarray import interp as compiled_interp
from skimage.transform import PiecewiseAffineTransform, warp, warp_coords, estimate_transform

warnings.simplefilter('ignore')

cm = plt.cm
FRAME_TIMES = {'SUBSTRIP96':2.213, 'SUBSTRIP256':5.491, 'FULL':10.737}
SUBARRAY_Y = {'SUBSTRIP96':96, 'SUBSTRIP256':256, 'FULL':2048}

def wave_map_tform(order, **kwargs):
    """
    Apply the inverse transform to the wave_map to get the vertical placement of the trace and the average wavelength of each column
    """
    # Get the order coeffs
    coeffs = trace_polynomials(subarray='SUBSTRIP256', order=4, generate=False)[order-1]
    
    # Get the transform
    tform = transform_from_polynomial(coeffs, **kwargs)
    
    # Transform the wave_map
    wave_map = wave_solutions(256)[order-1].astype(float)
    
    # Warp it and plot!
    linear_wave_map = warp(wave_map, inverse_map=tform)
    
    # Test that the std in each column has decreased
    print('Input wave map average std:',np.mean(np.std(wave_map, axis=(0))))
    print('Output wave map average std:',np.mean(np.std(linear_wave_map, axis=(0))))

def transform_from_polynomial(coeffs, cols=100, rows=5, delta_row=38, test_point=197, plot=False):
    """
    Calculate the Affine transform that takes into account the curvature of the trace
    
    Parameters
    ----------
    coeffs: sequence
        The polynomial coefficients
    cols: int
        The number of columns of control points
    rows: int
        The number of rows of control points
    delta_row: float
        The distance between rows of control points (default=38 since the webbpsf frame is (76,76))
    test_point: int
        Plot a test point to show which scr and dst points correspond to each other
    plot: bool
        Plot the control points
    
    Returns
    -------
    sequence
        The (x,y) coordinates of the control points
    """
    # Get the polynomial and evaluate
    X = np.linspace(0, 2048, cols)
    Y = np.polyval(coeffs, X)
    idx = np.arange(-1, rows-1)
    
    # Interpolate the data with a spline
    spl = splrep(X, Y)
    
    if plot:
        plt.figure(figsize=(20,8))
        plt.plot(X, Y, c='r')
        
    # Calculate all dst and src points
    src_points = []
    dst_points = []
    for N0,x0 in enumerate(X):
        
        # Evaluate f(x0) and f'(x0) and the slope of the normal line
        f = splev(x0, spl, der=0)
        fprime = splev(x0, spl, der=1)
        m = -1./fprime
        
        # Calculate the normal line to the dst points
        x = np.arange(x0-delta_row, x0+delta_row)
        normal = m*(x-x0)+f
        y0 = normal[delta_row]
        
        # Generate (x,y) of each control point row
        for N1,i in enumerate(idx):
            
            # Calculate the dst point
            # xp = x0 + (-1 if x0>1460 else 1)*i*delta_row*np.sqrt(1/(1+m**2))
            xp = x0 + (m/abs(m))*i*delta_row*np.sqrt(1/(1+m**2))
            yp = m*(xp-x0) + y0
            dst_points.append((xp,yp))
            
            # Calculate the src point
            yl = np.min(Y)+i*delta_row
            src_points.append((x0,yl))
            
            # Plot both points
            if plot:
                plt.plot(x, normal, '--g', lw=2)
                plt.plot(xp, yp, 'sg')
                plt.plot(x0, yl, 'ob')
                
                # Plot a test point for visual inspection
                if N0*len(idx)+N1==test_point:
                    plt.plot(xp, yp, 'sr', markersize=10)
                    plt.plot(x0, yl, 'or', markersize=10)
                    
        # Plot the normal line to the src points
        if plot:
            plt.axvline(x0, c='b')
            
    # Put into an array
    dst_points = np.array(dst_points)
    src_points = np.array(src_points)
    
    if plot:
        plt.xlim(0,2048)
        plt.ylim(0,256)
    
    # Perform transform
    tform = PiecewiseAffineTransform()
    tform.estimate(src_points, dst_points)
    
    return tform

def make_linear_SOSS_trace(psfs, offset=27, subarray='SUBSTRIP256', plot=False):
    """
    Construct the trace from the 2D psf generated by webbpsf at each wavelength
    
    Parameters
    ----------
    psfs: np.ndarray
        The data cube of shape (n_wavelengths, n_frames, x, y)
    offset: int
        The y-offset for placement of the 76x76 psf on the linear trace
    subarray: str
        The subarray to use
    plot: bool
        Plot the trace
    
    Returns
    -------
    np.ndarray
        The 3D linear trace
    """
    # Get dimensions
    n_frames, n_waves, x, y = psfs.shape
    
    # Get the psf center
    c = int(x/2)
    
    # Get the subarray height
    Y = SUBARRAY_Y.get(subarray)
    
    # Empty trace (with padding for overflow)
    linear_trace = np.zeros((n_frames, n_waves+x, Y+y))
    
    # For each frame in the exposure...
    for N,frame in enumerate(psfs):
        
        # ... place psf in each column (i.e. wavelength)
        for n,wave in enumerate(frame):
            
            # Place the trace center in the correct column
            linear_trace[N,n:n+x,offset:y+offset] += wave
            
    # Transpose and invert x-axis  DMS orientation
    linear_trace = linear_trace.swapaxes(1,2)[:,:,::-1]
   
    # Plot it
    if plot:
        plt.figure(figsize=(13,2))
        plt.imshow(linear_trace[0], origin='lower', norm=matplotlib.colors.LogNorm())
        plt.xlim(0,2048)
        plt.ylim(0,Y)
        
    return linear_trace

def ADUtoFlux(order):
    """
    Return the wavelength dependent conversion from ADUs to erg s-1 cm-2 
    in SOSS traces 1, 2, and 3
    
    Parameters
    ==========
    order: int
        The trace order, must be 1, 2, or 3
    
    Returns
    =======
    np.ndarray
        Arrays to convert the given order trace from ADUs to units of flux
    """
    ADU2mJy, mJy2erg = 7.586031e-05, 2.680489e-15
    scaling = np.genfromtxt(pkg_resources.resource_filename('AWESim_SOSS', 'files/GR700XD_{}.txt'.format(order)), unpack=True)
    scaling[1] *= ADU2mJy*mJy2erg
    
    return scaling

def generate_SOSS_ldcs(wavelengths, ld_profile, grid_point, model_grid='', subarray='SUBSTRIP256', n_bins=100, plot=False, save=''):
    """
    Generate a lookup table of limb darkening coefficients for full SOSS wavelength range
    
    Parameters
    ----------
    wavelengths: sequence
        The wavelengths at which to calculate the LDCs
    ld_profile: str
        A limb darkening profile name supported by `ExoCTK.ldc.ldcfit.ld_profile()`
    grid_point: dict, sequence
        The stellar parameters [Teff, logg, FeH] or stellar model dictionary from `ExoCTK.core.ModelGrid.get()`
    n_bins: int
        The number of bins to break up the grism into
    save: str
        The path to save to file to
    
    Example
    -------
    from AWESim_SOSS.sim2D import awesim
    lookup = awesim.soss_ldc('quadratic', [3300, 4.5, 0])
    """
    # Get the model grid
    if not isinstance(model_grid, core.ModelGrid):
        model_grid = core.ModelGrid(os.environ['MODELGRID_DIR'], resolution=700)
    
    # Load the model grid
    model_grid = core.ModelGrid(os.environ['MODELGRID_DIR'], resolution=700, wave_rng=(0.6,2.8))
    
    # Get the grid point
    if isinstance(grid_point, (list,tuple,np.ndarray)):
        grid_point = model_grid.get(*grid_point)
        
    # Abort if no stellar dict
    if not isinstance(grid_point, dict):
        print('Please provide the grid_point argument as [Teff, logg, FeH] or ExoCTK.core.ModelGrid.get(Teff, logg, FeH).')
        return
        
    # Break the bandpass up into n_bins pieces
    bandpass = svo.Filter('NIRISS.GR700XD', n_bins=n_bins, verbose=False)
    
    # Calculate the LDCs
    ldc_results = lf.ldc(None, None, None, model_grid, [ld_profile], bandpass=bandpass, grid_point=grid_point.copy(), mu_min=0.08, verbose=False)
    
    # Interpolate the LDCs to the desired wavelengths
    coeff_table = ldc_results[ld_profile]['coeffs']
    coeff_cols = [c for c in coeff_table.colnames if c.startswith('c')]
    coeffs = [np.interp(wavelengths, coeff_table['wavelength'], coeff_table[c]) for c in coeff_cols]
    
    # Compare
    if plot:
        plt.figure()
        plt.scatter(coeff_table['c1'], coeff_table['c2'], c=coeff_table['wavelength'], marker='x')
        plt.scatter(coeffs[0], coeffs[1], c=wavelengths, marker='o')
        
    return np.array(coeffs).T

def generate_SOSS_psfs(filt):
    """
    Gnerate a cube of the psf at 100 wavelengths from the min to the max wavelength
    
    Parameters
    ----------
    filt: str
        The filter to use, ['CLEAR','F277W']
    """
    # Get the file
    file = pkg_resources.resource_filename('AWESim_SOSS', 'files/SOSS_{}_PSF.fits'.format(filt))
    
    # Get the NIRISS class from webbpsf and set the filter
    ns = webbpsf.NIRISS()
    ns.filter = filt
    ns.pupil_mask = 'GR700XD'
    
    # Get the min and max wavelengths
    wavelengths = wave_solutions(256).flatten()
    wave_min = np.max([ns.SHORT_WAVELENGTH_MIN*1E6,np.min(wavelengths[wavelengths>0])])
    wave_max = np.min([ns.LONG_WAVELENGTH_MAX*1E6,np.max(wavelengths[wavelengths>0])])
    
    # webbpsf.calc_datacube can only handle 100 but that's sufficient
    W = np.linspace(wave_min, wave_max, 100)*1E-6
    
    # Calculate the psfs
    print("Generating SOSS psfs. This takes about 8 minutes...")
    start = time.time()
    PSF = ns.calc_datacube(W, oversample=1)[0].data
    print("Finished in",time.time()-start)
    
    # Make the HDUList
    psfhdu = fits.PrimaryHDU(data=PSF)
    wavhdu = fits.ImageHDU(data=W*1E6, name='WAV')
    hdulist = fits.HDUList([psfhdu, wavhdu])
    
    # Write the file
    hdulist.writeto(file, overwrite=True)
    hdulist.close()
    
def SOSS_psf_cube(filt='CLEAR', order=1, generate=False):
    """
    Generate/retrieve a data cube of shape (3, 2048, 76, 76) 
    
    Parameters
    ----------
    filt: str
        The filter to use, ['CLEAR','F277W']
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
        
        # Get the wavelengths
        wavelengths = np.mean(wave_solutions(256), axis=1)
        
        # Get the file
        psf_file = pkg_resources.resource_filename('AWESim_SOSS', 'files/SOSS_{}_PSF.fits'.format(filt))
        
        # Load the SOSS psf cube
        cube = fits.getdata(psf_file).swapaxes(-1,-2)
        wave = fits.getdata(psf_file, ext=1)
        
        # Initilize interpolator
        psfs = interp1d(wave, cube, axis=0, kind=3)
        
        # Run datacube
        for n,order in enumerate(wavelengths):
            
            # Get the file
            file = pkg_resources.resource_filename('AWESim_SOSS', 'files/SOSS_{}_PSF_order{}.fits'.format(filt,n+1))
            
            print('Calculating order {} SOSS psfs for {} filter...'.format(n+1,filt))
            start = time.time()
            
            pool = multiprocessing.Pool(8)
            
            # Set wavelength independent inputs of lightcurve function
            func = partial(get_SOSS_psf, filt=filt, psfs=psfs)
            
            # Generate the psfs at each wavelength
            all_psfs = pool.map(func, order)
            
            # Close the pool
            pool.close()
            pool.join()
            
            print('Finished in {} seconds.'.format(time.time()-start))
            
            # Write to the file
            fits.HDUList([fits.PrimaryHDU(data=np.array(all_psfs))]).writeto(file, overwrite=True)
            print('Data saved to',file)
        
    else:
        
        # Get the file
        file = pkg_resources.resource_filename('AWESim_SOSS', 'files/SOSS_{}_PSF_order{}.fits'.format(filt,order))
        
        return fits.getdata(file)

def get_SOSS_psf(wavelength, filt='CLEAR', psfs=''):
    """
    Retrieve the SOSS psf for the given wavelength
    
    Parameters
    ----------
    wavelength: float
        The wavelength to retrieve [um]
    filt: str
        The filter to use, ['CLEAR','F277W']
    psfs: numpy.interp1d object (optional)
        The interpolator
    
    Returns
    -------
    np.ndarray
        The 2D psf for the input wavelength
    """
    if psfs=='':
        
        # Get the file
        file = pkg_resources.resource_filename('AWESim_SOSS', 'files/SOSS_{}_PSF.fits'.format(filt))
        
        # Load the SOSS psf cube
        cube = fits.getdata(file).swapaxes(-1,-2)
        wave = fits.getdata(file, ext=1)
        
        # Initilize interpolator
        psfs = interp1d(wave, cube, axis=0, kind=3)
        
    # Check the wavelength
    if wavelength<psfs.x[0]:
        wavelength = psfs.x[0]

    if wavelength>psfs.x[-1]:
        wavelength = psfs.x[-1]
        
    # Interpolate and scale psf
    psf = psfs(wavelength)
    psf *= 1./np.nansum(psf)
        
    return psf

def psf_lightcurve(wavelength, psf, response, ld_coeffs, rp, planet, time, tmodel, params, plot=False):
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
    planet: sequence
        The wavelength and Rp/R* of the planet at t=0 
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
    from AWESim_SOSS.sim2D import awesim
    import astropy.units as q, os, AWESim_SOSS
    DIR_PATH = os.path.dirname(os.path.realpath(AWESim_SOSS.__file__))
    vega = np.genfromtxt(DIR_PATH+'/files/scaled_spectrum.txt', unpack=True) # A0V with Jmag=9
    w = vega[0]*q.um
    f = (vega[1]*q.W/q.m**2/q.um).to(q.erg/q.s/q.cm**2/q.AA)
    psf = np.ones((76,76))
    time = np.linspace(-0.2, 0.2, 200)
    lc = awesim.psf_lightcurve(0.97, psf, 1, 1,'', [w,f], '', time, '', '', plot=True)
    
    Example 2
    ---------
    # With a planet
    planet1D = np.genfromtxt(DIR_PATH+'/files/WASP107b_pandexo_input_spectrum.dat', unpack=True)
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
    params.u = [1,1]                              # limb darkening coefficients
    tmodel = batman.TransitModel(params, time)
    lc = awesim.psf_lightcurve(0.97, psf, 1, [1,1], 0.05, [w,f], planet1D, time, tmodel, params, plot=True)
    """
    # Expand to shape of time axis
    flux = np.tile(psf, (len(time),1,1))
    
    # If there is a transiting planet...
    if not isinstance(planet, str):
        
        # Set the wavelength dependent orbital parameters
        params.u = ld_coeffs
        params.rp = rp
        
        # Generate the light curve for this pixel
        lightcurve = tmodel.light_curve(params)
        
        # Scale the flux with the lightcurve
        flux *= lightcurve[:, None, None]
        
    # Apply the filter response to convert to [ADU/s]
    flux *= response
    
    # Plot
    if plot:
        plt.plot(time, np.nanmean(flux, axis=(1,2)))
        plt.xlabel("Time from central transit")
        plt.ylabel("Flux Density [photons/s/cm2/A]")
        
    return flux

def wave_solutions(subarr, directory=pkg_resources.resource_filename('AWESim_SOSS', '/files/soss_wavelengths_fullframe.fits')):
    """
    Get the wavelength maps for SOSS orders 1, 2, and 3
    This will be obsolete once the apply_wcs step of the JWST pipeline
    is in place.
     
    Parameters
    ==========
    subarr: str
        The subarray to return, accepts '96', '256', or 'full'
    directory: str
        The directory containing the wavelength FITS files
        
    Returns
    =======
    np.ndarray
        An array of the wavelength solutions for orders 1, 2, and 3
    """
    try:
        idx = int(subarr)
    except:
        idx = None
    
    wave = fits.getdata(directory).swapaxes(-2,-1)[:,:idx]
    
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
    if subarray not in ['SUBSTRIP256','SUBSTRIP96','FULL']:
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

# def get_frame_times(subarray, ngrps, nints, t0, nresets=1):
#     """
#     Calculate a time axis for the exposure in the given SOSS subarray
#
#     Parameters
#     ----------
#     subarray: str
#         The subarray name, i.e. 'SUBSTRIP256', 'SUBSTRIP96', or 'FULL'
#     ngrps: int
#         The number of groups per integration
#     nints: int
#         The number of integrations for the exposure
#     t0: float
#         The start time of the exposure
#     nresets: int
#         The number of reset frames per integration
#
#     Returns
#     -------
#     sequence
#         The time of each frame
#     """
#     # Check the subarray
#     if subarray not in ['SUBSTRIP256','SUBSTRIP96','FULL']:
#         subarray = 'SUBSTRIP256'
#         print("I do not understand subarray '{}'. Using 'SUBSTRIP256' instead.".format(subarray))
#
#     # Get the appropriate frame time
#     ft = FRAME_TIMES[subarray]
#
#     # Generate the time axis, removing reset frames
#     time_axis = []
#     t = t0
#     for _ in range(nints):
#
#         # Generate a time for each group in the integration
#         times = t+np.arange(nresets+ngrps)*ft
#         time_axis.append(times[nresets:])
#
#         # Update the start time of the next integration
#         t = times[-1]+ft
#
#     # Flatten the times of each integration
#     time_axis = np.concatenate(time_axis)
#
#     # Convert time axis from seconds to days
#     # since the period is input as days
#     time_axis /= 86400
#
#     return time_axis

def trace_polynomials(subarray='SUBSTRIP256', order=4, generate=False):
    """
    Determine the polynomials of the SOSS traces from the IDT's values
    
    Parameters
    ----------
    subarray: str
        The name of the subarray
    order: int
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
        file = pkg_resources.resource_filename('AWESim_SOSS', 'files/soss_wavelength_trace_table1.txt')
        x1, y1,w1, x2, y2, w2 = np.genfromtxt(file, unpack=True)
        
        # Subarray 96
        if subarray=='SUBSTRIP96':
            y1 -= 10
            y2 -= 10
            
        # Fit the polynomails
        fit1 = np.polyfit(x1, y1, order)
        fit2 = np.polyfit(x2, y2, order)
        
        # Plot the results
        plt.figure(figsize=(13,2))
        plt.plot(x1, y1, c='b', marker='o', ls='none', label='Order 1')
        plt.plot(x2, y2, c='b', marker='o', ls='none', label='Order 2')
        plt.plot(x1, np.polyval(fit1, x1), c='r', label='Order 1 Fit')
        plt.plot(x2, np.polyval(fit2, x2), c='r', label='Order 2 Fit')
        plt.xlim(0,2048)
        if subarray=='SUBSTRIP96':
            plt.ylim(0,96)
        else:
            plt.ylim(0,256)
        plt.legend(loc=0)
        
        return fit1, fit2
        
    else:
        
        if subarray=='SUBSTRIP96':
            coeffs = [[1.71164994e-11, -4.72119272e-08, 5.10276801e-05, -5.91535309e-02, 7.30680347e+01], [2.35792131e-13, 2.42999478e-08, 1.03641247e-05, -3.63088657e-02, 8.96766537e+01]]
        else:
            coeffs = [[1.71164994e-11, -4.72119272e-08, 5.10276801e-05, -5.91535309e-02, 8.30680347e+01], [2.35792131e-13, 2.42999478e-08, 1.03641247e-05, -3.63088657e-02, 9.96766537e+01]]

        return coeffs

def warp_frames(frames, tform, multiprocess=False):
    """
    Warp the given frames using the given transform
    
    Parameters
    ----------
    frames: sequence
        The 3D data cube
    tform: skimage.transform._geometric.PiecewiseAffineTransform
        The image transform object
    multiprocess: bool
        Use multiprocessing
    
    Returns
    -------
    np.ndarray
        The warped frames
    """
    if multiprocess:
        
        # Init the pool
        pool = multiprocessing.Pool(8)

        # Load the warp function with the precomputed transform
        func = partial(warp, inverse_map=tform.inverse)

        # Generate the warped trace at each frame
        warped = pool.map(func, frames)

        # Close the pool
        pool.close()
        pool.join()

    else:

        # Manually warp the frames
        warped = []
        for f in frames:
            warped.append(warp(f, inverse_map=tform.inverse))

    # Put into an array
    warped = np.asarray(warped)

    return warped

class TSO(object):
    """
    Generate NIRISS SOSS time series observations
    """
    def __init__(self, ngrps, nints, star, snr=700, subarray='SUBSTRIP256', t0=0, target=None):
        """
        Iterate through all pixels and generate a light curve if it is inside the trace
        
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
        from AWESim_SOSS.sim2D import awesim
        import astropy.units as q, astropy.constants as ac, os, AWESim_SOSS, batman
        DIR_PATH = os.path.dirname(os.path.realpath(AWESim_SOSS.__file__))
        star = np.genfromtxt(DIR_PATH+'/files/scaled_spectrum.txt', unpack=True)
        star1D = [star[0]*q.um, (star[1]*q.W/q.m**2/q.um).to(q.erg/q.s/q.cm**2/q.AA)]
        
        # Initialize simulation
        tso = awesim.TSO(ngrps=5, nints=2, star=star1D)
        """
        # Set instance attributes for the exposure
        self.subarray = subarray
        self.nrows = SUBARRAY_Y[subarray]
        self.ncols = 2048
        self.ngrps = ngrps
        self.nints = nints
        self.nresets = 1
        self.frame_time = FRAME_TIMES[subarray]
        self.time = get_frame_times(subarray, ngrps, nints, t0, self.nresets)
        self.nframes = len(self.time)
        self.target = target or 'Simulated Target'
        self.obs_date = '2016-01-04'
        self.obs_time = '23:37:52.226'
        self.filter = 'CLEAR'
        self.header = ''
        self.snr = snr
        
        # Set instance attributes for the target
        self.star = star
        self.wave = wave_solutions(str(self.nrows))
        self.avg_wave = np.mean(self.wave, axis=1)
        self.ld_coeffs = np.zeros((3, 2048, 2))
        self.planet = ''
        self.tmodel = ''
        self.params = ''
        
        # Get the cube of SOSS psfs
        order1_flux = np.interp(self.avg_wave[0], self.star[0], self.star[1], left=0, right=0)[:, np.newaxis, np.newaxis]
        order2_flux = np.interp(self.avg_wave[1], self.star[0], self.star[1], left=0, right=0)[:, np.newaxis, np.newaxis]
        self.CLEAR_order1 = SOSS_psf_cube(filt='CLEAR', order=1)*order1_flux
        self.CLEAR_order2 = SOSS_psf_cube(filt='CLEAR', order=1)*order1_flux
        self.F277W_order1 = SOSS_psf_cube(filt='F277W', order=2)*order2_flux
        self.F277W_order2 = SOSS_psf_cube(filt='F277W', order=2)*order2_flux
        
        # Get absolute calibration reference file
        calfile = pkg_resources.resource_filename('AWESim_SOSS', 'files/jwst_niriss_photom_0028.fits')
        caldata = fits.getdata(calfile)
        self.photom = caldata[caldata['pupil']=='GR700XD']
            
        # Create the empty exposure
        self.dims = (self.nframes, self.nrows, self.ncols)
        self.tso = np.zeros(self.dims)
        self.tso_ideal = np.zeros(self.dims)
        self.tso_order1_ideal = np.zeros(self.dims)
        self.tso_order2_ideal = np.zeros(self.dims)
    
    def run_simulation(self, filt='CLEAR', orders=[1,2], planet=None, tmodel=None, params=None, ld_coeffs=None, ld_profile='quadratic', model_grid=None, verbose=True):
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
        ld_profile: str (optional)
            The limb darkening profile to use
        ld_coeffs: array-like (optional)
            A 3D array that assigns limb darkening coefficients to each pixel, i.e. wavelength
        model_grid: ExoCTK.core.ModelGrid (optional)
            The model atmosphere grid to calculate LDCs
        verbose: bool
            Print helpful stuff
        
        Example
        -------
        # Run simulation of star only
        tso.run_simulation()
        
        # Simulate star with transiting exoplanet by including transmission spectrum and orbital params
        planet1D = np.genfromtxt(DIR_PATH+'/files/WASP107b_pandexo_input_spectrum.dat', unpack=True)
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
        params.u = [1,1]                              # limb darkening coefficients
        tmodel = batman.TransitModel(params, tso.time)
        tso.run_simulation(planet=planet1D, tmodel=tmodel, params=params)
        """
        if verbose:
            begin = time.time()
        
        # Clear previous results
        self.tso = np.zeros(self.dims)
        self.tso_ideal = np.zeros(self.dims)
        self.tso_order1_ideal = np.zeros(self.dims)
        self.tso_order2_ideal = np.zeros(self.dims)
        
        # Set single order to list
        if isinstance(orders,int):
            orders = [orders]
        if not all([o in [1,2] for o in orders]):
            raise TypeError('Order must be either an int, float, or list thereof; i.e. [1,2]')
        orders = list(set(orders))
        
        # Check if it's F277W to speed up calculation
        self.filter = filt.upper()
        if self.filter=='F277W':
            orders = [1]
                
        # If there is a planet transmission spectrum but no LDCs, generate them
        if isinstance(planet, np.ndarray):
            
            # Check if the stellar params are the same
            old_params = [getattr(self.params, p, None) for p in ['teff','logg','feh']]
            
            # Store planet details
            self.planet = planet
            self.params = params
            self.tmodel = tmodel
            self.params.limb_dark = ld_profile
            self.params.t0 = self.time[self.nframes//2]
            
            # Use input ld coeffs
            if isinstance(ld_coeffs[0], float):
                self.ld_coeffs = [np.transpose([[ld_coeffs[0], ld_coeffs[1]]] * self.avg_wave[order-1].size) for order in orders]
                
            # Use input ld coeff array
            elif isinstance(ld_coeffs, np.ndarray):
                self.ld_coeffs = ld_coeffs
                
            # Or generate them if the stellar paramaeters have changed
            else:
                stellar_params = [getattr(params, p) for p in ['teff','logg','feh']]
                if stellar_params!=old_params:
                    self.ld_coeffs = [generate_SOSS_ldcs(self.avg_wave[order-1], self.params.limb_dark, stellar_params, model_grid=model_grid) for order in orders]
                
        # Generate simulation for each order
        for order in orders:
            
            # Get the wavelength map
            wave = self.avg_wave[order-1]
            
            # Get the psf cube
            cube = getattr(self, '{}_order{}'.format(self.filter, order))
            
            # Get limb darkening coeffs and make into a list
            ld_coeffs = self.ld_coeffs[order-1]
            ld_coeffs = list(map(list, ld_coeffs))
            
            # Set the radius at the given wavelength from the transmission spectrum (Rp/R*)**2
            if planet!=None:
                tdepth = np.interp(wave, planet[0], planet[1])
            else:
                tdepth = np.ones_like(wave)
            rp = np.sqrt(tdepth)
            
            # Get relative spectral response for the order (from /grp/crds/jwst/references/jwst/jwst_niriss_photom_0028.fits)
            throughput = self.photom[(self.photom['order']==order)&(self.photom['filter']==self.filter)]
            ph_wave = throughput.wavelength[throughput.wavelength>0][1:-2]
            ph_resp = throughput.relresponse[throughput.wavelength>0][1:-2]
            response = np.interp(wave, ph_wave, ph_resp)
            
            # Convert response in [mJy/ADU/s] to [Flam/ADU/s] then invert so that we can convert the flux at each wavelegth into [ADU/s]
            response = 1./(response*q.mJy*ac.c/(wave*q.um)**2).to(self.star[1].unit).value
            setattr(self, 'photom_order{}'.format(order), response)
            
            # Caluclate the transform for the desired polynomial
            # tform = transform_from_polynomial(self.nrows+76, self.ncols+76, coeffs=trace_polynomials(self.subarray, generate=False)[order-1])
            trace_coeffs = trace_polynomials(self.subarray)[order-1]
            tform = transform_from_polynomial(trace_coeffs)
            
            # Run multiprocessing to generate lightcurves
            if verbose:
                print('Calculating order {} light curves...'.format(order))
                start = time.time()
            pool = ThreadPool(8) 
            
            # Set wavelength independent inputs of lightcurve function
            func = partial(psf_lightcurve, planet=self.planet, time=self.time, tmodel=self.tmodel, params=self.params)
            
            # Generate the lightcurves at each wavelength
            psfs = np.asarray(pool.starmap(func, list(zip(wave, cube, response, ld_coeffs, rp))))
            psfs = psfs.swapaxes(0,1)
            psfs = psfs.swapaxes(2,3)
            
            # Close the pool
            pool.close()
            pool.join()
            
            # Multiply by the frame time to convert to [ADU]
            ft = np.tile(self.time[:self.ngrps], self.nints)
            psfs *= ft[:,None,None,None]
            
            # Generate TSO frames with linear traces
            if verbose:
                print('Lightcurves finished:',time.time()-start)
                # print('Constructing order {} linear traces...'.format(order))
                # start = time.time()
            frames = make_linear_SOSS_trace(psfs, subarray=self.subarray)
            
            # Run multiprocessing to construct trace
            if verbose:
                # print('Order {} linear traces finished:'.format(order),time.time()-start)
                print('Constructing order {} warped traces...'.format(order))
                start = time.time()
            
            
            # Warp the frames
            coords = warp_coords(tform.inverse, (self.nrows,self.ncols))
            all_frames = []
            for f in frames:
                all_frames.append(map_coordinates(f, coords))
            tso_order = np.array(all_frames)
            
            # Test warp a single frame
            # warped_frame = warp(frames[0], inverse_map=tform.inverse, output_shape=(self.nrows,self.ncols))
            # plt.figure()
            # plt.imshow(tso_order[0]-warped_frame)

            if verbose:
                print('Order {} warped traces finished:'.format(order), time.time()-start)
                
            # Add it to the individual order
            setattr(self, 'tso_order{}_ideal'.format(order), tso_order)
            
        # Add to the master TSO
        self.tso = np.sum([getattr(self, 'tso_order{}_ideal'.format(order)) for order in orders], axis=0)
        
        # Add noise to the observations using Kevin Volk's dark ramp simulator
        self.tso_ideal = self.tso.copy()
        
        # Add noise and ramps
        # self.add_noise()
        
        if verbose:
            print('\nTotal time:',time.time()-begin)
    
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
        orders = np.asarray([self.tso_order1_ideal,self.tso_order2_ideal])
        
        # Load all the reference files
        photon_yield = fits.getdata(pkg_resources.resource_filename('AWESim_SOSS', 'files/photon_yield_dms.fits'))
        pca0_file = pkg_resources.resource_filename('AWESim_SOSS', 'files/niriss_pca0.fits')
        zodi = fits.getdata(pkg_resources.resource_filename('AWESim_SOSS', 'files/soss_zodiacal_background_scaled.fits'))
        nonlinearity = fits.getdata(pkg_resources.resource_filename('AWESim_SOSS', 'files/substrip256_forward_coefficients_dms.fits'))
        pedestal = fits.getdata(pkg_resources.resource_filename('AWESim_SOSS', 'files/substrip256pedestaldms.fits'))
        darksignal = fits.getdata(pkg_resources.resource_filename('AWESim_SOSS', 'files/substrip256signaldms.fits'))*self.gain
        
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
        
    def plot_frame(self, frame='', scale='linear', order='', noise=True, cmap=cm.jet):
        """
        Plot a TSO frame
        
        Parameters
        ----------
        frame: int
            The frame number to plot
        scale: str
            Plot in linear or log scale
        order: int (optional)
            The order to isolate
        noise: bool
            Plot with the noise model
        cmap: str
            The color map to use
        """
        if order:
            tso = getattr(self, 'tso_order{}_ideal'.format(order))
        else:
            if noise:
                tso = self.tso
            else:
                tso = self.tso_ideal
                
        # Get data for plotting
        vmax = int(np.nanmax(tso[tso<np.inf]))
        frame = np.array(tso[frame or self.nframes//2].data)
        
        # Draw plot
        plt.figure(figsize=(13,2))
        if scale=='log':
            frame[frame<1.] = 1.
            plt.imshow(frame, origin='lower', interpolation='none', norm=matplotlib.colors.LogNorm(), vmin=1, vmax=vmax, cmap=cmap)
        else:
            plt.imshow(frame, origin='lower', interpolation='none', vmin=1, vmax=vmax, cmap=cmap)
        plt.colorbar()
        plt.title('Injected Spectrum')
    
    def plot_snr(self, frame='', cmap=cm.jet):
        """
        Plot the SNR of a TSO frame
        
        Parameters
        ----------
        frame: int
            The frame number to plot
        cmap: matplotlib.cm.colormap
            The color map to use
        """
        # Get the SNR
        snr  = np.sqrt(self.tso[frame or self.nframes//2].data)
        vmax = int(np.nanmax(snr))
        
        # Plot it
        plt.figure(figsize=(13,2))
        plt.imshow(snr, origin='lower', interpolation='none', vmin=1, vmax=vmax, cmap=cmap)
        plt.colorbar()
        plt.title('SNR over Spectrum')
        
    def plot_saturation(self, frame='', saturation=80.0, cmap=cm.jet):
        """
        Plot the saturation of a TSO frame
        
        Parameters
        ----------
        frame: int
            The frame number to plot
        saturation: float
            Percentage of full well that defines saturation
        cmap: matplotlib.cm.colormap
            The color map to use
        """
        # The full well of the detector pixels
        fullWell = 65536.0
        
        # Get saturated pixels
        saturated = np.array(self.tso[frame or self.nframes//2].data) > (saturation/100.0) * fullWell
        
        # Plot it
        plt.figure(figsize=(13,2))
        plt.imshow(saturated, origin='lower', interpolation='none', cmap=cmap)
        plt.colorbar()
        plt.title('{} Saturated Pixels'.format(len(saturated[saturated>fullWell])))
    
    def plot_slice(self, col, trace='tso', frame=0, order='', **kwargs):
        """
        Plot a column of a frame to see the PSF in the cross dispersion direction
        
        Parameters
        ----------
        col: int, sequence
            The column index(es) to plot a light curve for
        trace: str
            The attribute name to plot
        frame: int
            The frame number to plot
        """
        if order:
            tso = getattr(self, 'tso_order{}_ideal'.format(order))
        else:
            tso = self.tso
            
        f = tso[frame].T
        
        if isinstance(col, int):
            col = [col]
            
        for c in col:
            plt.plot(f[c], label='Column {}'.format(c), **kwargs)
            
        plt.xlim(0,256)
        
        plt.legend(loc=0, frameon=False)
        
    def plot_ramp(self):
        """
        Plot the total flux on each frame to display the ramp
        """
        plt.figure()
        plt.plot(np.sum(self.tso, axis=(1,2)), ls='none', marker='o')
        plt.xlabel('Group')
        plt.ylabel('Count Rate [ADU/s]')
        plt.grid()
        
    def plot_lightcurve(self, col):
        """
        Plot a lightcurve for each column index given
        
        Parameters
        ----------
        col: int, float, sequence
            The integer column index(es) or float wavelength(s) in microns 
            to plot as a light curve
        """
        # Get the scaled flux in each column for the last group in each integration
        f = np.nansum(self.tso_ideal[self.ngrps::self.ngrps], axis=1)
        f = f/np.nanmax(f, axis=1)[:,None]
        
        # Make it into an array
        if isinstance(col, (int,float)):
            col = [col]
            
        for c in col:
            
            # If it is an index
            if isinstance(c, int):
                lc = f[:,c]
                label = 'Col {}'.format(c)
                
            # Or assumed to be a wavelength in microns
            elif isinstance(c, float):
                W = np.mean(self.wave[0], axis=0)
                lc = [np.interp(c, W, F) for F in f]
                label = '{} um'.format(c)
                
            else:
                print('Please enter an index, astropy quantity, or array thereof.')
                return
            
            plt.plot(self.time[self.ngrps::self.ngrps], lc, label=label, marker='.', ls='None')
            
        plt.legend(loc=0, frameon=False)
        
    def plot_spectrum(self, frame=0, order=''):
        """
        Parameters
        ----------
        frame: int
            The frame number to plot
        """
        if order:
            tso = getattr(self, 'tso_order{}_ideal'.format(order))
        else:
            tso = self.tso
        
        # Get extracted spectrum (Column sum for now)
        wave = np.mean(self.wave[0], axis=0)
        flux = np.sum(tso[frame].data, axis=0)
        response = 1./self.photom_order1
        
        # Convert response in [mJy/ADU/s] to [Flam/ADU/s] then invert so that we can convert the flux at each wavelegth into [ADU/s]
        flux *= response*self.time[np.mod(self.ngrps, frame)]
        
        # Plot it along with input spectrum
        plt.figure(figsize=(13,5))
        plt.loglog(wave, flux, label='Extracted')
        plt.loglog(*self.star, label='Injected')
        plt.xlim(wave[0]*0.95,wave[-1]*1.05)
        plt.legend()
    
    def save(self, filename='dummy.save'):
        """
        Save the TSO data to file
        
        Parameters
        ----------
        filename: str
            The path of the save file
        """
        print('Saving TSO class dict to {}'.format(filename))
        joblib.dump(self.__dict__, filename)
    
    def load(self, filename):
        """
        Load a previously calculated TSO
        
        Paramaters
        ----------
        filename: str
            The path of the save file
        
        Returns
        -------
        awesim.TSO()
            A TSO class dict
        """
        print('Loading TSO class dict to {}'.format(filename))
        load_dict = joblib.load(filename)
        # for p in [i for i in dir(load_dict)]:
        #     setattr(self, p, getattr(params, p))
        for key in load_dict.keys():
            exec("self." + key + " = load_dict['" + key + "']")
    
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
                ('TFRAME', FRAME_TIMES[self.subarray], 'Time in seconds between frames'),
                ('TGROUP', FRAME_TIMES[self.subarray], 'Delta time between groups (s)'),
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
        data = np.swapaxes(self.tso, 1, 2)[:,:,::-1]
        
        # Make the HDUList
        prihdu = fits.PrimaryHDU(data=data, header=prihdr)
        
        # Write the file
        prihdu.writeto(outfile, overwrite=True)
        
        print('File saved as',outfile)
