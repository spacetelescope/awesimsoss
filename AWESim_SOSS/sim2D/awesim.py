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
import multiprocessing
import time
import AWESim_SOSS
import inspect
import warnings
import datetime
import webbpsf
from . import generate_darks as gd
from ExoCTK import svo
from ExoCTK import core
from ExoCTK.ldc import ldcfit as lf
from astropy.io import fits
from scipy.optimize import curve_fit
from functools import partial
from sklearn.externals import joblib
from numpy.core.multiarray import interp as compiled_interp

warnings.simplefilter('ignore')

cm = plt.cm
FILTERS = svo.filters()
DIR_PATH = os.path.dirname(os.path.realpath(AWESim_SOSS.__file__))
FRAME_TIMES = {'SUBSTRIP96':2.213, 'SUBSTRIP256':5.491, 'FULL':10.737}

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
    scaling = np.genfromtxt(DIR_PATH+'/files/GR700XD_{}.txt'.format(order), unpack=True)
    scaling[1] *= ADU2mJy*mJy2erg
    
    return scaling

def norm_to_mag(spectrum, magnitude, bandpass):
    """
    Returns the flux of a given *spectrum* [W,F] normalized to the given *magnitude* in the specified photometric *band*
    """
    # Get the current magnitude and convert to flux
    mag, mag_unc = get_mag(spectrum, bandpass, fetch='flux')
    
    # Convert input magnitude to flux
    flx, flx_unc = mag2flux(bandpass.filterID.split('/')[1], magnitude, sig_m='', units=spectrum[1].unit)
    
    # Normalize the spectrum
    spectrum[1] *= np.trapz(bandpass.rsr[1], x=bandpass.rsr[0])*np.sqrt(2)*flx/mag
    
    return spectrum

def flux2mag(bandpass, f, sig_f='', photon=False):
    """
    For given band and flux returns the magnitude value (and uncertainty if *sig_f*)
    """
    eff = bandpass.WavelengthEff
    zp = bandpass.ZeroPoint
    unit = q.erg/q.s/q.cm**2/q.AA
    
    # Convert to f_lambda if necessary
    if f.unit == 'Jy':
        f,  = (ac.c*f/eff**2).to(unit)
        sig_f = (ac.c*sig_f/eff**2).to(unit)
    
    # Convert energy units to photon counts
    if photon:
        f = (f*(eff/(ac.h*ac.c)).to(1/q.erg)).to(unit/q.erg), 
        sig_f = (sig_f*(eff/(ac.h*ac.c)).to(1/q.erg)).to(unit/q.erg)
    
    # Calculate magnitude
    m = -2.5*np.log10((f/zp).value)
    sig_m = (2.5/np.log(10))*(sig_f/f).value if sig_f else ''
    
    return [m, sig_m]

def mag2flux(band, mag, sig_m='', units=q.erg/q.s/q.cm**2/q.AA):
    """
    Caluclate the flux for a given magnitude
    
    Parameters
    ----------
    band: str, svo.Filter
        The bandpass
    mag: float, astropy.unit.quantity.Quantity
        The magnitude
    sig_m: float, astropy.unit.quantity.Quantity
        The magnitude uncertainty
    units: astropy.unit.quantity.Quantity
        The unit for the output flux
    """
    try:
        # Get the band info
        filt = FILTERS.loc[band]
        
        # Make mag unitless
        if hasattr(mag,'unit'):
            mag = mag.value
        if hasattr(sig_m,'unit'):
            sig_m = sig_m.value
        
        # Calculate the flux density
        zp = q.Quantity(filt['ZeroPoint'], filt['ZeroPointUnit'])
        f = zp*10**(mag/-2.5)
        
        if isinstance(sig_m,str):
            sig_m = np.nan
        
        sig_f = f*sig_m*np.log(10)/2.5
            
        return [f, sig_f]
        
    except IOError:
        return [np.nan, np.nan]

def rebin_spec(spec, wavnew, oversamp=100, plot=False):
    """
    Rebin a spectrum to a new wavelength array while preserving 
    the total flux
    
    Parameters
    ----------
    spec: array-like
        The wavelength and flux to be binned
    wavenew: array-like
        The new wavelength array
        
    Returns
    -------
    np.ndarray
        The rebinned flux
    
    """
    nlam = len(spec[0])
    x0 = np.arange(nlam, dtype=float)
    x0int = np.arange((nlam-1.)*oversamp + 1., dtype=float)/oversamp
    w0int = np.interp(x0int, x0, spec[0])
    spec0int = np.interp(w0int, spec[0], spec[1])/oversamp
    try:
        err0int = np.interp(w0int, spec[0], spec[2])/oversamp
    except:
        err0int = ''
        
    # Set up the bin edges for down-binning
    maxdiffw1 = np.diff(wavnew).max()
    w1bins = np.concatenate(([wavnew[0]-maxdiffw1], .5*(wavnew[1::]+wavnew[0:-1]), [wavnew[-1]+maxdiffw1]))
    
    # Bin down the interpolated spectrum:
    w1bins = np.sort(w1bins)
    nbins = len(w1bins)-1
    specnew = np.zeros(nbins)
    errnew = np.zeros(nbins)
    inds2 = [[w0int.searchsorted(w1bins[ii], side='left'), w0int.searchsorted(w1bins[ii+1], side='left')] for ii in range(nbins)]

    for ii in range(nbins):
        specnew[ii] = np.sum(spec0int[inds2[ii][0]:inds2[ii][1]])
        try:
            errnew[ii] = np.sum(err0int[inds2[ii][0]:inds2[ii][1]])
        except:
            pass
            
    if plot:
        plt.figure()
        plt.loglog(spec[0], spec[1], c='b')    
        plt.loglog(wavnew, specnew, c='r')
        
    return [wavnew,specnew,errnew]

def get_mag(spectrum, bandpass, exclude=[], fetch='mag', photon=False, Flam=False, plot=False):
    """
    Returns the integrated flux of the given spectrum in the given band
    
    Parameters
    ---------
    spectrum: array-like
        The [w,f,e] of the spectrum with astropy.units
    bandpass: str, svo_filters.svo.Filter
        The bandpass to calculate
    exclude: sequecne
        The wavelength ranges to exclude by linear interpolation between gap edges
    photon: bool
        Use units of photons rather than energy
    Flam: bool
        Use flux units rather than the default flux density units
    plot: bool
        Plot it
    
    Returns
    -------
    list
        The integrated flux of the spectrum in the given band
    """
    # Get the Filter object if necessary
    if isinstance(bandpass, str):
        bandpass = svo.Filter(bandpass)
        
    # Get filter data in order
    unit = q.Unit(bandpass.WavelengthUnit)
    mn = bandpass.WavelengthMin*unit
    mx = bandpass.WavelengthMax*unit
    wav, rsr = bandpass.raw
    wav = wav*unit
    
    # Unit handling
    a = (1 if photon else q.erg)/q.s/q.cm**2/(1 if Flam else q.AA)
    b = (1 if photon else q.erg)/q.s/q.cm**2/q.AA
    c = 1/q.erg
    
    # Test if the bandpass has full spectral coverage
    if np.logical_and(mx < np.max(spectrum[0]), mn > np.min(spectrum[0])) \
    and all([np.logical_or(all([i<mn for i in rng]), all([i>mx for i in rng])) for rng in exclude]):
        
        # Rebin spectrum to bandpass wavelengths
        w, f, sig_f = rebin_spec([i.value for i in spectrum], wav.value)*spectrum[1].unit
        
        # Calculate the integrated flux, subtracting the filter shape
        F = (np.trapz((f*rsr*((wav/(ac.h*ac.c)).to(c) if photon else 1)).to(b), x=wav)/(np.trapz(rsr, x=wav))).to(a)
        
        # Caluclate the uncertainty
        if sig_f:
            sig_F = np.sqrt(np.sum(((sig_f*rsr*np.gradient(wav).value*((wav/(ac.h*ac.c)).to(c) if photon else 1))**2).to(a**2)))
        else:
            sig_F = ''
            
        # Make a plot
        if plot:
            plt.figure()
            plt.step(spectrum[0], spectrum[1], color='k', label='Spectrum')
            plt.errorbar(bandpass.WavelengthEff, F.value, yerr=sig_F.value, marker='o', label='Magnitude')
            try:
                plt.fill_between(spectrum[0], spectrum[1]+spectrum[2], spectrum[1]+spectrum[2], color='k', alpha=0.1)
            except:
                pass
            plt.plot(bandpass.rsr[0], bandpass.rsr[1]*F, label='Bandpass')
            plt.xlabel(unit)
            plt.ylabel(a)
            plt.legend(loc=0, frameon=False)
            
        # Get magnitude from flux
        m, sig_m = flux2mag(bandpass, F, sig_f=sig_F)
        
        return [m, sig_m, F, sig_F] if fetch=='both' else [F, sig_F] if fetch=='flux' else [m, sig_m]
        
    else:
        return ['']*4 if fetch=='both' else ['']*2

def ldc_lookup(ld_profile, grid_point, model_grid, delta_w=0.005, save=''):
    """
    Generate a lookup table of limb darkening coefficients for full SOSS wavelength range
    
    Parameters
    ----------
    ld_profile: str
        A limb darkening profile name supported by `ExoCTK.ldc.ldcfit.ld_profile()`
    grid_point: dict
        The stellar model dictionary from `ExoCTK.core.ModelGrid.get()`
    model_grid: ExoCTK.core.ModelGrid
        The model grid
    delta_w: float
        The width of the wavelength bins in microns
    save: str
        The path to save to file to
    
    Example
    -------
    import os
    from AWESim_SOSS.sim2D import awesim
    from ExoCTK import core
    grid = core.ModelGrid(os.environ['MODELGRID_DIR'], Teff_rng=(3000,4000), logg_rng=(4,5), FeH_rng=(0,0.5), resolution=700)
    model = G.get(3300, 4.5, 0)
    awesim.ldc_lookup('quadratic', model, grid, save='/Users/jfilippazzo/Desktop/')
    """
    print("Go get a coffee! This takes about 5 minutes to run.")
    
    # Initialize the lookup table
    lookup = {}
    
    # Get the full wavelength range
    wave_maps = wave_solutions(256)
    
    # Define function for multiprocessing
    def gr700xd_ldc(wavelength, delta_w, ld_profile, grid_point, model_grid):
        """
        Calculate the LCDs for the given wavelength range in the GR700XD grism
        """
        try:
            # Get the bandpass in that wavelength range
            mn = (wavelength-delta_w/2.)*q.um
            mx = (wavelength+delta_w/2.)*q.um
            throughput = np.genfromtxt(DIR_PATH+'/files/NIRISS.GR700XD.1.txt', unpack=True)
            bandpass = svo.Filter('GR700XD', throughput, n_bins=1, wl_min=mn, wl_max=mx, verbose=False)
            
            # Calculate the LDCs
            ldcs = lf.ldc(None, None, None, model_grid, [ld_profile], bandpass=bandpass, grid_point=grid_point.copy(), mu_min=0.08, verbose=False)
            coeffs = list(zip(*ldcs[ld_profile]['coeffs']))[1::2]
            coeffs = [coeffs[0][0],coeffs[1][0]]
            
            return ('{:.9f}'.format(wavelength), coeffs)
            
        except:
            
            print(wavelength)
            
            return ('_', None)
            
    # Pool the LDC calculations across the whole wavelength range for each order
    for order in [1,2,3]:
        
        # Get the wavelength limits for this order
        min_wave = np.nanmin(wave_maps[order-1][wave_maps[order-1]>0])
        max_wave = np.nanmax(wave_maps[order-1][wave_maps[order-1]>0])
        
        # Generate list of binned wavelengths
        wavelengths = np.arange(min_wave, max_wave, delta_w)
        
        # Turn off printing
        print('Calculating order {} LDCs at {} wavelengths...'.format(order,len(wavelengths)))
        sys.stdout = open(os.devnull, 'w')
        
        # Pool the LDC calculations across the whole wavelength range
        processes = 8
        start = time.time()
        pool = multiprocessing.pool.ThreadPool(processes)
        
        func = partial(gr700xd_ldc, 
                       delta_w    = delta_w,
                       ld_profile = ld_profile,
                       grid_point = grid_point,
                       model_grid = model_grid)
                       
        # Turn list of coeffs into a dictionary
        order_dict = dict(pool.map(func, wavelengths))
        
        pool.close()
        pool.join()
        
        # Add the dict to the master
        try:
            order_dict.pop('_')
        except:
            pass
        lookup['order{}'.format(order)] = order_dict
        
        # Turn printing back on
        sys.stdout = sys.__stdout__
        print('Order {} LDCs finished: '.format(order), time.time()-start)
        
    if save:
        t, g, m = grid_point['Teff'], grid_point['logg'], grid_point['FeH']
        joblib.dump(lookup, save+'/{}_ldc_lookup_{}_{}_{}.save'.format(ld_profile,t,g,m))
        
    else:
    
        return lookup

def ld_coefficient_map(lookup_file, subarray='SUBSTRIP256', save=True):
    """
    Generate  map of limb darkening coefficients at every NIRISS pixel for all SOSS orders
    
    Parameters
    ----------
    lookup_file: str
        The path to the lookup table of LDCs
    
    Example
    -------
    ld_coeffs_lookup = ld_coefficient_lookup(1, 'quadratic', star, model_grid)
    """
    # Get the lookup table
    ld_profile = os.path.basename(lookup_file).split('_')[0]
    lookup = joblib.load(lookup_file)
    
    # Get the wavelength map
    nrows = 256 if subarray=='SUBSTRIP256' else 96 if subarray=='SUBSTRIP96' else 2048
    wave_map = wave_solutions(nrows)
        
    # Make dummy array for LDC map results
    ldfunc = lf.ld_profile(ld_profile)
    ncoeffs = len(inspect.signature(ldfunc).parameters)-1
    ld_coeffs = np.zeros((3, nrows*2048, ncoeffs))
    
    # Calculate the coefficients at each pixel for each order
    for order,wavelengths in enumerate(wave_map[:1]):
        
        # Get a flat list of all wavelengths for this order
        wave_list = wavelengths.flatten()
        lkup = lookup['order{}'.format(order+1)]
        
        # Get the bin size
        delta_w = np.mean(np.diff(sorted(np.array(list(map(float,lkup))))))/2.
        
        # For each bin in the lookup table...
        for bin, coeffs in lkup.items():
            
            try:
                
                # Get all the pixels that fall within the bin
                w = float(bin)
                idx, = np.where(np.logical_and(wave_list>=w-delta_w,wave_list<=w+delta_w))
                
                # Place them in the coefficient map
                ld_coeffs[order][idx] = coeffs
                
            except:
                 
                print(bin)
                
    if save:
        path = lookup_file.replace('lookup','map')
        joblib.dump(ld_coeffs, path)
        
        print('LDC coefficient map saved at',path)
        
    else:
        
        return ld_coeffs

def trace_polynomial(trace, start=4, end=2040, order=4):
    # Make a scatter plot where the pixels in each column are offset by a small amount
    x, y = [], []
    for n,col in enumerate(trace.T):
        vals = np.where(~col)
        if vals:
            v = list(vals[0])
            y += v
            x += list(np.random.normal(n, 1E-16, size=len(v)))
            
    # Now fit a polynomial to it!
    height, length = trace.shape
    coeffs = np.polyfit(x[start:], y[start:], order)
    X = np.arange(start, length, 1)
    Y = np.polyval(coeffs, X)
    
    return X, Y

def distance_map(order, generate=False, start=4, end=2044, p_order=4, plot=False):
    """
    Generate a map where each pixel is the distance from the trace polynomial
    
    Parameters
    ----------
    plot: bool
        Plot the distance map
    
    Returns
    -------
    np.ndarray
        An array the same shape as masked_data
    
    """   
    # If missing, generate it
    if generate:
        
        print('Generating distance map...')
        
        mask = joblib.load(DIR_PATH+'/files/order{}_mask.save'.format(order)).swapaxes(-1,-2)
        
        # Get the trace polynomial
        X, Y = trace_polynomial(mask, start, end, p_order)
        
        # Get the distance from the pixel to the polynomial
        def dist(p0, Poly):
            return min(np.sqrt((p0[0]-Poly[0])**2 + (p0[1]-Poly[1])**2))
            
        # Make a map of pixel locations
        height, length = mask.shape
        d_map = np.zeros(mask.shape)
        for i in range(length):
            for j in range(height):
                d_map[j,i] = dist((j,i), (Y,X))
                
        joblib.dump(d_map, DIR_PATH+'/files/order_{}_distance_map.save'.format(order))
        
    else:
        d_map = joblib.load(DIR_PATH+'/files/order_{}_distance_map.save'.format(order))
        
    
    if plot:
        plt.figure(figsize=(13,2))
        
        plt.title('Order {}'.format(order))
        
        plt.imshow(d_map, interpolation='none', origin='lower', norm=matplotlib.colors.LogNorm())
        
        plt.colorbar()
    
    return d_map

def generate_psf(filt, oversample=4, plot=False):
    """
    Generate the SOSS psf with 'CLEAR' or 'F277W' filter
    
    Parameters
    ----------
    filt: str
        The filter to use, 'CLEAR' or 'F277W'
    oversample:int
        The oversampling factor
    plot: bool
        Plot the 1D and 2D psf for visual inspection
    
    Returns
    -------
    np.ndarray
        The 1D psf
    """
    print("Generating the psf with {} filter and GR700XD pupil mask...".format(filt))
    
    # Get the NIRISS class from webbpsf and set the filter
    ns = webbpsf.NIRISS()
    ns.filter = filt
    ns.pupil_mask = 'GR700XD'
    psf2D = ns.calcPSF(oversample=oversample)[0].data
    psf1D = np.sum(psf2D, axis=0)
    
    if plot:
        plt.figure(figsize=(4,6))
        plt.suptitle('PSF for NIRISS GR700XD and {} filter'.format(filt))
        gs = matplotlib.gridspec.GridSpec(2, 1, height_ratios=[3, 1]) 
        ax1 = plt.subplot(gs[0])
        ax1.imshow(psf2D)
        
        ax2 = plt.subplot(gs[1])
        ax2.plot(psf1D)
        ax2.set_xlim(0,psf2D.shape[0])
        
        plt.tight_layout()
    
    return psf1D

def psf_position(distance, extend=25, generate=False, filt='CLEAR', plot=False):
    """
    Scale the flux based on the pixel's distance from the center of the cross dispersed psf
    """
    # Generate the PSF from webbpsf
    if generate:
        
        # Generate the 1D psf
        psf1D = generate_psf(filt)
        
        # Scale the transmission to 1
        psf = psf1D/np.trapz(psf1D)
        
    # Or just use these (already scaled to 1!)
    else:
        
        if filt=='F277W':
            psf = np.array([ 8.26876242e-05, 8.31425198e-05, 8.36901427e-05, 8.46984936e-05, 8.61497853e-05, 8.76418567e-05, 8.87479230e-05, 8.93855914e-05, 8.99024883e-05, 9.08101878e-05, 9.23767786e-05, 9.44060057e-05, 9.63931141e-05, 9.79550282e-05, 9.92078159e-05, 1.00784317e-04, 1.03444072e-04, 1.07537558e-04, 1.12713072e-04, 1.18094130e-04, 1.22817738e-04, 1.26556080e-04, 1.29648187e-04, 1.32748554e-04, 1.36254697e-04, 1.39943104e-04, 1.43094540e-04, 1.45029921e-04, 1.45670868e-04, 1.45713549e-04, 1.46282594e-04, 1.48308785e-04, 1.52068824e-04, 1.57191477e-04, 1.63073610e-04, 1.69338873e-04, 1.75953704e-04, 1.82909988e-04, 1.89764805e-04, 1.95489595e-04, 1.98874877e-04, 1.99296931e-04, 1.97311087e-04, 1.94576030e-04, 1.93046014e-04, 1.93901904e-04, 1.96932787e-04, 2.00821141e-04, 2.04176786e-04, 2.06627873e-04, 2.09204841e-04, 2.13722011e-04, 2.21557750e-04, 2.32666614e-04, 2.45508699e-04, 2.57947441e-04, 2.68496129e-04, 2.77067760e-04, 2.84764184e-04, 2.92944642e-04, 3.02315696e-04, 3.12709649e-04, 3.23630103e-04, 3.35004390e-04, 3.47407241e-04, 3.61470404e-04, 3.76931148e-04, 3.92198950e-04, 4.05035720e-04, 4.14094020e-04, 4.20265411e-04, 4.26716681e-04, 4.37306322e-04, 4.54265915e-04, 4.76725264e-04, 5.01252123e-04, 5.24233696e-04, 5.44546562e-04, 5.64576977e-04, 5.88665410e-04, 6.19858376e-04, 6.57201679e-04, 6.95659588e-04, 7.29059107e-04, 7.54329509e-04, 7.74248709e-04, 7.96773396e-04, 8.31325274e-04, 8.84543854e-04, 9.58406202e-04, 1.05185203e-03, 1.16432941e-03, 1.29802564e-03, 1.45637958e-03, 1.63942505e-03, 1.83951977e-03, 2.04166583e-03, 2.22997856e-03, 2.39736289e-03, 2.55221964e-03, 2.71672652e-03, 2.91613226e-03, 3.16476282e-03, 3.45780782e-03, 3.77556039e-03, 4.09936155e-03, 4.43038639e-03, 4.79900967e-03, 5.25672307e-03, 5.85261044e-03, 6.60634426e-03, 7.49312823e-03, 8.45018840e-03, 9.40227978e-03, 1.02924364e-02, 1.11009074e-02, 1.18422240e-02, 1.25435050e-02, 1.32182705e-02, 1.38521039e-02, 1.44081651e-02, 1.48472222e-02, 1.51473332e-02, 1.53089822e-02, 1.53420437e-02, 1.52443871e-02, 1.49890088e-02, 1.45320281e-02, 1.38401339e-02, 1.29219973e-02, 1.18433404e-02, 1.07136907e-02, 9.64965386e-03, 8.73424351e-03, 7.99475418e-03, 7.41051124e-03, 6.94342254e-03, 6.57039356e-03, 6.29648707e-03, 6.14152616e-03, 6.11165186e-03, 6.17851740e-03, 6.28437576e-03, 6.37355498e-03, 6.43114087e-03, 6.50150427e-03, 6.66917074e-03, 7.00741380e-03, 7.52190883e-03, 8.12352793e-03, 8.65044862e-03, 8.93247859e-03, 8.86580942e-03, 8.45923964e-03, 7.82792980e-03, 7.13935385e-03, 6.54155787e-03, 6.11144530e-03, 5.84681236e-03, 5.69945201e-03, 5.62404110e-03, 5.61184591e-03, 5.69165688e-03, 5.90312657e-03, 6.26491979e-03, 6.76123501e-03, 7.35559810e-03, 8.02083273e-03, 8.76231341e-03, 9.61574640e-03, 1.06178792e-02, 1.17672667e-02, 1.30001557e-02, 1.41984226e-02, 1.52274384e-02, 1.59838104e-02, 1.64273412e-02, 1.65814890e-02, 1.65055163e-02, 1.62574342e-02, 1.58701429e-02, 1.53525040e-02, 1.47099453e-02, 1.39667850e-02, 1.31724874e-02, 1.23859283e-02, 1.16473208e-02, 1.09564786e-02, 1.02728123e-02, 9.53911811e-03, 8.71640890e-03, 7.81002270e-03, 6.87223179e-03, 5.98021932e-03, 5.20214049e-03, 4.56985444e-03, 4.07185037e-03, 3.66762463e-03, 3.31298907e-03, 2.98125008e-03, 2.66973354e-03, 2.39104624e-03, 2.15729326e-03, 1.96816471e-03, 1.80961108e-03, 1.66223824e-03, 1.51265803e-03, 1.36019023e-03, 1.21530658e-03, 1.09184879e-03, 9.98571738e-04, 9.35070504e-04, 8.93575233e-04, 8.64206578e-04, 8.39634314e-04, 8.16384144e-04, 7.92941930e-04, 7.67078931e-04, 7.34975735e-04, 6.92932236e-04, 6.40239592e-04, 5.80818988e-04, 5.22121748e-04, 4.71775063e-04, 4.34058065e-04, 4.08390061e-04, 3.90618332e-04, 3.76051948e-04, 3.62178151e-04, 3.49431474e-04, 3.39844827e-04, 3.34820518e-04, 3.33677693e-04, 3.33886427e-04, 3.32624863e-04, 3.28425713e-04, 3.21765450e-04, 3.14323291e-04, 3.07596278e-04, 3.01915507e-04, 2.96480261e-04, 2.90207948e-04, 2.82627114e-04, 2.74113579e-04, 2.65369375e-04, 2.56669306e-04, 2.47565303e-04, 2.37338532e-04, 2.25856449e-04, 2.14128246e-04, 2.04041998e-04, 1.97369073e-04, 1.94683483e-04, 1.94944600e-04, 1.96052177e-04, 1.96019305e-04, 1.93996074e-04, 1.90501367e-04, 1.86780675e-04, 1.83798030e-04, 1.81578475e-04, 1.79318575e-04, 1.76113648e-04, 1.71726649e-04, 1.66834543e-04, 1.62598709e-04, 1.59899312e-04, 1.58792218e-04, 1.58551426e-04, 1.58217078e-04, 1.57214721e-04, 1.55597944e-04, 1.53782818e-04, 1.52035993e-04, 1.50151102e-04, 1.47580065e-04, 1.43916134e-04, 1.39340877e-04, 1.34663483e-04, 1.30883707e-04, 1.28570110e-04, 1.27493069e-04, 1.26777313e-04, 1.25470587e-04, 1.23133009e-04, 1.20046189e-04, 1.16920743e-04, 1.14337703e-04, 1.12334968e-04, 1.10424483e-04, 1.08000771e-04, 1.04819773e-04, 1.01191444e-04, 9.77562868e-05, 9.50284779e-05, 9.30529679e-05, 9.14243765e-05, 8.96347420e-05, 8.74708893e-05, 8.51546406e-05, 8.31260569e-05, 8.16497932e-05, 8.05627626e-05, 7.94707536e-05])
        
        else:
            psf = np.array([ 4.67532046e-05, 5.30389911e-05, 5.52111126e-05, 5.82565134e-05, 6.29642065e-05, 6.43427527e-05, 6.35454453e-05, 5.64016028e-05, 6.14386571e-05, 6.63152736e-05, 6.45836693e-05, 5.97018163e-05, 5.64699516e-05, 6.37256778e-05, 6.26547570e-05, 6.76732571e-05, 7.43267641e-05, 7.28510234e-05, 7.31834852e-05, 7.21726907e-05, 8.03215200e-05, 7.86206810e-05, 7.62869586e-05, 8.40947695e-05, 8.61639687e-05, 8.52853108e-05, 7.79900730e-05, 8.30735151e-05, 8.32745819e-05, 7.97100789e-05, 8.88448618e-05, 8.90333696e-05, 9.41118174e-05, 9.61236544e-05, 1.09285261e-04, 1.23403519e-04, 1.10380871e-04, 1.04814592e-04, 1.05592775e-04, 1.16542161e-04, 1.19241758e-04, 1.19166498e-04, 1.31663501e-04, 1.27784966e-04, 1.32110587e-04, 1.28684719e-04, 1.25988343e-04, 1.31598251e-04, 1.32435338e-04, 1.52556496e-04, 1.46832766e-04, 1.33821551e-04, 1.36250894e-04, 1.52623507e-04, 1.73268125e-04, 1.61132129e-04, 1.70577280e-04, 1.80799608e-04, 1.87473794e-04, 2.04902910e-04, 2.05139140e-04, 2.15557529e-04, 2.06723465e-04, 2.23812740e-04, 2.62062327e-04, 2.68732068e-04, 2.71526830e-04, 2.71967259e-04, 3.00966208e-04, 2.92628851e-04, 2.67017847e-04, 2.86082202e-04, 3.08270037e-04, 3.39736009e-04, 3.53375388e-04, 3.98351864e-04, 4.25012421e-04, 4.27704377e-04, 4.90079280e-04, 5.41962518e-04, 5.86557111e-04, 5.70792241e-04, 5.92670678e-04, 6.52366175e-04, 6.55253137e-04, 7.36077424e-04, 7.76164752e-04, 8.45971390e-04, 9.21847045e-04, 9.90503988e-04, 1.21725673e-03, 1.29166082e-03, 1.35374259e-03, 1.47948092e-03, 1.60471722e-03, 1.80520640e-03, 1.84774536e-03, 2.03975179e-03, 2.29605272e-03, 2.46311542e-03, 2.68088119e-03, 2.90375259e-03, 3.38939619e-03, 3.78066148e-03, 4.41756959e-03, 5.42688445e-03, 6.19145203e-03, 7.17143901e-03, 7.92333224e-03, 8.84960288e-03, 1.01850521e-02, 1.07554693e-02, 1.17338385e-02, 1.29769792e-02, 1.33461906e-02, 1.33389623e-02, 1.28507697e-02, 1.27756195e-02, 1.32516301e-02, 1.30323908e-02, 1.27057194e-02, 1.28642856e-02, 1.27860018e-02, 1.23414494e-02, 1.21723933e-02, 1.19170157e-02, 1.14300727e-02, 1.13637833e-02, 1.11633853e-02, 1.05698477e-02, 1.03689500e-02, 1.07786259e-02, 1.11621594e-02, 1.10047997e-02, 1.03911145e-02, 9.32890298e-03, 8.20610056e-03, 7.43046895e-03, 6.93812282e-03, 6.69310715e-03, 6.51997377e-03, 6.43079394e-03, 6.30644641e-03, 6.02216742e-03, 5.73117451e-03, 5.43214367e-03, 5.22635979e-03, 5.12016140e-03, 5.28646690e-03, 5.85364221e-03, 6.60082764e-03, 7.35825944e-03, 7.68991527e-03, 7.44692551e-03, 7.06042168e-03, 7.04991239e-03, 7.31200217e-03, 7.30827250e-03, 7.24418461e-03, 7.35995149e-03, 7.47868564e-03, 7.36104043e-03, 7.04386526e-03, 7.01922914e-03, 7.38266867e-03, 7.84837817e-03, 8.11470906e-03, 8.33445520e-03, 8.83703454e-03, 9.32904378e-03, 9.53477959e-03, 9.69112519e-03, 1.02205491e-02, 1.08214226e-02, 1.13861505e-02, 1.21308377e-02, 1.23174085e-02, 1.22238751e-02, 1.27276635e-02, 1.36998429e-02, 1.48242300e-02, 1.57437872e-02, 1.60741361e-02, 1.57583114e-02, 1.57282181e-02, 1.56693988e-02, 1.48816612e-02, 1.44346344e-02, 1.35730914e-02, 1.20114458e-02, 1.05774446e-02, 8.99085992e-03, 8.35046648e-03, 8.04890115e-03, 6.92229762e-03, 5.84691772e-03, 4.77903623e-03, 4.03576548e-03, 3.56314200e-03, 2.98915463e-03, 2.69490981e-03, 2.33716463e-03, 2.07583212e-03, 1.97813196e-03, 1.81292084e-03, 1.62459617e-03, 1.30855188e-03, 1.22614057e-03, 1.25655137e-03, 1.17536895e-03, 1.08122662e-03, 9.43570327e-04, 8.99601631e-04, 7.98558975e-04, 7.10841052e-04, 6.96810140e-04, 6.28386228e-04, 6.24061730e-04, 5.81937127e-04, 5.31816303e-04, 4.75103378e-04, 4.12041552e-04, 4.50269014e-04, 4.53033181e-04, 4.23416652e-04, 3.66636302e-04, 3.29994765e-04, 3.34665840e-04, 2.97625016e-04, 2.97668861e-04, 3.03587057e-04, 3.02718386e-04, 2.86075102e-04, 2.59891697e-04, 2.70125175e-04, 2.42026260e-04, 2.22208835e-04, 2.15300750e-04, 2.08372142e-04, 2.07092087e-04, 1.79671119e-04, 1.86602377e-04, 1.90800938e-04, 1.86188859e-04, 1.86937030e-04, 1.72504728e-04, 1.65337390e-04, 1.50629008e-04, 1.59829604e-04, 1.60656418e-04, 1.38049604e-04, 1.31229620e-04, 1.33746488e-04, 1.52264194e-04, 1.39088531e-04, 1.18928827e-04, 1.17896220e-04, 1.12746394e-04, 1.19514902e-04, 1.18938473e-04, 1.24787268e-04, 1.20815800e-04, 1.12248295e-04, 1.18788432e-04, 1.01365989e-04, 8.51736677e-05, 8.02599731e-05, 9.26018605e-05, 1.02932187e-04, 8.73052579e-05, 8.64193243e-05, 8.91090041e-05, 9.11837248e-05, 9.00273650e-05, 8.35614333e-05, 8.10481055e-05, 7.07822970e-05, 7.32719951e-05, 7.72961943e-05, 7.28543449e-05, 6.69122844e-05, 6.61182322e-05, 7.63999076e-05, 7.25856200e-05, 6.86758599e-05, 6.89267057e-05, 6.95440707e-05, 7.09900125e-05, 6.21983283e-05, 6.22718616e-05, 6.09387267e-05, 6.02835003e-05, 6.32692403e-05, 5.86089890e-05, 5.49296230e-05, 4.65582454e-05, 5.02560024e-05, 5.68807628e-05, 5.53049445e-05, 5.48354906e-05, 4.85464590e-05, 4.90749424e-05, 4.50639432e-05, 4.20226741e-05])
            
    # Function to extend wings
    # def add_wings(a, pts):
    #     w = min(a)*(np.arange(pts)/pts)*50
    #     a = np.concatenate([np.abs(np.random.normal(w,w)),a,np.abs(np.random.normal(w[::-1],w[::-1]))])
    #
    #     return a
        
    # Extend the wings for a nice wide PSF that tapers off rather than ending sharply for bright targets
    # if extend:
    #     lpsf = add_wings(lpsf.copy(), extend)
    
    # Shift the psf so that the center is 0
    l = len(psf)
    x0 = l//2
    x = np.arange(l)-x0
    
    # Interpolate lpsf to distance
    val = np.interp(distance, x, psf)
    
    if plot:
        plt.plot(x, psf)
        plt.scatter(distance, val, c='r', zorder=5)
        
    return val

def lambda_lightcurve(wavelength, response, distance, pfd2adu, ld_coeffs, ld_profile, star, planet, time, params, filt, trace_radius=25, snr=100, floor=2, extend=25, plot=False):
    """
    Generate a lightcurve for a given wavelength
    
    Parameters
    ----------
    wavelength: float
        The wavelength value in microns
    response: float
        The spectral response of the detector at the given wavelength
    distance: float
        The Euclidean distance from the center of the cross-dispersed PSF
    ld_coeffs: array-like
        A 3D array that assigns limb darkening coefficients to each pixel, i.e. wavelength
    ld_profile: str
        The limb darkening profile to use
    pfd2adu: sequence
        The factor that converts photon flux density to ADU/s
    star: sequence
        The wavelength and flux of the star
    planet: sequence
        The wavelength and Rp/R* of the planet at t=0 
    t: sequence
        The time axis for the TSO
    params: batman.transitmodel.TransitParams
        The transit parameters of the planet
    throughput: float
        The CLEAR or F277W filter throughput at the given wavelength
    trace_radius: int
        The radius of the trace
    snr: float
        The signal-to-noise for the observations
    floor: int
        The noise floor in counts
    extend: int
        The number of points to extend the lpsf wings by
    plot: bool
        Plot the lightcurve
    
    Returns
    -------
    sequence
        A 1D array of the lightcurve with the same length as *t* 
    """
    nframes = len(time)
    
    # If it's a background pixel, it's just noise
    if distance>trace_radius+extend \
    or wavelength<np.nanmin(star[0].value) \
    or (filt=='F277W' and wavelength<2.36989) \
    or (filt=='F277W' and wavelength>3.22972):
        
        # flux = np.abs(np.random.normal(loc=floor, scale=1, size=nframes))
        flux = np.repeat(floor, nframes)
        
    else:
        
        # I = (Stellar Flux)*(LDC)*(Transit Depth)*(Filter Throughput)*(PSF position)
        # Don't use astropy units! It quadruples the computing time!
        
        # Get the energy flux density [erg/s/cm2/A] at the given wavelength [um] at t=t0
        flux0 = np.interp(wavelength, star[0], star[1], left=0, right=0)
        
        # Convert from energy flux density to photon flux density [photons/s/cm2/A]
        # by multiplying by (lambda/h*c)
        flux0 *= wavelength*503411665111.4543 # [1/erg*um]
        
        # Convert from photon flux density to ADU/s by multiplying by the 
        # wavelength interval [um/pixel], primary mirror area [cm2], and gain [ADU/e-]
        flux0 *= pfd2adu
        
        # Expand to shape of time axis and add noise
        # flux0 = np.abs(flux0)
        # flux = np.abs(np.random.normal(loc=flux0, scale=flux0/snr, size=len(time)))
        flux = np.repeat(flux0, nframes)
        
        # If there is a transiting planet...
        if not isinstance(planet,str):
            
            # Set the wavelength dependent orbital parameters
            params.limb_dark = ld_profile
            params.u = ld_coeffs
            
            # Set the radius at the given wavelength from the transmission spectrum (Rp/R*)**2
            tdepth = np.interp(wavelength, planet[0], planet[1])
            params.rp = np.sqrt(tdepth)
            
            # Generate the light curve for this pixel
            model = batman.TransitModel(params, time) 
            lightcurve = model.light_curve(params)
            
            # Scale the flux with the lightcurve
            flux *= lightcurve
            
        # Apply the filter response
        flux *= response
        
        # Scale pixel based on distance from the center of the cross-dispersed psf
        flux *= psf_position(distance, filt=filt, extend=extend)
        
        # Replace very low signal pixels with noise floor
        # flux[flux<floor] += np.random.normal(loc=floor, scale=1, size=len(flux[flux<floor]))
        flux[flux<floor] += np.repeat(floor, len(flux[flux<floor]))
        
        # Plot
        if plot:
            plt.plot(t, flux)
            plt.xlabel("Time from central transit")
            plt.ylabel("Flux Density [photons/s/cm2/A]")
        
    return flux

def wave_solutions(subarr, directory=DIR_PATH+'/files/soss_wavelengths_fullframe.fits'):
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

class TSO(object):
    """
    Generate NIRISS SOSS time series observations
    """

    def __init__(self, ngrps, nints, star,
                        planet      = '', 
                        params      = '', 
                        ld_coeffs   = '', 
                        ld_profile  = 'quadratic',
                        snr         = 700,
                        subarray    = 'SUBSTRIP256',
                        t0          = 0,
                        extend      = 25, 
                        trace_radius= 50, 
                        target      = ''):
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
        planet: sequence (optional)
            The wavelength and Rp/R* of the planet at t=0 
        params: batman.transitmodel.TransitParams (optional)
            The transit parameters of the planet
        ld_coeffs: array-like (optional)
            A 3D array that assigns limb darkening coefficients to each pixel, i.e. wavelength
        ld_profile: str (optional)
            The limb darkening profile to use
        snr: float
            The signal-to-noise
        subarray: str
            The subarray name, i.e. 'SUBSTRIP256', 'SUBSTRIP96', or 'FULL'
        t0: float
            The start time of the exposure
        extend: int
            The number of pixels to extend the wings of the pfs
        trace_radius: int
            The radius of the trace
        target: str (optional)
            The name of the target
        """
        # Set instance attributes for the exposure
        self.subarray     = subarray
        self.nrows        = 256 if '256' in subarray else 96 if '96' in subarray else 2048
        self.ncols        = 2048
        self.ngrps        = ngrps
        self.nints        = nints
        self.nresets      = 1
        self.time         = get_frame_times(self.subarray, self.ngrps, self.nints, t0, self.nresets)
        self.nframes      = len(self.time)
        self.target       = target or 'Simulated Target'
        self.obs_date     = ''
        self.filter       = 'CLEAR'
        self.header       = ''
        
        # Set instance attributes for the target
        self.star         = star
        self.planet       = planet
        self.params       = params
        self.ld_coeffs    = ld_coeffs
        self.ld_profile   = ld_profile or 'quadratic'
        self.trace_radius = trace_radius
        self.snr          = snr
        self.extend       = extend
        self.wave         = wave_solutions(str(self.nrows))
        
        # Calculate a map for each order that converts photon flux density to ADU/s
        self.gain = 1.61 # [e-/ADU]
        self.primary_mirror = 253260 # [cm2]
        avg_wave = np.mean(self.wave, axis=1)
        self.pfd2adu = np.ones((3,self.ncols*self.nrows))
        for n,aw in enumerate(avg_wave):
            coeffs = np.polyfit(aw[:-1], np.diff(aw), 1)
            wave_int = (np.polyval(coeffs, self.wave[n])*q.um).to(q.AA)
            self.pfd2adu[n] = (wave_int*self.primary_mirror*q.cm**2/self.gain).value.flatten()
        
        # Add the orbital parameters as attributes
        for p in [i for i in dir(self.params) if not i.startswith('_')]:
            setattr(self, p, getattr(self.params, p))
        
        # Create the empty exposure
        self.tso = np.zeros((self.nframes, self.nrows, self.ncols))
        self.tso_order1 = np.zeros((self.nframes, self.nrows, self.ncols))
        self.tso_order2 = np.zeros((self.nframes, self.nrows, self.ncols))
    
    def run_simulation(self, orders=[1,2], filt='CLEAR'):
        """
        Generate the simulated 2D data given the initialized TSO object
        
        Parameters
        ----------
        orders: sequence
            The orders to simulate
        filt: str
            The element from the filter wheel to use, i.e. 'CLEAR' or 'F277W'
        """
        # Set single order to list
        if isinstance(orders,int):
            orders = [orders]
        if not all([o in [1,2] for o in orders]):
            raise TypeError('Order must be either an int, float, or list thereof; i.e. [1,2]')
        orders = list(set(orders))
        
        # Check if it's F277W to speed up calculation
        if 'F277W' in filt.upper():
            orders = [1]
            self.filter = 'F277W'
            
        # Make dummy array of LDCs if no planet (required for multiprocessing)
        if isinstance(self.planet, str):
            self.ld_coeffs = np.zeros((2, self.nrows*self.ncols, 2))
            
        # Generate simulation for each order
        for order in orders:
            
            # Get the wavelength map
            local_wave = self.wave[order-1].flatten()
            
            # Get the distance map 
            local_distance = distance_map(order=order).flatten()
            
            # Get limb darkening map
            local_ld_coeffs = self.ld_coeffs.copy()[order-1]
            
            # Get relative spectral response map
            throughput = np.genfromtxt(DIR_PATH+'/files/gr700xd_{}_order{}.dat'.format(self.filter,order), unpack=True)
            local_response = np.interp(local_wave, throughput[0], throughput[-1], left=0, right=0)
            
            # Get the wavelength interval per pixel map
            local_pfd2adu = self.pfd2adu[order-1]
            
            # Run multiprocessing
            print('Calculating order {} light curves...'.format(order))
            start = time.time()
            pool = multiprocessing.Pool(8)
            
            # Set wavelength independent inputs of lightcurve function
            func = partial(lambda_lightcurve, 
                           ld_profile    = self.ld_profile,
                           star          = self.star,
                           planet        = self.planet,
                           time          = self.time,
                           params        = self.params,
                           filt          = self.filter,
                           trace_radius  = self.trace_radius,
                           snr           = self.snr,
                           extend        = self.extend)
                    
            # Generate the lightcurves at each pixel
            lightcurves = pool.starmap(func, zip(local_wave, local_response, local_distance, local_pfd2adu, local_ld_coeffs))
            
            # Close the pool
            pool.close()
            pool.join()
            
            # Clean up and time of execution
            tso_order = np.asarray(lightcurves).swapaxes(0,1).reshape([self.nframes, self.nrows, self.ncols])
            
            print('Order {} light curves finished: '.format(order), time.time()-start)
            
            # Add to the master TSO
            self.tso += tso_order
            
            # Add it to the individual order
            setattr(self, 'tso_order{}'.format(order), tso_order)
            
        # Add noise to the observations using Kevin Volk's dark ramp simulator
        # self.tso += dark_ramps(self.time, self.subarray)
    
    def add_noise_model(self):
        """
        Generate the noise model and add to the simulation
        """
        pass
    
    def plot_frame(self, frame='', scale='linear', order='', cmap=cm.jet):
        """
        Plot a frame of the TSO
        
        Parameters
        ----------
        frame: int
            The frame number to plot
        scale: str
            Plot in linear or log scale
        order: int (optional)
            The order to isolate
        cmap: str
            The color map to use
        """
        if order:
            tso = getattr(self, 'tso_order{}'.format(order))
        else:
            tso = self.tso
        
        vmax = int(np.nanmax(tso))
        
        plt.figure(figsize=(13,2))
        if scale=='log':
            plt.imshow(tso[frame or self.nframes//2].data, origin='lower', interpolation='none', norm=matplotlib.colors.LogNorm(), vmin=1, vmax=vmax, cmap=cmap)
        else:
            plt.imshow(tso[frame or self.nframes//2].data, origin='lower', interpolation='none', vmin=1, vmax=vmax, cmap=cmap)
        plt.colorbar()
        plt.title('Injected Spectrum')
    
    def plot_snr(self, frame='', cmap=cm.jet):
        """
        Plot a frame of the TSO
        
        Parameters
        ----------
        frame: int
            The frame number to plot
        """
        snr  = np.sqrt(self.tso[frame or self.nframes//2].data)
        vmax = int(np.nanmax(snr))
        
        plt.figure(figsize=(13,2))
        plt.imshow(snr, origin='lower', interpolation='none', vmin=1, vmax=vmax, cmap=cmap)
        
        plt.colorbar()
        plt.title('SNR over Spectrum')
        
    def plot_saturation(self, frame='', saturation = 80.0, cmap=cm.jet):
        """
        Plot a frame of the TSO
        
        Parameters
        ----------
        frame: int
            The frame number to plot
        
        fullWell: percentage [0-100] of maximum value, 65536
        """
        
        fullWell    = 65536.0
        
        saturated = np.array(self.tso[frame or self.nframes//2].data) > (saturation/100.0) * fullWell
        
        plt.figure(figsize=(13,2))
        plt.imshow(saturated, origin='lower', interpolation='none', cmap=cmap)
        
        plt.colorbar()
        plt.title('Saturated Pixels')
    
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
            tso = getattr(self, 'tso_order{}'.format(order))
        else:
            tso = self.tso
            
        f = tso[frame].T
        
        if isinstance(col, int):
            col = [col]
            
        for c in col:
            plt.plot(f[c], label='Column {}'.format(c), **kwargs)
            
        plt.xlim(0,256)
        
        plt.legend(loc=0, frameon=False)
        
    def plot_lightcurve(self, col):
        """
        Plot a lightcurve for each column index given
        
        Parameters
        ----------
        col: int, sequence
            The column index(es) to plot a light curve for
        """
        if isinstance(col, int):
            col = [col]
        
        for c in col:
            # ld = self.ldc[c*self.tso.shape[1]]
            w = np.mean(self.wave[0], axis=0)[c]
            f = np.nansum(self.tso[:,:,c], axis=1)
            f *= 1./np.nanmax(f)
            plt.plot(self.time/3000., f, label='Col {}'.format(c), marker='.', ls='None')
            
        # Plot whitelight curve too
        # plt.plot(self.time)
            
        plt.legend(loc=0, frameon=False)
        
    def plot_spectrum(self, frame=0, order=''):
        """
        Parameters
        ----------
        frame: int
            The frame number to plot
        """
        if order:
            tso = getattr(self, 'tso_order{}'.format(order))
        else:
            tso = self.tso
        
        # Get extracted spectrum
        wave = np.mean(self.wave[0], axis=0)
        flux = np.sum(tso[frame].data, axis=0)
        
        # Deconvolve with the grism
        throughput = np.genfromtxt(DIR_PATH+'/files/gr700xd_{}_order{}.dat'.format(self.filter,order or 1), unpack=True)
        flux *= np.interp(wave, throughput[0], throughput[-1], left=0, right=0)
        
        # Convert from ADU/s to photon flux density
        wave_int = np.diff(wave)*q.um.to(q.AA)
        flux /= (np.array(list(wave_int)+[wave_int[-1]])*self.primary_mirror*q.cm**2/self.gain).value.flatten()
        
        # Convert from photon flux density to energy flux density
        flux /= wave*503411665111.4543 # [1/erg*um]
        
        # Plot it along with input spectrum
        plt.figure(figsize=(13,2))
        plt.plot(wave, flux, label='Extracted')
        plt.plot(*self.star, label='Injected')
    
    def save_tso(self, filename='dummy.save'):
        """
        Save the TSO data to file
        
        Parameters
        ----------
        filename: str
            The path of the save file
        """
        print('Saving TSO class dict to {}'.format(filename))
        joblib.dump(self.__dict__, filename)
    
    def load_tso(self, filename):
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
        cards = [('DATE', datetime.datetime.now().strftime("%Y-%m-%d%H:%M:%S"), 'Date file created yyyy-mm-ddThh:mm:ss, UTC'),
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
                ('TIME-OBS', self.obs_date, 'UT time at the start of exposure'),
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
                ('ZEROFRAM', False, 'Zero frame was downlinkws separately'),
                ('DATAPROB', False, 'Science telemetry indicated a problem'),
                ('SCA_NUM', 496, 'Sensor Chip Assembly number'),
                ('DATAMODE', 91, 'post-processing method used in FPAP'),
                ('COMPRSSD', False, 'data compressed on-board (T/F)'),
                ('SUBARRAY', 'SUBSTRIP256', 'Subarray pattern name'),
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
                ('BZERO', 32768, '')]
        
        # Make the header
        prihdr = fits.Header()
        for card in cards:
            prihdr.append(card, end=True)
        
        # Store the header in the object too
        self.header = prihdr
        
        # Make the HDUList
        prihdu  = fits.PrimaryHDU(header=prihdr)
        sci_hdu = fits.ImageHDU(data=self.tso, name='SCI')
        hdulist = fits.HDUList([prihdu, sci_hdu])
        
        # Write the file
        hdulist.writeto(outfile, overwrite=True)
        hdulist.close()
        
        print('File saved as',outfile)

