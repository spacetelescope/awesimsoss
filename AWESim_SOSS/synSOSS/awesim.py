import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import batman
# import pickle
import astropy.units as q
import astropy.constants as ac
import multiprocessing
import time
from ExoCTK import svo
from ExoCTK import core
from ExoCTK.ldc import ldcfit as lf
from astropy.io import fits
from scipy.optimize import curve_fit
from functools import partial

from sklearn.externals import joblib

cm = plt.cm

FILTERS = svo.filters()
dir_path = os.path.dirname(os.path.realpath(__file__))

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
    scaling = np.genfromtxt(dir_path+'/refs/GR700XD_{}.txt'.format(order), unpack=True)
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

def ld_coefficient_lookup(wave_map, ld_profile, grid_point, model_grid):
    """
    Generate limb darkening coefficients for the wavelength at every pixel
    
    Example
    -------
    ld_coeffs_lookup = ld_coefficient_lookup(lam1, 'quadratic', star, model_grid)
    """
    wave_list = wave_map.flatten()
    filt = os.path.join(os.path.join(os.path.dirname(core.__file__),'AWESim_SOSS/data/filters/NIRISS.GR700XD.1.txt'))
    throughput = np.genfromtxt(filt, unpack=True)
    ld_coeffs = []
    
    # Generate a lookup
    lookup = {}
    delta_w = 0.005 #np.nanmean(np.diff(sorted(wave_list)))
    for bin in np.arange(np.min(wave_map), np.max(wave_map), delta_w):
        try:
            bandpass = svo.Filter('GR700XD', throughput, n_bins=1, wl_min=(bin-delta_w/2.)*q.um, wl_max=(bin+delta_w/2.)*q.um)
            ldcs = lf.ldc(None, None, None, model_grid, [ld_profile], bandpass=bandpass, grid_point=grid_point.copy(), mu_min=0.08)
            coeffs = list(zip(*ldcs[ld_profile]['coeffs']))[1::2]
            coeffs = [coeffs[0][0],coeffs[1][0]]
            lookup['{:.9f}'.format(bin)] = coeffs
        except:
            pass
    
    print('Finished making LDC lookup table.')

    return lookup

def ld_coefficients(wave_map, lookup):
    """
    Generate limb darkening coefficients for the wavelength at every pixel
    
    Example
    -------
    coeff_map = ld_coefficients(lam1, ld_coeffs_lookup)
    """
    wave_list = wave_map.flatten()
    ld_coeffs = np.zeros((len(wave_list),2))
    delta_w = 0.005/2. #np.nanmean(np.diff(sorted(wave_list)))
    l = np.array(list(map(float,lookup)))
    
    # Get all the values 
    done = np.zeros(wave_list.shape)
    for w,d in zip(wave_list,done):
        if not d:
            
            try:
                # Determine coeffs
                W, = l[(w>=l-delta_w)&(w<l+delta_w)]
                print(W)
                
                # Put coeffs into the array for all wavelengths in the bin
                pix = np.where((w>=l-delta_w)&(w<l+delta_w))
                ld_coeffs[pix] = lookup['{:.9f}'.format(W)]
                done[pix] = 1
            except:
                pass
    
    print('All coefficients calculated.')
    
    return ld_coeffs

def soss_polynomials(plot=False):
    ''' LEGACY CODE '''
    # Load the trace masks
    path = '/Users/jfilippazzo/Documents/Modules/NIRISS/soss_extract_spectrum/'
    mask1 = joblib.load(path+'order1_mask.save').swapaxes(-1,-2)
    mask2 = joblib.load(path+'order2_mask.save').swapaxes(-1,-2)
    spec1 = np.ma.array(np.ones(mask1.shape), mask=mask1)
    spec2 = np.ma.array(np.ones(mask2.shape), mask=mask2)
    
    # Generate the polynomials
    poly1 = trace_polynomial(spec1, start=4, end=2040, order=4)
    poly2 = trace_polynomial(spec2, start=470, end=2040, order=4)
    
    # Plot
    plt.figure(figsize=(13,2))
    file = open(path+'/trace_mask.p', 'rb')
    trace = pickle.load(file, encoding='latin1')[::-1,::-1]
    plt.imshow(trace.data, origin='lower', norm=matplotlib.colors.LogNorm())
    plt.plot(*poly1)
    plt.plot(*poly2)
    plt.xlim(0,2048)
    plt.ylim(0,256)
    
    return poly1, poly2

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
        
        path = '/Users/jfilippazzo/Documents/Modules/NIRISS/soss_extract_spectrum/'
        mask = joblib.load(path+'order{}_mask.save'.format(order)).swapaxes(-1,-2)
        
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
                
        joblib.dump(d_map, 'AWESim_SOSS/structSOSS/order_{}_distance_map.save'.format(order))
        
    else:
        d_map = joblib.load('AWESim_SOSS/structSOSS/order_{}_distance_map.save'.format(order))
        
    
    if plot:
        plt.figure(figsize=(13,2))
        
        plt.title('Order {}'.format(order))
        
        plt.imshow(d_map, interpolation='none', origin='lower', norm=matplotlib.colors.LogNorm())
        
        plt.colorbar()
    
    return d_map

def psf_position(distance, extend=25, plot=False):
    """
    Scale the flux based on the pixel's distance from the center of the cross dispersed psf
    """
    # Get the LPSF
    lpsf = np.array([7.701976722368496E-008,  1.540395344473699E-007,  3.080790688947398E-007, \
                     6.161581377894797E-007,  1.232316275578959E-006,  2.464632551157919E-006, \
                     4.929265102315838E-006,  9.714837387708730E-006,  5.671904909021475E-006, \
                     4.548023730510664E-006,  1.022713226439542E-005,  6.886893882507295E-006, \
                     8.177790144225927E-006,  1.357109057534278E-005,  8.916710340478584E-006, \
                     1.239566539967818E-005,  2.781745489985332E-005,  2.509716449416999E-005, \
                     2.631011432652208E-005,  4.830151269574756E-005,  5.380108778450451E-005, \
                     7.547263667725956E-005,  1.022118883162726E-004,  1.420077972523748E-004, \
                     2.362241206079752E-004,  3.385566380821325E-004,  5.477043893594158E-004, \
                     6.696818098559376E-004,  5.493319867611035E-004,  4.720754680409556E-004, \
                     2.991750642213908E-004,  3.058475983204190E-004,  3.109660592775787E-004, \
                     2.226914950899106E-004,  2.979418360802288E-004,  3.397708704659941E-004, \
                     2.990017218531538E-004,  2.758223087866440E-004,  3.294162992516503E-004, \
                     2.381257536346881E-004,  3.407167609725814E-004,  5.361983993812380E-004, \
                     6.230353641937803E-004,  6.140843798414508E-004,  5.070604643273580E-004, \
                     3.306009460586345E-004,  2.371751859966409E-004,  1.155928608405077E-004, \
                     6.370671124544813E-005,  7.242587988226523E-005,  3.417951946560471E-005, \
                     2.799611461752616E-005,  3.403609616187131E-005,  1.659919620922157E-005, \
                     1.873137450902895E-005,  1.825263581423098E-005,  9.144579730557822E-006, \
                     8.962197003747896E-006,  9.059127708432868E-006,  5.408733372069818E-006, \
                     7.623564329893584E-006,  7.990429056435600E-006,  4.048852234816991E-006, \
                     6.761537376720472E-006])
                     
    # Function to extend wings
    def add_wings(a, pts):
        w = min(a)*(np.arange(pts)/pts)*50
        a = np.concatenate([np.abs(np.random.normal(w,w)),a,np.abs(np.random.normal(w[::-1],w[::-1]))])
        
        return a
        
    # Extend the wings for a nice wide PSF that tapers off rather than ending sharply for bright targets
    if extend:
        lpsf = add_wings(lpsf.copy(), extend)
        
    # Scale the transmission to 1
    psf = lpsf/np.trapz(lpsf)
    
    # Interpolate lpsf to distance
    p0 = len(psf)//2
    val = np.interp(distance, range(len(psf[p0:])), psf[p0:])
    
    if plot:
        plt.plot(range(len(psf[p0:])), psf[p0:])
        plt.scatter(distance, val, c='r', zorder=5)
        
    return val

def lambda_lightcurve(wavelength, response, distance, ld_coeffs, ld_profile, star, planet, time, params, trace_radius=25, SNR=100, floor=2, extend=25, plot=False):
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
    star: sequence
        The wavelength and flux of the star
    planet: sequence
        The wavelength and Rp/R* of the planet at t=0 
    t: sequence
        The time axis for the TSO
    params: batman.transitmodel.TransitParams
        The transit parameters of the planet
    trace_radius: int
        The radius of the trace
    SNR: float
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
    # print(distance, trace_radius)
    # If it's a background pixel, it's just noise
    if distance>trace_radius+extend \
        or wavelength<np.nanmin(star[0].value):
        
        flux = np.abs(np.random.normal(loc=floor, scale=1, size=len(time)))
        
    else:
        
        # Get the stellar flux at the given wavelength at t=t0
        flux0 = np.interp(wavelength, star[0], star[1])
        
        # Expand to shape of time axis and add noise
        flux = np.abs(np.random.normal(loc=flux0, scale=flux0/SNR, size=len(time)))
        
        # If there is a transiting planet...
        if not isinstance(planet,str):
            
            # Set the wavelength dependent orbital parameters
            params.limb_dark = ld_profile
            params.u         = ld_coeffs
            
            # Set the radius at the given wavelength from the transmission spectrum (Rp/R*)**2
            tdepth    = np.interp(wavelength, planet[0], planet[1])
            params.rp = np.sqrt(tdepth)
            
            # Generate the light curve for this pixel
            model      = batman.TransitModel(params, time) 
            lightcurve = model.light_curve(params)
            
            # Scale the flux with the lightcurve
            flux *= lightcurve
            
        # Convert the flux into counts
        flux /= response
        
        # Scale pixel based on distance from the center of the cross-dispersed psf
        flux *= psf_position(distance, extend=extend)
        
        # Replace very low signal pixels with noise floor
        flux[flux<floor] += np.random.normal(loc=floor, scale=1, size=len(flux[flux<floor]))
        
        # Plot
        if plot:
            plt.plot(t, flux)
            plt.xlabel("Time from central transit")
            plt.ylabel("Flux [erg/s/cm2/A]")
        
    return flux

def wave_solutions(subarr, directory=dir_path+'/refs/soss_wavelengths_fullframe.fits'):
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

class TSO(object):
    """
    Generate NIRISS SOSS time series observations
    """

    def __init__(self, time, star, 
                        planet      = '', 
                        params      = '', 
                        ld_coeffs   = '', 
                        ld_profile  = 'quadratic',
                        trace_radius= 50, 
                        SNR         = 700, 
                        extend      = 25, 
                        subsize     = 256):
        """
        Iterate through all pixels and generate a light curve if it is inside the trace
        
        Parameters
        ----------
        t: sequence
            The time axis for the TSO
        star: sequence
            The wavelength and flux of the star
        planet: sequence
            The wavelength and Rp/R* of the planet at t=0 
        params: batman.transitmodel.TransitParams
            The transit parameters of the planet
        ld_coeffs: array-like
            A 3D array that assigns limb darkening coefficients to each pixel, i.e. wavelength
        ld_profile: str
            The limb darkening profile to use
        trace_radius: int
            The radius of the trace
        """
        #super(TSO, self).__init__(time, star, **kwargs)
        
        defaults = dict(planet      = '', 
                        params      = '', 
                        ld_coeffs   = '', 
                        ld_profile  = 'quadratic',
                        trace_radius= 50, 
                        SNR         = 700, 
                        extend      = 25, 
                        subsize     = 256)
        
        # nArgs     = len(args)
        # usedKeys  = np.array(list(defaults.keys()))[:nArgs]
        # for k, key in enumerate(usedKeys):
        #     print("self." + key + " = args[{}]".format(k))
        #     exec( "self.__dict__['" + key + "'] = args[{}]".format(k), self.__dict__)
        #     # setattr(self, key, arg)
        
        # while len(args) < nArgs:
        #     args += kwargs.popitem()
        #     super(TSO, self).__init__(*args[:nArgs], **kwargs)
        
        # Assign local to class values
        self.nwaves      = 2048
        
        self.time = time#kwargs['time'] if 'time' in kwargs.keys() else args[0]
        self.star = star#kwargs['star'] if 'star' in kwargs.keys() else args[1]
        
        self.ntimes      = len(self.time)
        self.planet      = planet
        self.params      = params
        self.ld_coeffs   = ld_coeffs
        self.ld_profile  = ld_profile
        self.trace_radius= trace_radius
        self.SNR         = SNR
        self.extend      = extend
        self.subsize     = subsize
        # for key in defaults:
        #     if key not in usedKeys and key in kwargs.keys():
        #         exec("self." + key + " = kwargs['" + key + "']")
        #     else:
        #         exec("self." + key + " = defaults['" + key + "']")
        
        # Save some attributes
        self.wave = wave_solutions(str(self.subsize))
        
        for p in [i for i in dir(self.params) if not i.startswith('_')]:
            setattr(self, p, getattr(self.params, p))
        
        self.tso = np.zeros((self.ntimes, self.subsize, self.nwaves))
    
    def compute_light_curve_order(self, orders=[1,2], nOrders=2):
        # Flatten the wavelength and distance maps
        
        if isinstance(self.planet,str):
            if self.ld_profile == 'linear':
                raise ValueError("ld_profile == 'linear' has not been implemented yet! :(")
                print('Why are you setting the linear LDC to zero?')
                local_ld_coeffs = np.zeros((self.subsize*self.nwaves, 1))
            elif self.ld_profile == 'quadratic':
                print('Why are you setting the quadratic LDCs to zero?')
                local_ld_coeffs = np.zeros((self.subsize*self.nwaves, 2))
            else:
                raise ValueError("`limb_dark` must be either `'linear'` or `'quadratic'`")
        else:
            print('Using Injected Limb Darkenging Coefficients')
            local_ld_coeffs = self.ld_coeffs.copy()
        
        if isinstance(orders,int) or isinstance(orders,float):
            orders = [int(orders)]
        
        if not isinstance(orders, list):
            raise TypeError('order must be either an int, float, or list thereof; i.e. [1,2]')
        
        # only use valid and unique values of [1,2]
        order = []
        for o in orders:
            if o in [1,2]: # exclude non
                order.append(o)
            else:
                print('{} is not a valid choice of `orders`; \
                            must be either [1,2] or both'.format(o))
        
        orders = list(set(order)) # only keep unique values of [1,2]
        
        for order in orders:
            local_wave      = self.wave[order-1].flatten()
            local_distance  = distance_map(order=order).flatten()
            
            # Get relative spectral response to convert flux to counts
            local_scaling   = ADUtoFlux(order)
            
            # Get relative spectral response to convert flux to counts
            # ===============================================================================
            # ======================== HERE BE DRAGONS!!!! ==================================
            # ===============================================================================
            # Order 2 scaling too bright! Fix factor of 50 below!
            # ===============================================================================
            dragons = [1,50] # remove the dragons
            
            local_response  = np.interp(local_wave, 
                                        local_scaling[0], 
                                        local_scaling[1])/dragons[order-1]
            
            # Required for multiprocessing...
            # Run multiprocessing
            print('Calculating order {} light curves...'.format(order))
            
            processes = 8
            start = time.time()
            pool = multiprocessing.Pool(processes)
            
            func = partial(lambda_lightcurve, 
                    ld_profile  = self.ld_profile, 
                    star        = self.star, 
                    planet      = self.planet, 
                    time        = self.time, 
                    params      = self.params, 
                    trace_radius= self.trace_radius, 
                    SNR         = self.SNR, 
                    extend      = self.extend)
            
            lightcurves = pool.starmap(func, zip(local_wave, 
                                                 local_response, 
                                                 local_distance, 
                                                 local_ld_coeffs)
                                      )
            
            pool.close()
            pool.join()
            # Clean up and time of execution
            tso_order = np.asarray(lightcurves).swapaxes(0,1).reshape([len(self.time),
                                                                           self.subsize,
                                                                           self.nwaves])
            
            print('Order {} light curves finished: '.format(order), time.time()-start)
            
            self.tso = np.abs(self.tso+tso_order)
        
        print('Why are you subtracting Gaussian noise -- centered at 1 -- \
                    instead of adding it -- centered at 0?')
        
        # self.noise = np.random.normal(loc=1, scale=1, size=self.tso.shape)
        # self.tso  = np.abs(self.tso - noise)
        self.tso  = np.abs(self.tso)
    
    def plot_frame(self, frame='', scale='log', cmap=cm.jet):
        """
        Plot a frame of the TSO
        
        Parameters
        ----------
        frame: int
            The frame number to plot
        """
        vmax = int(np.nanmax(self.tso))
        
        plt.figure(figsize=(13,2))
        if scale=='log':
            plt.imshow(self.tso[frame or len(self.time)//2].data, origin='lower', interpolation='none', norm=matplotlib.colors.LogNorm(), vmin=1, vmax=vmax, cmap=cmap)
        else:
            plt.imshow(self.tso[frame or len(self.time)//2].data, origin='lower', interpolation='none', vmin=1, vmax=vmax, cmap=cmap)
        plt.colorbar()
        plt.title('Injected Spectrum')
        
    def plot_slice(self, col, trace='tso', frame=0, **kwargs):
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
        f = getattr(self, trace)[frame].T
        
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
            plt.plot(self.time, f, label='Col {}'.format(c), marker='.', ls='None')
            
        # Plot whitelight curve too
        # plt.plot(self.time)
            
        plt.legend(loc=0, frameon=False)
    
    def save_tso(self, filename='dummy.save'):
        print('Saving TSO class dict to {}'.format(filename))
        joblib.dump(self.__dict__, filename)
    
    def load_tso(self, filename):
        print('Loading TSO class dict to {}'.format(filename))
        load_dict = joblib.load(filename)
        # for p in [i for i in dir(load_dict)]:
        #     setattr(self, p, getattr(params, p))
        for key in load_dict.keys():
            exec("self." + key + " = load_dict['" + key + "']")