import os
import multiprocessing
import time
import numpy as np
import scipy.signal as ss
import astropy.units as q
import matplotlib.pyplot as plt
import matplotlib
import warnings
import AWESim_SOSS
from functools import partial
from numpy import ma
from scipy.optimize import curve_fit
from astropy.io import fits
from matplotlib.colors import LogNorm
from skimage import measure
from scipy.ndimage.interpolation import map_coordinates

warnings.simplefilter('ignore')

dir_path = os.path.dirname(os.path.realpath(AWESim_SOSS.__file__))

WS = {1: np.array([2.60188,-0.000984839,3.09333e-08,-4.19166e-11,1.66371e-14]),
      2: np.array([1.30816,-0.000480837,-5.21539e-09,8.11258e-12,5.77072e-16]),
      3: np.array([0.880545,-0.000311876,8.17443e-11,0.0,0.0])}

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
    scaling = np.genfromtxt(os.path.dirname(AWESim_SOSS.__file__)+'/files/GR700XD_{}.txt'.format(order), unpack=True)
    scaling[1] *= ADU2mJy*mJy2erg
    
    return scaling

def bimodal(x, mu1, sigma1, A1, mu2, sigma2, A2):
    """
    Generate a bimodal function of two Gaussians of the given parameters
    
    Parameters
    ----------
    x: array-like
        The x-axis on which to generate the function
    mu1: float
        The x-position of the first peak center
    sigma1: float
        The stanfard deviation of the first distribution
    A1: float
        The amplitude of the first peak
    mu2: float
        The x-position of the second peak center
    sigma2: float
        The stanfard deviation of the second distribution
    A2: float
        The amplitude of the second peak

    Returns
    -------
    np.ndarray
        The y-axis values of the mixed Gaussian
    """
    B = gaussian(x,mu1,sigma1,A1)+gaussian(x,mu2,sigma2,A2)
     
    return B

def fetch_pixels(lam, d_lam, order, masked_trace, plot=False):
    """
    Get all the pixels for a given wavelength bin
    
    Parameters
    ----------
    lam: float
        The center of the wavelength bin in microns
    d_lam: float
        The radius of the wavelength bin in microns
    order: int
        The order to extract, [1, 2]
    masked_trace: np.ma.array
        The SOSS frame with the masked trace
    bkg: array-like
        The mean background for each column in ADU
    plot: bool
        Plot the slice from the masked trace
    
    Returns
    -------
    float
        The scaled flux value in the wavelength bin
    """
    # Get the wavelength map
    wavelength = wave_solutions('256')[order-1]
    
    try:
        
        # Find all the pixels in the wavelength bin lam-d_lam < lam < lam+d_lam
        pixels = np.where(np.logical_and(wavelength>=lam-d_lam,wavelength<lam+d_lam))
        mask = masked_trace.T.mask
        col = np.ones(mask.shape)
        col[pixels] = 0
        
        # Get the masked pixels in ADU
        adu = np.ma.array(masked_trace.T.copy(), mask=np.logical_xor(col, mask))
        msk = adu.mask
        
        # Counts
        n_pix = np.size(msk)-np.count_nonzero(msk)
        n_cols = np.count_nonzero(np.sum(msk, axis=0)-msk.shape[0])
        
        # Plot
        if plot:
            print('Order {}: {} pixels in {} columns for wavelength bin {} < lambda < {}'.format(order, n_pix, n_cols, lam-d_lam, lam+d_lam))
                  
            plt.figure(figsize=(12,2))
            plt.imshow(adu, interpolation='none', origin='lower', cmap=plt.cm.Blues)
            plt.title('Order {}'.format(order))
            
    except:
        
        # If something goes wrong, just return all masked pixels
        msk = np.ones(wavelength.shape, dtype=bool)
        
    return msk

def batmen(x, mu1, sigma1, A1, sigma2, A2, sep, mu3, sigma3, A3, sigma4, A4):
    """
    Generate two batman functions of the given parameters
    
    Parameters
    ----------
    x: array-like
        The x-axis on which to generate the function
    mu1: float
        The x-position of the first peak center
    sigma1: float
        The stanfard deviation of the two peak distributions
    A1: float
        The amplitude of the two peaks
    sigma2: float
        The stanfard deviation of the center peak
    A2: float
        The amplitude of the center peak
    sep: float
        The separation between the two peaks

    Returns
    -------
    np.ndarray
        The y-axis values of the mixed Gaussian
    """
    batman1 = batman(x, mu1, sigma1, A1, sigma2, A2, sep)
    batman2 = batman(x, mu3, sigma3, A3, sigma4, A4, sep)
    
    return batman1+batman2

def batman(x, mu1, sigma1, A1, sigma2, A2, sep):
    """
    Generate a batman function of the given parameters
    
    Parameters
    ----------
    x: array-like
        The x-axis on which to generate the function
    mu1: float
        The x-position of the first peak center
    sigma1: float
        The stanfard deviation of the two peak distributions
    A1: float
        The amplitude of the two peaks
    sigma2: float
        The stanfard deviation of the center peak
    A2: float
        The amplitude of the center peak
    sep: float
        The separation between the two peaks

    Returns
    -------
    np.ndarray
        The y-axis values of the mixed Gaussian
    """
    peak1 = gaussian(x,mu1,sigma1,A1)
    peak2 = gaussian(x,mu1+(sep/2.),sigma2,A2)
    peak3 = gaussian(x,mu1+sep,sigma1,A1)
    
    return peak1+peak2+peak3

def composite_spectrum(spectra, plot=False):
    """
    Create a composite spectrum from the given spectra
    
    Parameters
    ==========
    spectra: list
        The list of spectra in increasing order
    plot: bool
        Plot the composite specrum
    
    Returns
    -------
    np.ndarray
        The composite spectrum
    """
    n = len(spectra)
    composite = spectra.pop(0)
    
    for spec in spectra:
        
        # Trim zeros from ends
        component = np.array([i[np.logical_and(spec[0]>0,spec[1]>0)] for i in spec])
        
        # Normalize the component to the composite
        component = norm_spec(spec, composite)
        
        # Trim the composite so the lower wavelength, higher resolution component can be appended
        composite = np.array([i[composite[0]>max(component[0])] for i in composite])
        
        # Append the component
        composite = np.concatenate([component,composite], axis=1)
    
    if plot:
        # Plot the composite spectrum
        plt.figure(figsize=(20,5))
        plt.title('{} Spectra Composite'.format(n))
        plt.step(*composite)
        plt.xlim(0.65, 2.9)
        plt.yscale('log')
        plt.xlabel('Wavelength (um)')
        plt.ylabel('Flux (erg s-1 cm-2)')
    
    return composite
    
def contour_trace(data2D, threshold=15, manual_mask='', padding=4, smoothed=False, plot=False):
    """
    Find the contour at the given threshold to create a signal mask
    
    Parameters
    ==========
    data2D: array-like
        The 2D data to search for a contour       
    threshold: float
        The threshold at which to draw the contour
    manual_mask: array-like
        A 2D boolean array of additional pixels to mask
    padding: int
        The number of pixels to expand the trace by
    smoothed: bool
        Find smoother trace bounds by fitting polynomials 
        the the (lower,upper) contours, e.g. (4,5)
    plot: bool
        Plot the masked 2D data
        
    Returns
    =======
    np.ndarray
        The given 2D data with non-signal pixels masked
    
    """    
    # Define an additional mask
    if isinstance(manual_mask, str):
        mmask = np.zeros(data2D.shape, dtype=bool)
    else:
        mmask = manual_mask
        
    # Add borders for contour fitting
    frame = np.zeros([d+4 for d in data2D.shape])
    frame_mask = np.zeros([d+4 for d in data2D.shape])
    frame[2:-2,2:-2] = data2D.copy()
    frame_mask[2:-2,2:-2] = np.invert(mmask)
    fs, ds = frame.shape, data2D.shape
    
    # Apply the manual mask
    frame *= frame_mask
    
    # Find the contours
    contours = measure.find_contours(frame, threshold)

    if contours:
        # Get the largest connected area (i.e. the trace)
        contour = sorted(contours, key=lambda x: len(x))[-1].T

        # Convert the contour into a closed path
        closed_path = matplotlib.path.Path(contour.T)

        # Get the points that lie within the closed path
        idx = np.array([[(i,j) for i in range(fs[0])] for j in range(fs[1])])
        idx = idx.swapaxes(0,1).reshape(np.prod(fs),2)
        mask = closed_path.contains_points(idx).reshape(fs)

        # Invert the mask and apply to the image
        mask = np.invert(mask)
        masked_data = ma.array(frame.copy(), mask=mask)

        # Do another iteration of pixel selection with smoother contours
        if smoothed:

            # Get the positions of the signal edges
            pos = np.array(ma.notmasked_contiguous(masked_data.copy(), axis=1))

            # Get polynomial lines of top and bottom trace bounds
            x = np.array([n for n,p in enumerate(pos) if p])
            bot = np.array([p[0].start-padding for p in pos if p])
            top = np.array([p[0].stop+padding for p in pos if p])
            deg = smoothed if isinstance(smoothed, (tuple,list)) else [4,5]
            b = polynomial(x, np.polyfit(x, bot, deg[0])[::-1])
            t = polynomial(x, np.polyfit(x, top, deg[-1])[::-1])

            # Make the polynomial evaluations into a contour
            x = np.concatenate([x,x[::-1]])
            c = np.concatenate([t,b[::-1]])
            contour = np.asarray([x,c])

            # Turn the contour into a path, then an array mask, AGAIN!
            closed_path = matplotlib.path.Path(contour.T)
            mask = closed_path.contains_points(idx).reshape(fs)

            # Invert the mask and apply to the image
            mask = np.invert(mask)
            masked_data = ma.array(frame.copy(), mask=mask)
    else:
        masked_data = frame
        print("Could not locate contours at a threshold of {}{}.".format(threshold,\
               '' if isinstance(manual_mask, str) else ' with the given mask'))
    
    # Plot it
    if plot:
        plt.figure(figsize=(20,5))
        plt.imshow(data2D, origin='lower', norm=LogNorm())
        X, Y = plt.xlim(), plt.ylim()
        if contours:
            plt.step(contour[1], contour[0], c='b')
        plt.xlim(X), plt.ylim(Y)
    
    # Remove frame used for contour fitting
    masked_data = masked_data[2:-2,2:-2]
    
    return masked_data

def extract_flux(wavelength, order, masked_trace, bkg='', wavelength_bins='', processes=4, plot=True):
    """
    Extract the full spectrum for a given order
    
    Parameters
    ----------
    wavelength: array-like
        The wavelength array to map to, in microns
    order: int
        The order to extract, [1, 2]
    masked_trace: np.ma.array
        The SOSS frame with the masked trace
    bkg: array-like
        The background counts to subtract
    wavelength_bins: str
        The path to the .npz file of saved wavelength bins
    processes: int
        The number of CPUs to use
    
    Returns
    -------
    np.array
        The array of flux values at the input wavelength values
    
    """
    flx = np.zeros(len(wavelength))
    d_lam = np.mean(np.diff(wavelength))
    filename = 'soss_{}_wavelength_bins_{}um.npy'.format(order,d_lam)
    
    # 1: Check for manual input wavelength bins
    if wavelength_bins:
        
        try:
            wavelength_bins = np.load(wavelength_bins)
        except:
            print('Could not read file',wavelength_bins)
            wavelength_bins = ''
            
    # 2: If no manual input, check to see if the bins are saved
    if wavelength_bins=='':
        
        try:
            wavelength_bins = np.load(filename)
            print('Using wavelength bins stored in',filename)
        except IOError:
            wavelength_bins = ''
            
    # 3: Calculate bins if not saved
    if wavelength_bins=='':
        
        # Get binned flux at each wavelength
        # using a pool for multiple processes
        start = time.time()
        pool = multiprocessing.Pool(processes)
        func = partial(fetch_pixels, d_lam=d_lam, order=order, masked_trace=masked_trace)
        wavelength_bins = pool.map(func, wavelength)
        pool.close()
        pool.join()
        print('Run time in seconds: ', time.time()-start)
        
        # Save the mask to speed up caluclations
        wavelength_bins = np.asarray(wavelength_bins)
        np.save(filename, wavelength_bins)
    
    # Get the counts per wavelength bin
    adu = np.array([np.ma.sum(np.ma.array(masked_trace.data.T, mask=bin)) for bin in wavelength_bins])
    n_pix = np.array([np.size(msk)-np.count_nonzero(msk) for msk in wavelength_bins])
    
    # Subtract mean background ADU of column from each pixel
    if bkg!='':
        adu -= bkg
        
    # Scale each wavelength bin ADU to flux
    flux = adu*np.interp(wavelength, *ADUtoFlux(order))/n_pix
    
    # Plot and compare with the simple column sum
    if plot:
        plt.figure(figsize=(13,4))
        col_wave = np.mean(wave_solutions('256')[order-1], axis=0)
        col_pix = wavelength_bins[0].shape[0]-np.count_nonzero(masked_trace.mask, axis=1)
        col_flux = (np.sum(masked_trace, axis=1))*np.interp(col_wave, *ADUtoFlux(order))/col_pix
        plt.plot(col_wave, col_flux, label='Column Sum')
        plt.plot(wavelength, flux, label='Optimal Extraction')
        plt.xlabel('Wavelength [um]')
        plt.ylabel('Flux Density [erg/s/cm2/A]')
        plt.legend(loc=0, frameon=False)
        print('{} wavelength bins with width {}um for order {}'.format(len(wavelength),d_lam,order))
    
    return flux

def function_trace(data2D, plot=True, start=(4,150), xdisp=20, smooth=3, **kwargs):
    """
    Return the given data with non-trace pixels masked
    
    Parameters
    ==========
    data2D: array-like
        The 2D data to search for a trace
    plot: bool
        Plot the masked 2D data
        
    Returns
    =======
    np.ndarray
        The given 2D data with non-signal pixels masked
        
    """
    # Make a masked array of the 2D data
    masked_data = ma.array(data2D.copy(), mask=False)
    detection = False
    
    # Replace each column mask with the one calculated from isolate_signal()
    for n,col in enumerate(masked_data):
        
        try:
            # Get the list of signal pixels
            col_mask = isolate_signal(col, **kwargs)
            
            # Update the 2D mask with the fit or the previous column fit
            if len(col_mask)<20 or len(col_mask)>len(masked_data.mask[n-1])+3:
                masked_data.mask[n] = masked_data.mask[n-1]
            else:
                masked_data.mask[n] = col_mask.mask
            
            # If the signal is detected, fill in subsequent non-detctions
            detection = True
            
        # Use the PSF bounds of the previous column if isolate_signal times out
        except:
            
            # If a detection has been made, use it
            if detection:
                masked_data.mask[n] = masked_data.mask[n-1]
                
            # Otherwise, skip columns until a detection is made
            else:
                pass
                
    if not detection:
        print('Signal could not be detected.')
        return
        
    # Create new trace mask
    if smooth:
        smooth_masked_data = smooth_trace(masked_data, smooth, xdisp, start[0])
    else:
        smooth_masked_data = masked_data
    
    # Plot it
    if plot:
        plt.figure(figsize=(13,2))
        plt.imshow(masked_data.data.T, origin='lower', norm=LogNorm())
        plt.imshow(smooth_masked_data.T, origin='lower', norm=LogNorm(), cmap=plt.cm.Blues_r, alpha=0.7)
        plt.xlim(0,2048)
        plt.ylim(0,256)
    
    return smooth_masked_data.T

def function_traces(data2D, start1=(4,75), start2=(520,145), xdisp=20, plot=True, offset=0.5, **kwargs):
    """
    Return the given data with non-trace pixels masked
    
    Parameters
    ==========
    data2D: array-like
        The 2D data to search for a trace
    start1: tuple
        The (row,column) to start looking for order 1
    start2: tuple
        The (row,column) to start looking for order 2
    xdisp: int
        The radius of the psf in the cross-dispersion direction
    plot: bool
        Plot the masked 2D data
        
    Returns
    =======
    np.ndarray
        The given 2D data with non-signal pixels masked
        
    """
    # Make a masked array of the 2D data
    masked_data1 = ma.array(data2D.copy(), mask=False)
    masked_data2 = ma.array(data2D.copy(), mask=False)
    
    # Set initial constraints
    llim1 = start1[1]-xdisp-5
    ulim1 = start1[1]+xdisp+5
    llim2 = start2[1]-xdisp-5
    ulim2 = start2[1]+xdisp+5
        
    # Replace each column mask with the one calculated from isolate_signal,
    # using the mask of the previous column as curve_fit constraints
    for n,col in enumerate(masked_data1):
            
        # Update constraints
        if n>0:
            
            # For order 1
            try:
                if llim1<5:
                    llim1 = 5
                    ulim1 = 5+xdisp*2
                elif np.where(~masked_data1.mask[n-1])[0][0]<llim1:
                    llim1 -= offset
                    ulim1 -= offset
                else:
                    llim1 += offset
                    ulim1 += offset
                    
            except:
                pass
                
            # For order 2
            try:
                
                if np.where(~masked_data2.mask[n-1])[0][0]<llim2:
                    llim2 -= offset
                    ulim2 -= offset
                else:
                    llim2 += offset
                    ulim2 += offset
                    
            except:
                pass
                
        # Only extract order 1
        if n>start1[0] and n<start2[0]:
            
            try:
                bounds = ([llim1,1,5,1,5,3],[ulim1,5,2500,10,2500,30])
                col_mask1 = isolate_signal(col, func=batman, xdisp=xdisp, bounds=bounds)
            except:
                col_mask1 = ma.array(col, mask=masked_data1.mask[n-1])
            
            col_mask2 = ma.array(col, mask=True)
            
        # Extract both orders
        elif n>start1[0] and n>start2[0]:
            
            try:
                bounds = ([llim1,2,5,2,5,3,llim2,2,3,2,3],[ulim1,5,2500,10,2500,30,ulim2,4,10,8,10])
                col_mask1, col_mask2 = isolate_orders(col, bounds=bounds, xdisp=xdisp)
            except:
                col_mask1 = ma.array(col, mask=masked_data1.mask[n-1])
                col_mask2 = ma.array(col, mask=masked_data2.mask[n-1])
                
        # Don't extract either order
        else:
            col_mask1 = ma.array(col, mask=True)
            col_mask2 = ma.array(col, mask=True)
            
        # Set the mask in the image
        masked_data1.mask[n] = col_mask1.mask
        masked_data2.mask[n] = col_mask2.mask
            
    # Smooth the trace
    smooth_masked_data1 = smooth_trace(masked_data1, 4, xdisp, start1[0])
    try:
        smooth_masked_data2 = smooth_trace(masked_data2, 3, xdisp, start2[0])
    except:
        smooth_masked_data2 = ''
    
    # Plot it
    if plot:
        plt.figure(figsize=(13,2))
        plt.imshow(masked_data1.data.T, origin='lower', norm=LogNorm())
        plt.imshow(smooth_masked_data1.T, origin='lower', norm=LogNorm(), cmap=plt.cm.Blues_r, alpha=0.7)
        try:
            plt.imshow(smooth_masked_data2.T, origin='lower', norm=LogNorm(), cmap=plt.cm.Reds_r, alpha=0.7)
        except:
            pass
        plt.xlim(0,2048)
        plt.ylim(0,256)
        
    return smooth_masked_data1, smooth_masked_data2

def smooth_trace(masked_data, order, width, start, plot=False):
    """
    Smooth a trace by fitting a polynomial to the unmasked pixels
    
    Parameters
    ----------
    masked_data: numpy.ma.array
        The masked data to smooth
    order: int
        The polynomial order to fit to the unmasked data
    width: int
        The desired radius of the trace from the best fit line
    start: int
        The index of the first pixel to fit
    plot: bool
        Plot the original image with the smoothed fits
    
    Returns
    -------
    numpy.ma.array
        The smoothed masked array 
    """
    # Make a scatter plot where the pixels in each column are offset by a small amount
    x, y = [], []
    for n,col in enumerate(masked_data):
        vals = np.where(~col.mask)
        if vals:
            v = list(vals[0])
            y += v
            x += list(np.random.normal(n, 0.01, size=len(v)))
            
    # Now fit a polynomial to it!
    height, length = masked_data.shape[-1], masked_data.shape[-2]
    Y = np.polyfit(x[start:], y[start:], order)
    X = np.arange(start, length, 1)
    
    # Get pixel values of trace and top and bottom bounds
    trace = np.floor(np.polyval(Y, X)).round().astype(int)
    bottom = trace-width
    top = trace+width
    
    # Plot it
    if plot:
        plt.imshow(masked_data.data, origin='lower', norm=LogNorm())
        plt.plot(X, trace, color='b')
        plt.fill_between(X, bottom, top, color='b', alpha=0.3)
        plt.xlim(0,length)
        plt.ylim(0,height)
        
    # Create smoothed mask from top and bottom polynomials
    smoothed_data = ma.array(masked_data.data, mask=True)
    for n,(b,t) in enumerate(zip(bottom,top)):
        smoothed_data.mask[n+start][b:t] = False
    
    return smoothed_data

def gaussian(x, mu, sigma, A):
    """
    Generate a Gaussian function of the given parameters
    
    Parameters
    ----------
    x: array-like
        The x-axis on which to generate the Gaussian
    mu: float
        The x-position of the peak center
    sigma: float
        The stanfard deviation of the distribution
    A: float
        The amplitude of the peak
        
    Returns
    -------
    np.ndarray
        The y-axis values of the Gaussian
    """
    G = A*np.exp(-(x-mu)**2/2/sigma**2)
    
    return G

def get_TSO_spectra(data, plot=True, ext='primary', **kwargs):
    """
    Extract time-series 1D spectra for a given data_cube
    
    Parameters
    ==========
    data: str, array-like
        The path to the FITS file of the data cube
    plot: bool
        Plot the extracted spectra
    ext: str
        The name of the FITS extension with the data
        
    Returns
    =======
    np.ndarray
        The data cube
    """
    # Get the data from the FITS file
    HDU = fits.open(data)
    data = HDU[ext].data
    
    # Compose a median image from the stack
    median = np.median(data, axis=0)
    
    # Get a trace mask by fitting a bimodal gaussian 
    # to each column in the median image
    params = {'func':bimodal, 'bounds':([15,0,15]*2,[110,5,np.inf]*2)}
    median_spectrum, masked_data = get_spectrum(median, return_mask=True, **kwargs)
    mask = masked_data.mask
    
    # Apply the median trace mask to each image to mask the whole cube
    masked_cube = ma.array(data, mask=np.array([masked_data.mask]*len(data)))
    
    # Sum the columns of each image
    TSO_spectra = np.sum(masked_cube, axis=2)
    
    # Plot the spectra
    if plot:
        for spec in TSO_spectra:
            plt.figure(figsize=(20,5))
            plt.step(median_spectrum[0], spec)
    
    return TSO_spectra

def get_spectrum(data2D, ord=1, wavelength_solution=WS, method='contour', return_mask=False, plot=True, bins=2048, counts=False, **kwargs):
    """
    Extract a 1D spectrum from a raw image
    
    Parameters
    ----------
    data2D: str, array-like
        The 2 dimensional array (image) containing the spectral 
        trace or the path to the FITS file
    ord: int
        The SOSS spectrum order to consider, i.e. 1, 2, or 3
    wavelength_solution: dict (optional)
        A dictionary of the wavelength solution coefficients
    method: str
        The method used to isolate the trace, must be
        'function' for function fits to each column,
        'contour' for contour search at given threshold,
        or a 2D boolean mask
    return_mask: bool
        Return the trace mask as well as the spectrum
    plot: bool
        Plot the results
    bins: int
        The number of wavelength bins to use
    counts: bool
        Return counts instead of flux units
    
    Returns
    -------
    np.ndarray
        The 2D array containing the wavelength and counts/fllux of the extracted spectrum
    """
    # Open the file if necessary
    if isinstance(data2D, str):
        HDU = fits.open(data2D)
        data2D = HDU[1].data
        HDU.close()
        
        data2D = data2D.squeeze()
    
    # Determine array size
    subarr = '96' if data2D.shape[0]==96 \
             else '256' if data2D.shape[0]==256 \
             else 'full'
    
    # Fit a function to the signal in each column
    if method=='function':
        masked_data = function_trace(data2D, plot=plot, **kwargs)
    
    # Draw a contour around the trace
    elif method=='contour':
        masked_data = contour_trace(data2D, plot=plot, **kwargs)
    
    # Use the given mask
    elif isinstance(method, np.ndarray):
        masked_data = ma.array(data2D, mask=method)
        
        if plot:
            plt.figure(figsize=(13,2))
            plt.imshow(masked_data, origin='lower', norm=LogNorm())
    
    # Do a simple sum of the columns
    else:
        masked_data = data2D
        
    # Reverse the x axis
    masked_data = masked_data[:,::-1]
    
    # Get the appropriate wavelength map
    wav = wave_solutions(subarr)[ord-1].flatten()
    
    # Flatten 2D wavelength and count arrays 
    dat = masked_data.copy().flatten()[::-1]
    wav, dat = wav[wav>0], dat[wav>0]
        
    # Create an evenly spaced wavelength axis
    mn, mx = np.nanmin(wav), np.nanmax(wav)
    wavelength = np.linspace(mn, mx, bins)
    spectrum = np.zeros(len(wavelength))
    delta = np.mean(np.diff(wavelength))/2.
    
    # Add up the counts in each wavelength bin
    for n,x in enumerate(wavelength):
        
        # Get all the pixels for this wavelength bin
        in_bin = dat[(wav>=x-delta)&(wav<x+delta)]
        
        # Replace bad pixels with nans
        bad = np.where(np.logical_or(in_bin<0,in_bin>65500))
        in_bin[bad] = np.nan
        
        # Update the bin total
        spectrum[n] = np.nansum([i for i in in_bin if i>0 and i<65500])
        
        # Divide by the number of pixels to normalize
        spectrum[n] *= 1/in_bin.count()
    
    scaling = 1.    
    if not counts:
        # Get the wavelength dependent relative scaling
        scaling = np.interp(wavelength, *ADUtoFlux(ord))
    
    # Convert the spectrum from ADU/s to erg/s/cm2
    spectrum *= scaling
    
    if plot:
        # Plot the spectrum
        plt.figure(figsize=(13,5))
        plt.plot(wavelength, spectrum, label='Signal')
        
        # Format it
        plt.xlim(min(wavelength),max(wavelength))
        plt.ylabel('Counts' if counts else 'Flux (erg s-1 cm-2)')
        plt.xlabel('Wavelength (um)')
        plt.legend()
        
    if return_mask:
        return np.array([wavelength,spectrum]), masked_data
    else:
        return np.array([wavelength,spectrum])

def isolate_orders(col, err=None, xdisp=16, bounds=([5,2,5,2,5,3,50,2,3,2,3],[110,5,2500,10,2500,30,125,4,10,8,10]), sigma=3, plot=False):
    """
    Fit a mixed gaussian function to the signal in an array of data. 
    Identify all pixels within n-sigma as signal.
    
    Parameters
    ----------
    col: array-like
        The 1D data to search for a signal
    err: array-like (optional)
        The errors in the 1D data
    func: function
        The function to fit to the data
    bounds: tuple
        A sequence of length-n (lower,upper) bounds on the n-parameters of func
    sigma: float
        The number of standard deviations to use when defining the signal
    plot: bool
        Plot the signal with the fit function
    
    Returns
    -------
    np.ndarray
        The values of signal pixels and the upper and lower bounds on the fit function
    """        
    # Fit function to signal
    col = ma.array(col, mask=False)
    x = ma.array(np.arange(len(col)))
    params, cov = curve_fit(batmen, x, col, bounds=bounds, sigma=err)
    
    # Reduce to mixed gaussians
    p1 = params[:3]
    p2 = [params[0]+(params[5]/2.), params[3], params[4]]
    p3 = [params[0]+params[5], params[1], params[2]]
    p4 = params[6:9]
    p5 = [params[6]+(params[5]/2.), params[9], params[10]]
    p6 = [params[6]+params[5], params[7], params[8]]
    
    # Get the mu, sigma, and amplitude of each gaussian 'p' in each order
    order1 = np.array([p1,p2,p3])
    order2 = np.array([p4,p5,p6])
    
    params1 = np.array(sorted(order1, key=lambda x: x[0]))
    params2 = np.array(sorted(order2, key=lambda x: x[0]))
    
    # Get the limits of the fit parameters
    llim1 = params1[0][0]-params1[0][1]*sigma
    ulim1 = params1[-1][0]+params1[-1][1]*sigma
    
    llim2 = params2[0][0]-params2[0][1]*sigma
    ulim2 = params2[-1][0]+params2[-1][1]*sigma
    
    cen1 = params1[1][0]
    cen2 = params2[1][0]
    
    # If xdisp is given use a set number of pixels as the radius,
    # otherwise, use the sigma value
    if xdisp:
        llim1 = cen1-xdisp
        ulim1 = cen1+xdisp
        llim2 = cen2-xdisp
        ulim2 = cen2+xdisp
    
    if plot:
        if plot!='overplot':
            plt.figure(figsize=(13,4))
            
        # The data
        plt.step(x, col, c='k', alpha=0.2 if plot=='overplot' else 1, where='mid', label='data')
        
        # The fit function
        plt.plot(x, batmen(x, *params), c='r', lw=2, label='fit')
        if isinstance(err, np.ndarray):
            plt.step(x, err, c='r', alpha=0.2 if plot=='overplot' else 1, where='mid')
            
        # The component functions
        try:
            for g in [p1,p2,p3,p4,p5,p6]:
                plt.plot(x, gaussian(x, *g), ls='--', label=', '.join(['{:.2f}'.format(i) for i in g]))
        except:
            pass
            
        # The parameter limits
        plt.legend()
        plt.axvline(x=llim1, color='b')
        plt.axvline(x=ulim1, color='b')
        plt.axvline(x=llim2, color='m')
        plt.axvline(x=ulim2, color='m')
        
        plt.xlim(min(x),max(x))
        plt.ylim(0,np.nanmax(col)*1.2)
        
    # Return column with background pixels masked
    ord1 = col.copy()
    ord1.mask[(x>llim1)&(x<ulim1)] = True
    ord1.mask = np.invert(ord1.mask)
    
    ord2 = col.copy()
    ord2.mask[(x>llim2)&(x<ulim2)] = True
    ord2.mask = np.invert(ord2.mask)
    
    return ord1, ord2

def isolate_signal(col, err=None, func=bimodal, xdisp=16, bounds=([20,0,0]*2,[200,50,np.inf]*2), sigma=3, plot=False):
    """
    Fit a mixed gaussian function to the signal in an array of data. 
    Identify all pixels within n-sigma as signal.
    
    Parameters
    ----------
    col: array-like
        The 1D data to search for a signal
    err: array-like (optional)
        The errors in the 1D data
    func: function
        The function to fit to the data
    bounds: tuple
        A sequence of length-n (lower,upper) bounds on the n-parameters of func
    sigma: float
        The number of standard deviations to use when defining the signal
    plot: bool
        Plot the signal with the fit function
    
    Returns
    -------
    np.ndarray
        The values of signal pixels and the upper and lower bounds on the fit function
    """        
    # Fit function to signal
    col = ma.array(col, mask=False)
    x = ma.array(np.arange(len(col)))
    params, cov = curve_fit(func, x, col, bounds=bounds, sigma=err)

    # Reduce to mixed gaussians
    if func==batman:
        p1 = params[:3]
        p2 = [params[0]+(params[5]/2.), params[3], params[4]]
        p3 = [params[0]+params[5], params[1], params[2]]
        P = np.array([p1,p2,p3])
    elif func==batmen:
        p1 = params[:3]
        p2 = [params[0]+(params[5]/2.), params[3], params[4]]
        p3 = [params[0]+params[5], params[1], params[2]]
        p4 = params[6:9]
        p5 = [params[6]+(params[5]/2.), params[9], params[10]]
        p6 = [params[6]+params[5], params[7], params[8]]
        P = np.array([p1,p2,p3,p4,p5,p6])
        
    else:
        P = np.array(sorted(params.reshape(len(params)//3,3), key=lambda x: x[0]))

    # Get the limits of the fit parameters
    if xdisp:
        llim = P[1][0]-xdisp
        ulim = P[1][0]+xdisp
    else:
        llim = P[0][0]-P[0][1]*sigma
        ulim = P[-1][0]+P[-1][1]*sigma
    A = max(P.T[-1])
    
    if plot:
        if plot!='overplot':
            plt.figure(figsize=(13,4))
            
        # The data
        plt.step(x, col, c='k', alpha=0.2 if plot=='overplot' else 1, where='mid', label='data')
        
        # The fit function
        plt.plot(x, func(x, *params), c='r', lw=2, label='fit')
        if isinstance(err, np.ndarray):
            plt.step(x, err, c='r', alpha=0.2 if plot=='overplot' else 1, where='mid')
        
        # The component functions
        try:
            for g in P:
                plt.plot(x, gaussian(x, *g), ls='--', label=g)
        except:
            pass
        
        # The parameter limits
        plt.legend()
        plt.axvline(x=llim)
        plt.axvline(x=ulim)
        plt.xlim(min(x),max(x))
        plt.ylim(0,np.nanmax(col)*1.2)

    # Return column with background pixels masked
    col.mask[(x>llim)&(x<ulim)] = True
    col.mask = np.invert(col.mask)
    
    return col

def norm_spec(spectrum, template, exclude=[]):
    """
    Normalize a spectrum to a template spectrum using the wavelength range of overlap
    
    Parameters
    ==========
    spectrum: array-like
        The spectrum to be normalized
    template: array-like
        The spectrum to normalize to
    exclude: list
        The wavelength ranges to ignore in the normalization
        
    Returns
    =======
    np.ndarray
        The normalized spectrum
    """
    normed_spectrum = spectrum.copy()
    
    # Smooth both spectrum and template
    #template[1], spectrum[1] = [u.smooth(x, 1) for x in [template[1], spectrum[1]]]
    
    # Find wavelength range of overlap for array masking
    spec_mask = np.logical_and(spectrum[0] > template[0][0], spectrum[0] < template[0][-1])
    temp_mask = np.logical_and(template[0] > spectrum[0][0], template[0] < spectrum[0][-1])
    spectrum, template = [i[spec_mask] for i in spectrum], [i[temp_mask] for i in template]
    
    # Also mask arrays in wavelength ranges specified in *exclude*
    for r in exclude:
        spec_mask = np.logical_and(spectrum[0] > r[0], spectrum[0] < r[-1])
        temp_mask = np.logical_and(template[0] > r[0], template[0] < r[-1])
        spectrum, template = [i[~spec_mask] for i in spectrum], [i[~temp_mask] for i in template]

    # Normalize the spectrum to the template based on equal integrated flux inincluded wavelength ranges
    norm = np.trapz(template[1], x=template[0]) / np.trapz(spectrum[1], x=spectrum[0])
    normed_spectrum[1:] = [i * norm for i in normed_spectrum[1:]]
    
    return np.array(normed_spectrum)

def pixels2wavelengths(p, coeffs, noversample=1., units=q.um):
    """
    Converts the given pixels to microns given a wavelength solution
    
    Parameters
    ----------
    p: float, array-like
        The pixel position(s)
    coeffs: numpy.ndarray
        The coefficients for the wavelength solution of the given trace
    oversample: int
        Not sure.
    units: astropy.units
        The unit of length to use
    
    Returns
    -------
    float
        The wavelength value of the given pixel position
    """    
    # Get the true pixel value
    pix = p/noversample

    # Calculate the wavelength value in microns and convert to Angstroms
    wavelength = q.um.to(units)*np.array(polynomial(pix, coeffs))
    
    return wavelength

def polynomial(values, coeffs, plot=False, color='g', ls='-', lw=2):
    '''
    Evaluates *values* given the list of ascending polynomial *coeffs*.

    Parameters
    ----------
    values: int, array-like
      The value or values to to evaluated in ascending order, e.g. c0, c1, c2, ...
    coeffs: list, tuple, array
      The sequence of ascending polynomial coefficients beginning with zeroth order
    plot: bool (optional)
      Plot the results in the given color
    color: str
      The color of the line or fill color of the point to plot
    ls: str
      The linestyle of the line to draw
    lw: int
      The linewidth of the line to draw

    Returns
    -------
    float, list
      The evaluated result(s)

    '''

    def poly_eval(val):
        return sum([c * (val ** o) for o, c in enumerate(coeffs)])

    if isinstance(coeffs, dict):
        coeffs = [coeffs[j] for j in sorted([i for i in coeffs.keys() if i.startswith('c')])]

    if isinstance(values, (int, float)):
        out = poly_eval(values)
        if plot:
            plt.errorbar([values], [out], marker='*', markersize=18, color=color, markeredgecolor='k',
                         markeredgewidth=2, zorder=10)

    elif isinstance(values, (tuple, list, np.ndarray)):
        out = [poly_eval(v) for v in values]
        if plot:
            plt.plot(values, out, color=color, lw=lw, ls=ls)

    else:
        out = None; print("Input values must be an integer, float, or sequence of integers or floats!")

    return out

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def smooth_signal(data, err='', sigma=1.5, threshold=8):
    """
    Smooth over cosmic rays in a signal
    
    Parameters
    ----------
    data: array-like
        The 2D data to smooth over
    err: array-like
        The 2D errors to calculate sigma
    sigma: float
        The sigma value to divide by
    threshold: float
        The cutoff value for bad pixels
    
    Returns
    -------
    np.ndarray
        The smoothed data
    """
    # Smooth the signal for cosmic rays
    smoothed = ss.medfilt(data, 5)
    
    # Set the threshold for which pixels to smooth
    if isinstance(err,np.ndarray):
        sigma = np.median(err)
    bad = np.abs(data-smoothed)/sigma > threshold
    
    # Replace the bad pixels with smoothed values
    data[bad] = smoothed[bad]
    
    # Return the smoothed spectrum
    return data

def tilt_correction(exposure, wavelength_image, plot=True):
    """
    Transform the exposure to create iso-wavelength detector columns
    for more accurate counts per wavelength
    
    Parameters
    ==========
    exposure: array-like
        The 2D array of data to corect
    wavelength_image: array-like
        The 2D array which maps wavelengths to each pixel
    plot: bool
        Plot the input/output images for comparison
        
    Returns
    =======
    np.ndarray
        The tilt-corrected exposure
    """
    wave = wavelength_image.copy()
    x = 1 if wave[0,0]<wave[0,-1] else -1
    y = 1 if wave[0,0]<wave[-1,0] else -1
    wave = wave[::y,::x]
    
    # Make the new, wider image
    m, n = wave.shape
    j_shift = np.interp(wave[:,0], wave[0,:], np.arange(n))
    pad = int(np.max(j_shift))
    i, j = np.indices((m, n+pad))
    
    # Get the shifted coordinates
    idx = [i, j-j_shift[:,None]]
    corrected_exposure =  map_coordinates(exposure.copy(), idx, cval=np.nan)
    corrected_wave = map_coordinates(wavelength_image.copy(), idx, cval=np.nan)
    
    # Trim off incomplete columns
    trim = int((corrected_exposure.shape[1]-exposure.shape[1]))
    corrected_exposure = corrected_exposure[:,trim:exposure.shape[1]+trim]

    # Plot it
    if plot:
        plt.figure(figsize=(20,10))
        plt.subplot(211)
        plt.imshow(exposure, interpolation='none', origin='lower')
        plt.title('original')
        plt.subplot(212)
        plt.imshow(corrected_exposure, interpolation='none', origin='lower')
        plt.title('fixed')
        plt.tight_layout()
    
    return corrected_exposure.T, corrected_wave.T

def trimodal(x, mu1, sigma1, A1, mu2, sigma2, A2, mu3, sigma3, A3):
    """
    Generate a trimodal function of three Gaussians of the given parameters
    
    Parameters
    ----------
    x: array-like
        The x-axis on which to generate the function
    mu1: float
        The x-position of the first peak center
    sigma1: float
        The stanfard deviation of the first distribution
    A1: float
        The amplitude of the first peak
    mu2: float
        The x-position of the second peak center
    sigma2: float
        The stanfard deviation of the second distribution
    A2: float
        The amplitude of the second peak
    mu3: float
        The x-position of the third peak center
    sigma3: float
        The stanfard deviation of the third distribution
    A3: float
        The amplitude of the third peak

    Returns
    -------
    np.ndarray
        The y-axis values of the mixed Gaussian
    """
    B = gaussian(x,mu1,sigma1,A1)+gaussian(x,mu2,sigma2,A2)+gaussian(x,mu3,sigma3,A3)
     
    return B

def wave_solutions(subarr, directory=dir_path+'/files/soss_wavelengths_fullframe.fits'):
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

def p2w(p,noversample,ntrace):
    "Usage: w=p2w(p,noversample,ntrace) Converts x-pixel (p) to wavelength (w)"
    
    #co-efficients for polynomial that define the trace-position
    nc=5 #number of co-efficients
    c=[[2.60188,-0.000984839,3.09333e-08,-4.19166e-11,1.66371e-14],\
     [1.30816,-0.000480837,-5.21539e-09,8.11258e-12,5.77072e-16],\
     [0.880545,-0.000311876,8.17443e-11,0.0,0.0]]
    
    pix=p/noversample
    w=c[ntrace-1][0]
    for i in range(1,nc):
        #print(w)
        w+=np.power(pix,i)*c[ntrace-1][i]
                  
    return w
