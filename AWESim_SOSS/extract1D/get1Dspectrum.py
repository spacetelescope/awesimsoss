import os
import numpy as np
import astropy.units as q
import matplotlib.pyplot as plt
import matplotlib
import warnings
from numpy import ma
from scipy.optimize import curve_fit
from astropy.io import fits
from matplotlib.colors import LogNorm

warnings.simplefilter('ignore')

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

def isolate_signal(col, err=None, func=batman, xdisp=16, bounds=([1,2,100,3,50,1],[120,5,1500,10,1500,20]), sigma=3, plot=False):
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
