#! /usr/bin/env python
import numpy as np
import astropy.io.fits as fits
from . import noise_simulation as ng
import os

def add_dark_current(ramp, seed, gain, darksignal):
    np.random.seed(seed)
    dims=ramp.shape
    total=darksignal*0.
    ngroups=ramp.shape[0]
    for n in range(dims[0]):
        signal=np.random.poisson(darksignal)/gain
        total=total+signal
        ramp[n,:,:]=ramp[n,:,:]+total
    return ramp
    
def make_exposure(nints, ngrps, darksignal, gain, pca0_file, noise_seed=None, dark_seed=None, offset=500):
    """
    Make a simulated exposure with no source signal
    
    Parameters
    ----------
    nints: int
        The number of integrations
    ngrps: int
        The number of groups per integration
    """
    if nints < 1 or ngrps < 1:
        return None
        
    if not noise_seed:
        noise_seed = 7+int(np.random.uniform()*4000000000.)
        
    if not dark_seed:
        dark_seed = 5+int(np.random.uniform()*4000000000.)
    np.random.seed(dark_seed)
        
    # Make empty data array
    simulated_data = np.zeros([nints*ngrps,256,2048], dtype=np.float32)
    
    # Define some constants
    pedestal = 18.30
    c_pink = 9.6
    u_pink = 3.2
    acn = 2.0
    bias_amp = 0.
    #bias_amp = 5358.87
    #bias_offset = 20944.06
    pca0_amp = 0.
    rd_noise = 12.95
    dark_current = 0.0
    dc_seed = dark_seed
    bias_offset = offset*gain
    
    # Define the HXRGN instance to make a SUSBSTRIP256 array (in detector coordinates)
    noisecube = ng.HXRGNoise(naxis1=256, naxis2=2048, naxis3=ngrps, pca0_file=pca0_file, x0=0, y0=0, det_size=2048, verbose=False)
    
    # iterate over integrations
    for loop in range(nints):
        seed1 = noise_seed+24*int(loop)
        ramp = noisecube.mknoise(c_pink=c_pink, u_pink=u_pink, bias_amp=bias_amp, bias_offset=bias_offset, acn=acn, pca0_amp=pca0_amp, rd_noise=rd_noise, pedestal=pedestal, dark_current=dark_current, dc_seed=dc_seed, noise_seed=seed1, gain=gain)
        ramp = np.transpose(ramp,(0,2,1))
        ramp = ramp[::,::-1,::-1]
        ramp = add_dark_current(ramp,dc_seed,gain,darksignal)
        simulated_data[loop*ngrps:(loop+1)*ngrps,:,:] = np.copy(ramp)
        ramp = 0
        
    # noisecube=noisecube.astype('uint16')
    
    return simulated_data

def make_photon_yield(photon_yield, orders):
    """
    Generates a map of the photon yield for each order. 
    The shape of both arrays should be [order,nrows,ncols]
    
    Parameters
    ----------
    photon_yield: str
        The path to the file containg the calculated photon yield at each pixel
    orders: sequence
        An array of the median image of each order
    """
    # Get the shape and create empty arrays
    dims = orders.shape
    sum1 = np.zeros((dims[1],dims[2]),dtype=np.float32)
    sum2 = np.zeros((dims[1],dims[2]),dtype=np.float32)
    
    # Add the photon yield for each order
    for n in range(dims[0]):
        sum1 = sum1+photon_yield[n,:,:]*orders[n,:,:]
        sum2 = sum2+orders[n,:,:]
    
    # Take the ratio of the photon yield to the signal
    pyimage = sum1/sum2
    pyimage[np.where(sum2 == 0.)] = 1.
    
    return pyimage

def add_signal(signals,cube,pyimage,frametime,gain,zodi,zodi_scale,photon_yield=None):
    if photon_yield is None:
        photon_yield=False
    dims1=cube.shape
    dims2=signals.shape
    if dims1 != dims2:
        raise ValueError
    newcube=cube*0.
    # The background is assumed to be in electrons/second/pixel, not ADU/s/pixel.
    background=zodi*zodi_scale*frametime
    for n in range(dims1[0]):
        framesignal=signals[n,:,:]*gain*frametime
        if photon_yield:
            newvalues=np.random.poisson(framesignal)
            target=pyimage-1.
            for k in range(dims1[1]):
                for l in range(dims1[2]):
                    if target[k,l] > 0.:
                        n=int(newvalues[k,l])
                        values=np.random.poisson(target[k,l],size=n)
                        newvalues[k,l]=newvalues[k,l]+np.sum(values)
            newvalues=newvalues+np.random.poisson(background)
        else:
            newvalues=np.random.poisson(framesignal*pyimage+background)
        if n == 0:
            newcube[n,:,:]=newvalues
        else:
            newcube[n,:,:]=newcube[n-1,:,:]+newvalues
    newcube=cube+newcube/gain
    return newcube

def non_linearity(cube,nonlinearity,offset=None):
    dims1=nonlinearity.shape
    dims2=cube.shape
    if (dims1[1] != dims2[1]) | (dims1[1] != dims2[1]):
        raise ValueError
    if offset is None:
        offset=0.
    newcube=cube-offset
    for k in range(dims2[0]):
        frame=np.squeeze(np.copy(newcube[k,:,:]))
        sum1=frame*0.
        for n in range(dims1[0]-1,-1,-1):
            sum1=sum1+nonlinearity[n,:,:]*np.power(frame,n+1)
        sum1=frame*(1.+sum1)
        newcube[k,:,:]=sum1
    newcube=newcube+offset
    return newcube

def add_pedestal(cube,pedestal,offset):
    ped1=pedestal+(offset-500.)
    dims=cube.shape
    newcube=np.zeros_like(cube,dtype=np.float32)
    for n in range(dims[0]):
        newcube[n,:,:]=cube[n,:,:]+ped1
    newcube=newcube.astype(np.uint16)
    return newcube
