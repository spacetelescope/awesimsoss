#! /usr/bin/env python
import numpy
import astropy.io.fits as fits
import noise_simulation_special as ng
import os

path=os.getenv('NIRISS_NOISE_HOME')+'/'
if path is None:
  path='./'

def add_dark_current(ramp,seed,gain,darksignal):
  numpy.random.seed(seed)
  dims=ramp.shape
  total=darksignal*0.
  ngroups=ramp.shape[0]
  for n in range(dims[0]):
    signal=numpy.random.poisson(darksignal)/gain
    total=total+signal
    ramp[n,:,:]=ramp[n,:,:]+total
  return ramp
  
def make_exposure(nints,ngroups,darksignal,gain,noise_seed=None,dark_seed=None,offset=None):
  if noise_seed is None:
    noise_seed=7L+long(numpy.random.uniform()*4000000000.)
  if dark_seed is None:
    dark_seed=5L+long(numpy.random.uniform()*4000000000.)
  if offset is None:
    offset=500.
  numpy.random.seed(dark_seed)
  if (nints < 1) | (ngroups < 1):
    return None
  simulated_data=numpy.zeros([nints*ngroups,256,2048],dtype=numpy.float32)
  # Define the HXRGN instance to make a SUSBSTRIP256 array (in detector coordinates)
  noisecube=ng.HXRGNoise(naxis1=256,naxis2=2048,naxis3=ngroups,x0=0,y0=0,det_size=2048,verbose=False)
  pedestal=18.30
  c_pink=9.6
  u_pink=3.2
  acn=2.0
  bias_amp=0.
  #bias_amp=5358.87
  #bias_offset=20944.06
  pca0_amp=0.
  rd_noise=12.95
  dark_current=0.0
  dc_seed=dark_seed
  bias_offset=offset*gain
  for loop in range(nints):
    seed1=noise_seed+24*long(loop)
    ramp=noisecube.mknoise(c_pink=c_pink,u_pink=u_pink,bias_amp=bias_amp,bias_offset=bias_offset,acn=acn,pca0_amp=pca0_amp,rd_noise=rd_noise,pedestal=pedestal,dark_current=dark_current,dc_seed=dc_seed,noise_seed=seed1,gain=gain)
    ramp=numpy.transpose(ramp,(0,2,1))
    ramp=ramp[::,::-1,::-1]
    ramp=add_dark_current(ramp,dc_seed,gain,darksignal)
    simulated_data[loop*ngroups:(loop+1)*ngroups,:,:]=numpy.copy(ramp)
    ramp=0
#  noisecube=noisecube.astype('uint16')
  return simulated_data

def make_photon_yield(photon_yield,orders):
  dims=orders.shape
  sum1=numpy.zeros((dims[1],dims[2]),dtype=numpy.float32)
  sum2=numpy.zeros((dims[1],dims[2]),dtype=numpy.float32)
  for n in range(dims[0]):
    sum1=sum1+photon_yield[n,:,:]*orders[n,:,:]
    sum2=sum2+orders[n,:,:]
  pyimage=sum1/sum2
  pyimage[numpy.where(sum2 == 0.)]=1.
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
      newvalues=numpy.random.poisson(framesignal)
      target=pyimage-1.
      for k in range(dims1[1]):
        for l in range(dims1[2]):
          if target[k,l] > 0.:
            n=int(newvalues[k,l])
            values=numpy.random.poisson(target[k,l],size=n)
            newvalues[k,l]=newvalues[k,l]+numpy.sum(values)
      newvalues=newvalues+numpy.random.poisson(background)
    else:
      newvalues=numpy.random.poisson(framesignal*pyimage+background)
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
    frame=numpy.squeeze(numpy.copy(newcube[k,:,:]))
    sum1=frame*0.
    for n in range(dims1[0]-1,-1,-1):
      sum1=sum1+nonlinearity[n,:,:]*numpy.power(frame,n+1)
    sum1=frame*(1.+sum1)
    newcube[k,:,:]=sum1
  newcube=newcube+offset
  return newcube

def add_pedestal(cube,pedestal,offset):
  ped1=pedestal+(offset-500.)
  dims=cube.shape
  newcube=numpy.zeros_like(cube,dtype=numpy.float32)
  for n in range(dims[0]):
    newcube[n,:,:]=cube[n,:,:]+ped1
  newcube=newcube.astype(numpy.uint16)
  return newcube
