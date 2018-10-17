'''
    NGHXRG - Teledyne HxRG Noise Generator

    First adapted from pynrc.simul.nghxrg 
    
    Modification History:
    
    14 October 2017, J.M. Leisenring, University of Arizona
'''

import pynrc.nghxrg as ng

bias_file = 'SUPER_BIAS_NIRISS.FITS'
dark_file = 'SUPER_DARK_NIRISS.FITS' # This should be IPC corrected

wind_mode  = 'FULL'       # 'FULL', 'STRIPE', or 'WINDOW'
xpix, ypix = (2048, 2048) # Subarray size
x0, y0     = (0, 0)       # Subarray start location

nroh     = 12 # Extra pixels per line
nfoh     = 2 if 'WINDOW' in wind_mode else 1 # Extra lines per frame
nfoh_pix = 0 if 'WINDOW' in wind_mode else 1 # Extra pixels per frame

# Number of output channels
n_out    = 1 if 'WINDOW' in wind_mode else 4

# Number of frames in ramp
naxis3 = 10

# Option to use pyFFTW. I'm not sure this is really useful. You could instead
# just perform multiple instances of nghxrg
use_fftw = False
ncores   = None
# Extensive testing on both Python 2 & 3 shows that 4 cores is optimal for FFTW
# Beyond four cores, the speed improvement is small. Those other processors are
# are better used elsewhere.
if use_fftw and (ncores is None): ncores = 4

verbose = True

# Instantiate a noise generator object
ng_h2rg = ng.HXRGNoise(naxis1=xpix, naxis2=ypix, naxis3=naxis3, 
             n_out=nout, nroh=nroh, nfoh=nfoh, nfoh_pix=nfoh_pix,
             dark_file=dark_file, bias_file=bias_file,
             wind_mode=wind_mode, x0=x0, y0=y0,
             use_fftw=use_fftw, ncores=ncores, verbose=verbose)

# Gain in e/ADU
# ADU are the same thing as DN.
gn  = 1.61

###############################################################
# Everything below here needs to be determined experimentally #
###############################################################

# Noise Values (e)
# The scale factors are used for matching inputs and outputs
ktc_noise = 37.0 * 1.15
rd_noise  = np.array([8.5,8.5,8.5,8.5]) * 0.93
c_pink    = 6.0 * 1.6
u_pink    = np.array([2.0,2.0,2.0,2.0]) * 1.4
ref_rat   = 0.9 # Ratio of reference pixel noise to that of reg pixels

#################################################################
# All of these offsets get taken out after reference correction #
# They're mainly here for cosmetic reasons                      #
#################################################################

# Offset Values
bias_off_avg = gn * 11500 # On average, integrations start here
bias_off_sig = gn * 20    # bias_off_avg has some variation. This is its std dev.
bias_amp     = gn * 1.0   # A multiplicative factor to multiply bias_image. 1.0 for NIRCam.

# Offset of each channel relative to bias_off_avg.
ch_off = gn * np.array([560, 100, -440, -330])
# Random frame-to-frame reference offsets due to PA reset
ref_f2f_corr  = gn * 18.0 * 0.95
ref_f2f_ucorr = gn * np.array([15,15,15,15]) * 1.15 # per-amp
# Relative offsets of alternating columns
aco_a = gn * np.array([-100,600,550,400])
aco_b = -1 * aco_a
#Reference Instability
ref_inst = gn * 1.0

# If only one output (window mode) then select first elements of each array
if nout == 1:
    rd_noise = rd_noise[0]
    u_pink = u_pink[0]
    ch_off = ch_off[0]
    ref_f2f_ucorr = ref_f2f_ucorr[0]
    aco_a = aco_a[0]
    aco_b = aco_b[0]

# Output in ADU or electrons?
# Keep in e- if applying to a ramp observation 
# then convert combined data to ADU later.
out_ADU = True

hdu = ng_h2rg.mknoise(None, gain=gn, rd_noise=rd_noise, c_pink=c_pink, u_pink=u_pink, 
        reference_pixel_noise_ratio=ref_rat, ktc_noise=ktc_noise,
        bias_off_avg=bias_off_avg, bias_off_sig=bias_off_sig, bias_amp=bias_amp,
        ch_off=ch_off, ref_f2f_corr=ref_f2f_corr, ref_f2f_ucorr=ref_f2f_ucorr, 
        aco_a=aco_a, aco_b=aco_b, ref_inst=ref_inst, out_ADU=out_ADU)

hdu.header['UNITS'] = 'ADU' if out_ADU else 'e-'