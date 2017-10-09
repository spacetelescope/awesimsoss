def batman_wrapper_lmfit(period, tcenter, inc, aprs, rprs, ecc, omega, u1, u2, 
                         offset, slope, curvature,
                         times, ldtype='quadratic', transitType='primary'):
    '''
    Written By Jonathan Fraine
    https://github.com/exowanderer/Fitting-Exoplanet-Transits
    '''
    bm_params           = batman.TransitParams() # object to store transit parameters
    bm_params.per       = period   # orbital period
    bm_params.t0        = tcenter  # time of inferior conjunction
    bm_params.inc       = inc      # inclunaition in degrees
    bm_params.a         = aprs     # semi-major axis (in units of stellar radii)
    bm_params.rp        = rprs     # planet radius (in units of stellar radii)
    bm_params.ecc       = ecc      # eccentricity
    bm_params.w         = omega    # longitude of periastron (in degrees)
    bm_params.limb_dark = ldtype   # limb darkening model 
    bm_params.u         = [u1, u2] # limb darkening coefficients 

    m_eclipse = batman.TransitModel(bm_params, times, transittype=transitType)# initializes model

    # OoT == Out of transit    
    OoT_curvature = offset + slope*(times-times.mean()) + curvature*(times-times.mean())**2
    
    return m_eclipse.light_curve(bm_params) * OoT_curvature

def default_init_params(planetName, planetParams={}):
    import lmfit
    # planetParams = {'rprs':      (0.147   , True, 0.0, 0.2),
    #                 'period':    (5.721490, False         ),
    #                 'tcenter':   (0., True, -0.1, 0.1     ),
    #                 'inc':       (89.7    , True, 80., 90.),
    #                 'aprs':      (18.2    , True, 15., 20.),
    #                 'ecc':       (0.0     , False         ),
    #                 'omega':     (90.     , False         ),
    #                 'u1':        (0.284   , True, 0., 1.  ),
    #                 'u2':        (0.208   , True, 0., 1.  ),
    #                 'offset':    (1.0     , True, 0.0     ),
    #                 'slope':     (0.0     , False         ),
    #                 'curvature': (0.0     , False         )
    #                }
    
    if not len(planetParams):
        from exoparams import PlanetParams
        params  = PlanetParams(planetName)
    try:
        rprs      = planetParams['rprs'][0]
        rprsFit   = planetParams['rprs'][1]
        if len(planetParams['rprs']) > 2:
            rprsMin   = planetParams['rprs'][2]
            rprsMax   = planetParams['rprs'][3]
    except:
        rprs      = np.sqrt(params.depth)
        rprsFit   = False
    try:
        period      = planetParams['period'][0]
    except:
        period    = params.per
        periodFit = False
    try:
        tcenter      = planetParams['tcenter'][0]
    except:
        tcenter   = params.tt
        tcenterFit= False
    try:
        inc      = planetParams['inc'][0]
    except:
        inc       = params.i
        incFit    = False
    try:
        aprs      = planetParams['aprs'][0]
    except:
        aprs      = params.a
        aprsFit   = False
    try:
        ecc       = planetParams['ecc'][0]
    except:
        ecc       = params.ecc
        eccFit    = False
    try:
        omega      = planetParams['omega'][0]
    except:
        omega     = params.omega
        omegaFit  = False
    try:
        u1        = planetParams['u1'][0]
    except:
        u1        = 0.2
        u1Fit     = False
    try:
        u2        = planetParams['u2'][0]
    except:
        u2        = 0.2
        u2Fit     = False
    try:
        offset    = planetParams['offset'][0]
    except:
        offset    = 1.0
        offsetFit = False
    try:
        slope      = planetParams['slope'][0]
    except:
        slope     = 0.
        slopeFit  = False
    try:
        curvature = planetParams['rpcurvaturers'][0]
    except:
        curvature     = 0.0
        curvatureFit  = False
    
    initialParams = lmfit.Parameters()
    # Format: (key, value, vary?, min, max)
    initialParams.add_many(                     # WASP-107b paramters from Anderson et al (2017)
        ('rprs' , 0.147, True, 0.0, 0.2),
        ('period' , 5.721490, False),
        ('tcenter' , 0., True, -0.1, 0.1),
        ('inc' , 89.7, True, 80., 90.),
        ('aprs' , 18.2, True, 15., 20.),
        ('ecc' , 0., False),
        ('omega' , 90., False),
        ('u1' , 0.284, True, 0., 1.),
        ('u2' , 0.208, True, 0., 1.),
        ('offset', 1.0, True, 0.0),
        ('slope', 0.0, False),
        ('curvature', 0.0, False))
    
    return initialParams

def align_spectra_into_transit_lightcurve(stellar1DSpectra, iIngress=25, iEgress=25):
    # stellar1DSpectra='AWESim_SOSS/wasp107_data/wasp107_extracted_1D_spectra.save'
    
    if isinstance(filename,str):
        stellar1DSpectra = joblib.load(stellar1DSpectra)
    
    spec1D_0  = stellar1DSpectra['counts']
    spec1Derr = stellar1DSpectra['countsErr']
    waves     = stellar1DSpectra['waveslength']
    times     = stellar1DSpectra['times']#np.linspace(-0.114,0.114, spec1D.shape[0])
    
    err     = 3e-3                     # 3000 ppm uncertainty per pointnOOT    = 25
    
    beforeTransit = list(np.arange(iIngress))
    aftertransit  = list(-np.arange(iEgress)-1)
    
    ootInds = np.array(beforeTransit+afterTransit)
    medSpec = np.median(stellar1DSpectra['counts'][ootInds],axis=0)
    spec1D  = spec1D_0 / medSpec
    
    return times, waves, spec1D, spec1Derr

def plot_2D_transmission_spectrum(waves, spec1D, vmin=0.95, vmax=1.05):
    # Plot the time series of 1D spectra
    # This may take a few seconds to complete
    plt.figure(figsize=(12,8))
    for j in range(nint):
        plt.scatter(waves, np.zeros(spec1D.shape[-1])+j, c=spec1D[j], 
                    s=10,linewidths=0,vmin=vmin,vmax=vmax,marker='s',cmap=plt.cm.RdYlBu_r)
                    
    nint = spec1D.shape[0]
    
    plt.xlim(waves[0],waves[-1])
    plt.ylim(0,nint)
    plt.ylabel('Integration Number', size=14)
    plt.xlabel('waveslength ($\mu m$)', size=14)
    a=plt.colorbar()

def bin_flux_vs_wavelenght(wave, specID, wave_low, wave_hi, nchan):
    '''Bin 1D spectra into spectrophotometric channels'''
    
    nint = spec1D.shape[0]
    
    binflux = np.zeros((nchan,nint))
    binerr  = np.zeros((nchan))
    for i in range(nchan):
        index      = np.where((wave >= wave_low[i])*(wave <= wave_hi[i]))[0]
        binflux[i] = np.mean(spec1D[:,index],axis=1)
        binerr[i]  = np.sqrt(len(index))*err

def plot_example_transit_lightcurve(binwave, binflux, results, times, figsize=(8,6)):
    # Plot best fit light curve and residuals for the same channel
    plt.figure(figsize=figsize);
    plt.title(str(binwave[i]) + " microns");
    
    plt.plot(times, binflux[i], 'b.');
    plt.plot(times, results[i].best_fit, 'r-', lw=2);
    
    a=plt.xlim(times.min(), times.max());
    plt.ylabel("Flux",size=14);
    plt.xlabel("Time (Days)",size=14);
    
    plt.figure(figsize=figsize);
    plt.title(str(binwave[i]) + " microns");
    
    plt.plot(times, results[i].residual, 'k.');
    
    a=plt.xlim(times.min(), times.max());
    plt.ylabel("Normalized Residuals",size=14);
    plt.xlabel("Time (Days)",size=14);

def reorganize_transmission_spectrum(nchan, results, ):
    # Record best-fit radii and formal errors
    rprs    = np.zeros(nchan)
    rprserr = np.zeros(nchan)
    for i in range(nchan):
        rprs[i]    = results[i].best_values['rprs']
        try:
            rprserr[i] = np.sqrt(results[i].covar[0,0])       # Assumes RpRs is 0th free parameter
        except:
            rprserr[i] = np.nan       # Assumes RpRs is 0th free parameter
    
    return rprs, rprserr

def plot_transmission_spectrum(binwave, rprs, rprserr, waveMin=None, waveMax=None, figsize=(10,5)):
    # waveMin=wave[0], waveMax=wave[-1]
    
    medDiffWave = np.median(np.diff(binwave))
    if waveMin is None:
        waveMin = binwave.min() - 0.5*medDiffWave
    if waveMax is None:
        waveMax = binwave.max() + 0.5*medDiffWave
    
    # Plot transmission spectrum
    plt.figure(figsize=figsize)
    plt.errorbar(binwave, rprs, rprserr, fmt='b.')
    
    plt.ylabel("Planet-Star Radius Ratio",size=14)
    plt.xlabel("Wavelength ($\mu m$)",size=14)
    
    a=plt.xlim(waveMin, waveMax)
    plt.ylim(0.150, 0.160);
    
def rescale_injected_planetary_spectrum(binwave, planetSpec, minSpec=0.151, maxSpec=0.159):
    # planetSpec=planetSpec
    from scipy.signal import medfilt
    
    waveUse       = (planetSpec[0]>binwave.min())*(planetSpec[0]<binwave.max())
    
    planetSpec_wv = planetSpec[0][waveUse]
    planetSpec_cr = planetSpec[1][waveUse]
    planetSpec_sm = medfilt(planetSpec_cr,101)
    planetSpec_rng= (maxSpec - minSpec)
    planetSpec_wid= (maxSpec + minSpec)
    
    observed_rng  =(planetSpec_sm.max() - np.min(planetSpec_sm))
    planetSpec_rs = (planetSpec_sm - np.min(planetSpec_sm)) / observed_rng * planetSpec_rng + 0.5 * planetSpec_wid
    
    return planetSpec_rs

def save_transmission_spectrum(filename, binwave, rprs, rprserr):
    filename="AWESim_SOSS/example_results/planetSpec-TranSpec.save"
    joblib.dump(dict(binwave=binwave, rprs=rprs, rprserr=rprserr), filename)
