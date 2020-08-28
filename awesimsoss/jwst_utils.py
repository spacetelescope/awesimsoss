#! /usr/bin/env python
import os
from pkg_resources import resource_filename

from astropy.io import fits

try:
    import crds
except ImportError:
    print("Could not import 'crds' package. Using default reference files which may not be the most up-to-date.")

try:
    from jwst import datamodels as dm
except ImportError:
    print("Could not import 'jwst' package. Functionality will be limited.")

SUB_SLICE = {'SUBSTRIP96': slice(1792, 1888), 'SUBSTRIP256': slice(1792, 2048), 'FULL': slice(0, 2048)}
SUB_DIMS = {'SUBSTRIP96': (96, 2048), 'SUBSTRIP256': (256, 2048), 'FULL': (2048, 2048)}

def get_references(subarray, filter='CLEAR'):
    """
    Get dictionary of the reference file locations for the given subarray

    Parameters
    ----------
    subarray: str
        The subarray to use, ['SUBSTRIP96', 'SUBSTRIP256', 'FULL']
    filter: str
        The filter to use, ['CLEAR', 'F277W']

    Returns
    -------
    dict
        The dictionary of reference files
    """
    # Accepted subarrays
    subarrays = ['SUBSTRIP96', 'SUBSTRIP256', 'FULL']
    if subarray not in subarrays:
        raise ValueError("{} is not a supported subarray. Please use {}".format(subarray, subarrays))

    # Accepted filters
    filters = ['CLEAR', 'F277W']
    if filter not in filters:
        raise ValueError("{} is not a supported filter. Please use {}".format(filter, filters))

    # F277W not yet supported. Just delete this line when F277W support is added to crds
    filter = 'CLEAR'

    params = {"INSTRUME": "NIRISS",
              "READPATT": "NIS",
              "EXP_TYPE": "NIS_SOSS",
              "DETECTOR": "NIS",
              "PUPIL": "GR700XD",
              "DATE-OBS": "2020-07-28",
              "TIME-OBS": "00:00:00",
              "INSTRUMENT": "NIRISS",
              "FILTER" : filter,
              "SUBARRAY": subarray}

    # Collect reference files for subarray+filter combination
    refs = crds.getreferences(params)

    return refs


def jwst_photyield_ref(subarray):
    """
    Function to retrieve photon yield reference file from installed jwst calibration pipeline

    Parameters
    ----------
    subarray: str
        The subarray to use, ['SUBSTRIP96', 'SUBSTRIP256', 'FULL']

    Returns
    -------
    np.ndarray
        The sliced photon yield reference file data
    """
    photyield_file = resource_filename('awesimsoss', 'files/photonyieldfullframe.fits')
    photyield_data = fits.getdata(photyield_file)[:, SUB_SLICE[subarray], :]

    return photyield_data


def jwst_ramp_model(data, groupdq, pixeldq, err, **kwargs):
    """
    Function to retrieve RampModel object from installed jwst calibration pipeline

    Parameters
    ----------
    data: array-like
        The data for the data extension
    groupdg: array-like
        The data for the groupdq extension
    pixeldq: array-like
        The data for the pixeldq extension
    err: array-like
        The data for the err extension

    Returns
    -------
    jwst.datamodels.RampModel
        The populated RampModel object
    """
    model = dm.RampModel(data=data, groupdq=groupdq, pixeldq=pixeldq, err=err, **kwargs)

    return model


def jwst_zodi_ref(subarray):
    """
    Function to retrieve zodiacal background reference file from installed jwst calibration pipeline

    Parameters
    ----------
    subarray: str
        The subarray to use, ['SUBSTRIP96', 'SUBSTRIP256', 'FULL']

    Returns
    -------
    np.ndarray
        The sliced zodiacal background reference file data
    """
    zodi_file = resource_filename('awesimsoss', 'files/background_detectorfield_normalized.fits')
    zodi_data = fits.getdata(zodi_file)[SUB_SLICE[subarray], :]

    return zodi_data
