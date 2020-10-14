#! /usr/bin/env python
from copy import copy
import os
from pkg_resources import resource_filename

from astropy.io import fits
import numpy as np

try:
    from jwst import datamodels as dm
except ImportError:
    print("Could not import 'jwst' package. Functionality will be limited.")

SUB_SLICE = {'SUBSTRIP96': slice(1792, 1888), 'SUBSTRIP256': slice(1792, 2048), 'FULL': slice(0, 2048)}
SUB_DIMS = {'SUBSTRIP96': (96, 2048), 'SUBSTRIP256': (256, 2048), 'FULL': (2048, 2048)}

def add_refpix(data, counts=0):
    """
    Add reference pixels to detector edges

    Parameters
    ----------
    data: np.ndarray
        The data to add reference pixels to
    counts: int
        The number of counts or the reference pixels

    Returns
    -------
    np.ndarray
        The data with reference pixels
    """
    # Get dimensions
    dims = data.shape
    new_data = copy(data)

    # Convert to 3D
    if data.ndim == 4:
        new_data.shape = dims[0] * dims[1], dims[2], dims[3]
    elif data.ndim == 2:
        new_data.shape = (1, dims[0], dims[1])

    # Left, right (all subarrays)
    new_data[:, :, :4] = counts
    new_data[:, :, -4:] = counts

    # Top (excluding SUBSTRIP96)
    if dims[-2] != 96:
        new_data[:, -4:, :] = counts

    # Bottom (Only FULL frame)
    if dims[-2] == 2048:
        new_data[:, :4, :] = counts

    # Restore shape
    new_data.shape = dims

    return new_data


def get_references(subarray, filter='CLEAR', context='jwst_niriss_0134.imap'):
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
              "FILTER": filter,
              "SUBARRAY": subarray}

    # Default ref file path
    default_path = resource_filename('awesimsoss', 'files/refs/')

    # Collect reference files for subarray+filter combination
    try:
        import crds
        refs = crds.getreferences(params, context=context)
    except:

        refs = {'saturation': os.path.join(default_path, 'jwst_niriss_saturation_0010.fits'),
                'photom': os.path.join(default_path, 'jwst_niriss_photom_0037.fits'),
                'flat': os.path.join(default_path, 'jwst_niriss_flat_0190.fits'),
                'gain': os.path.join(default_path, 'jwst_niriss_gain_0005.fits'),
                'superbias': os.path.join(default_path, 'jwst_niriss_superbias_0120.fits'),
                'dark': os.path.join(default_path, 'jwst_niriss_dark_0114.fits'),
                'readnoise': os.path.join(default_path, 'jwst_niriss_readnoise_0001.fits'),
                'linearity': os.path.join(default_path, 'jwst_niriss_linearity_0011.fits')}

        if subarray == 'SUBSTRIP96':
            refs['superbias'] = os.path.join(default_path, 'jwst_niriss_superbias_0111.fits'),
            refs['dark'] = os.path.join(default_path, 'jwst_niriss_dark_0111.fits')

        if subarray == 'FULL':
            refs['gain'] = os.path.join(default_path, 'jwst_niriss_gain_0002.fits'),
            refs['superbias'] = os.path.join(default_path, 'jwst_niriss_superbias_0029.fits'),
            refs['dark'] = os.path.join(default_path, 'jwst_niriss_dark_0129.fits')

    # Check if reference files exist and load defaults if necessary
    for ref_name, ref_fn in refs.items():
        if not 'NOT FOUND' in ref_fn and not os.path.isfile(refs[ref_name]):
            refs[ref_name] = os.path.join(default_path, ref_fn)
            print("Could not get {} reference file from CRDS. Using {}.".format(ref_name, ref_fn))

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
