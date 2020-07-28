#! /usr/bin/env python
import os
from pkg_resources import resource_filename

from astropy.io import fits

from jwst import datamodels as dm

SUB_SLICE = {'SUBSTRIP96': slice(1792, 1888), 'SUBSTRIP256': slice(1792, 2048), 'FULL': slice(0, 2048)}


def jwst_dark_ref(subarray):
    """
    Function to retrieve dark current reference file from installed jwst calibration pipeline

    Parameters
    ----------
    subarray: str
        The subarray to use, ['SUBSTRIP96', 'SUBSTRIP256', 'FULL']

    Returns
    -------
    np.ndarray
        The sliced dark current reference file data
    """
    dark_file = resource_filename('awesimsoss', 'files/signaldms.fits')
    dark_data = fits.getdata(dark_file)[SUB_SLICE[subarray], :]

    return dark_data


def jwst_nonlinearity_ref(subarray):
    """
    Function to retrieve nonlinearity reference file from installed jwst calibration pipeline

    Parameters
    ----------
    subarray: str
        The subarray to use, ['SUBSTRIP96', 'SUBSTRIP256', 'FULL']

    Returns
    -------
    np.ndarray
        The sliced nonlinearity reference file data
    """
    nonlin_file = resource_filename('awesimsoss', 'files/forward_coefficients_dms.fits')
    nonlin_data = fits.getdata(nonlin_file)[:, SUB_SLICE[subarray], :]

    return nonlin_data


def jwst_pca0_ref():
    """
    Function to get the location of the PCA0 file from installed jwst calibration pipeline
    """
    pca0_file = resource_filename('awesimsoss', 'files/niriss_pca0.fits')

    return pca0_file


def jwst_pedestal_ref(subarray):
    """
    Function to retrieve pedestal reference file from installed jwst calibration pipeline

    Parameters
    ----------
    subarray: str
        The subarray to use, ['SUBSTRIP96', 'SUBSTRIP256', 'FULL']

    Returns
    -------
    np.ndarray
        The sliced pedestal reference file data
    """
    ped_file = resource_filename('awesimsoss', 'files/pedestaldms.fits')
    ped_data = fits.getdata(ped_file)[SUB_SLICE[subarray], :]

    return ped_data


def jwst_photom_ref(filter):
    """
    Function to retrieve photom reference file from installed jwst calibration pipeline

    Parameters
    ----------
    filter: str
        The filter to use, ['CLEAR', 'F277W']

    Returns
    -------
    np.ndarray
        The absolute photometric scaling data
    """
    photom_file = resource_filename('awesimsoss', 'files/niriss_ref_photom.fits')
    photom_dict = fits.getdata(photom_file)
    photom_data = photom_dict[(photom_dict['pupil'] == 'GR700XD') & (photom_dict['filter'] == filter)]

    return photom_data


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

    return photom_data


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
