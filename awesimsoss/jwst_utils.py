#! /usr/bin/env python
import os
from pkg_resources import resource_filename

from astropy.io import fits

from jwst.datamodels import RampModel


def jwst_dark_ref():
    """Function to retrieve dark current reference file from installed jwst calibration pipeline"""
    dark_file = resource_filename('awesimsoss', 'files/signaldms.fits')
    dark_data = fits.getdata(dark_file)

    return dark_data


def jwst_nonlinearity_ref():
    """Function to retrieve nonlinearity reference file from installed jwst calibration pipeline"""
    nonlin_file = resource_filename('awesimsoss', 'files/forward_coefficients_dms.fits')
    nonlin_data = fits.getdata(nonlin_file)

    return nonlin_data


def jwst_pca0_ref():
    """Function to get the location of the PCA0 file from installed jwst calibration pipeline"""
    pca0_file = resource_filename('awesimsoss', 'files/niriss_pca0.fits')

    return pca0_file


def jwst_pedestal_ref():
    """Function to retrieve pedestal reference file from installed jwst calibration pipeline"""
    ped_file = resource_filename('awesimsoss', 'files/pedestaldms.fits')
    ped_data = fits.getdata(ped_file)

    return ped_data


def jwst_photom_ref():
    """Function to retrieve photom reference file from installed jwst calibration pipeline"""
    photom_file = resource_filename('awesimsoss', 'files/niriss_ref_photom.fits')
    photom_data = fits.getdata(photom_file)

    return photom_data


def jwst_photyield_ref():
    """Function to retrieve photon yield reference file from installed jwst calibration pipeline"""
    photyield_file = resource_filename('awesimsoss', 'files/photonyieldfullframe.fits')
    photyield_data = fits.getdata(photyield_file)

    return photom_data


def jwst_ramp_model(data, groupdq, pixeldq, err, **kwargs):
    """Function to retrieve RampModel object from installed jwst calibration pipeline"""
    model = RampModel(data=data, groupdq=groupdq, pixeldq=pixeldq, err=err, **kwargs)

    return model


def jwst_zodi_ref():
    """Function to retrieve zodiacal background reference file from installed jwst calibration pipeline"""
    zodi_file = resource_filename('awesimsoss', 'files/background_detectorfield_normalized.fits')
    zodi_data = fits.getdata(zodi_file)

    return zodi_data
