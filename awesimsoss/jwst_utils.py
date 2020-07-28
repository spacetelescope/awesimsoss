#! /usr/bin/env python
import os
from pkg_resources import resource_filename

from astropy.io import fits

from jwst.datamodels import RampModel


def jwst_photom_ref(filt):
    """Function to retrieve photom reference file from installed jwst calibration pipeline"""
    calfile = resource_filename('jwst', 'files/niriss_ref_photom.fits')
    caldata = fits.getdata(calfile)
    photom = caldata[(caldata['pupil'] == 'GR700XD') & (caldata['filter'] == filt)]

    return photom


def jwst_ramp_model(data, groupdq, pixeldq, err, **kwargs):
    """Function to retrieve RampModel object from installed jwst calibration pipeline"""
    model = RampModel(data=data, groupdq=groupdq, pixeldq=pixeldq, err=err, **kwargs)

    return model