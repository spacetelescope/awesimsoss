#!/usr/bin/env python
from setuptools import setup

setup(name='awesimsoss',
      version=0.1,
      description='Advanced Webb Exposure Simulator for SOSS',
      install_requires=['astropy', 'scipy', 'matplotlib', 'numpy', 'batman-package', 'ExoCTK'],
      author='Joe Filippazzo and Jonathan Fraine',
      author_email='jfilippazzo@stsci.edu',
      license='MIT',
      url='https://github.com/spacetelescope/awesimsoss',
      long_description='awesimsoss is a Time Series Observation (TSO) simulator for the Single Object Slitless Spectroscopy (SOSS) mode of the Near Infrared Imaging and Slitless Spectrscopy (NIRISS) instrument onboard the James Webb Space Telescope (JWST)',
      zip_safe=False,
      use_2to3=False
)
