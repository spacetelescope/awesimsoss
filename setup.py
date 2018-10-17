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
      long_description='awesimsoss is a TSO simulator for the SOSS mode of the NIRISS instrument onboard the JWST',
      zip_safe=False,
      use_2to3=False
)
