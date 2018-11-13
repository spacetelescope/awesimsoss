#! /usr/bin/env python
# -*- coding: utf-8 -*-

try:
    from setuptools import setup, find_packages
    setup
except ImportError:
    from distutils.core import setup
    setup

from codecs import open
from os import path

setup(name='awesimsoss',
      version='0.1.0',
      description='Advanced Webb Exposure Simulator for SOSS',
      long_description='AWESim_SOSS is a TSO simulator for the Single Object Slitless Spectroscopy mode of the NIRISS instrument onboard JWST',
      url='https://github.com/spacetelescope/awesimsoss',
      author='Joe Filippazzo',
      author_email='jfilippazzo@stsci.edu',
      license='MIT',
      classifiers=[
          'Development Status :: 4 - Beta',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python',
      ],
      keywords='astrophysics',
      packages=find_packages(exclude=['contrib', 'docs']),
      package_data={'svo_filters': ['data/filters/*', 'data/plots/*', 'data/spectra/*']},
      include_package_data=True,
      zip_safe=False,
      install_requires=['astropy', 'scipy', 'matplotlib', 'numpy', 'batman-package', 'exoctk', 'h5py'],
      use_2to3=False

)