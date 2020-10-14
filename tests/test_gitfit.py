#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for `gitfit` module."""

from glob import glob
import os
from pkg_resources import resource_filename
import shutil
import sys
import unittest

from astropy.io import fits
import numpy as np

from awesimsoss import gitfit


class TestGitfit(unittest.TestCase):
    """Tests for gitfit.py"""
    def setUp(self):

        # Set minimum file size in MB
        self.MB_limit = 80

        # Set number of extensions
        self.n_ext = 2

        # Make sure temp directory exists
        self.temp_dir = resource_filename('awesimsoss', 'files/temp')
        if not os.path.exists(self.temp_dir):
            os.mkdir(self.temp_dir)

        # Make dummy large FITS file (960MB)
        self.file = os.path.join(self.temp_dir, 'big_fits_file.fits')
        self.shape = 15, 2000, 2000
        gitfit.make_dummy_file(self.file, shape=self.shape, n_ext=self.n_ext)

        # Make small file
        self.small_file = os.path.join(self.temp_dir, 'small_fits_file.fits')
        self.small_shape = 1, 10, 10
        gitfit.make_dummy_file(self.small_file, shape=self.small_shape, n_ext=self.n_ext)

    def test_disassemble(self):
        """Test that a large FITS file can be disassembled"""
        # Disassemble FITS file
        file_list = gitfit.disassemble(self.file, MB_limit=self.MB_limit)
        data_dir = os.path.basename(self.file).replace('.fits', '_data')
        temp_files = glob(os.path.join(self.temp_dir, '{}/*'.format(data_dir)))

        # Test files were created
        for file in file_list:
            self.assertTrue(file in temp_files)

        # Test files are all below 80MB
        for file in temp_files:
            file_MB = os.path.getsize(file)/1000000
            self.assertTrue(file_MB < self.MB_limit)

    def test_disassemble_small(self):
        """Test that a small FITS file is not disassembled"""
        # Disassemble FITS file
        file_list = gitfit.disassemble(self.small_file, MB_limit=self.MB_limit)
        data_dir = os.path.basename(self.small_file).replace('.fits', '_data')

        # Make sure no dir was created
        no_data = os.path.join(self.temp_dir, data_dir)
        self.assertFalse(os.path.exists(no_data))

        # Check list of files is empty
        self.assertEqual(len(file_list), 0)

    def test_reassemble(self):
        """Test that a large FITS file can be reassembled"""
        # Reassemble HDUList
        _ = gitfit.disassemble(self.file, MB_limit=self.MB_limit)
        hdulist = gitfit.reassemble(self.file)

        # Check number of extensions
        self.assertEqual(len(hdulist), 3)

        # Check data shapes
        for n in np.arange(self.n_ext):
            self.assertEqual(hdulist[n + 1].data.shape, self.shape)

        # Reassemble HDUList and delete data
        _ = gitfit.disassemble(self.file, MB_limit=self.MB_limit)
        hdulist = gitfit.reassemble(self.file, save=True)

        # Check number of extensions (PRIMARY + SCI extensions)
        self.assertEqual(len(hdulist), 1 + self.n_ext)

        # Check data was removed
        self.assertFalse(os.path.exists(self.file.replace('.fits', '_data')))

    def test_reassemble_small(self):
        """Test that a small FITS file is not reassembled"""
        # Reassemble HDUList
        _ = gitfit.disassemble(self.small_file, MB_limit=self.MB_limit)
        hdulist = gitfit.reassemble(self.small_file)

        # Check number of extensions (PRIMARY + SCI extensions)
        self.assertEqual(len(hdulist), 1 + self.n_ext)

        # Check data shapes
        for n in np.arange(self.n_ext):
            self.assertEqual(hdulist[n + 1].data.shape, self.small_shape)

    def test_make_dummy_file(self):
        """Test the make_dummy_file function"""
        # Dummy file path
        path = os.path.join(self.temp_dir, 'dummy_file.fits')

        # Make the dummy file
        gitfit.make_dummy_file(path)

        # Check file was created
        os.path.isfile(path)

    def tearDown(self):
        """Remove test files"""
        # Remove temp contents
        shutil.rmtree(self.temp_dir)
        os.mkdir(self.temp_dir)
