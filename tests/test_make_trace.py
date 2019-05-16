#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for `make_trace` module."""

import copy
import unittest

from bokeh.plotting import figure, show
import numpy as np

from awesimsoss import make_trace as mt


class TestGetSOSSpsf(unittest.TestCase):
    """Tests for the get_SOSS_psf function"""
    def setUp(self, plot=False):
        """Setup for the tests"""
        # Get a SOSS psf at 1um for CLEAR and F277W
        self.clear = mt.get_SOSS_psf(1.0, filt='CLEAR')
        self.f277w = mt.get_SOSS_psf(1.0, filt='F277W')

        self.plot = plot

    def testDims(self):
        """Make sure the dimensions are right"""
        # The dimensions should be the same
        self.assertEqual(self.clear.shape, self.f277w.shape, (76, 76))

        if self.plot:
            # Plot CLEAR
            fig1 = figure()
            fig1.image([self.clear], x=0, y=0, dw=76, dh=76)
            show(fig1)

            # Plot F277W
            fig2 = figure()
            fig2.image([self.f277w], x=0, y=0, dw=76, dh=76)
            show(fig2)

    def testVals(self):
        """Make sure the PSFs look right"""
        # The data should be different (when WebbPSF is updated!)
        # self.assertNotEqual(np.sum(self.clear), np.sum(self.f277w))

        # Check that the psfs are scaled to 1
        self.assertEqual(np.sum(self.clear), 1)
        self.assertEqual(np.sum(self.f277w), 1)

        if self.plot:
            clearsum = np.sum(self.clear, axis=0)
            f277wsum = np.sum(self.f277w, axis=0)
            fig3 = figure()
            fig3.line(np.arange(76), clearsum, color='blue', legend='CLEAR')
            fig3.line(np.arange(76), f277wsum, color='red', legend='F277W')
            show(fig3)
