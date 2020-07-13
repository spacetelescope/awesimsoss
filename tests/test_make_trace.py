#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for `make_trace` module."""

import unittest

import numpy as np

from awesimsoss import make_trace as mt


def test_nuke_psfs():
    """Test for nuke_psfs function. This invokes calculate_psf_tilts, generate_SOSS_psfs, and SOSS_psf_cube."""
    mt.nuke_psfs()


def test_generate_SOSS_ldcs():
    """A test of the generate_SOSS_ldcs function"""
    lookup = mt.generate_SOSS_ldcs(np.linspace(1., 2., 3), 'quadratic', [3300, 4.5, 0])

    # Make sure three wavelengths are returned
    assert len(lookup) == 3

    # Make sure 2 coefficients are returned (for quadratic profile)
    assert len(lookup[0]) == 2
