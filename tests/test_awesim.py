#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for `awesim` module."""

import copy
import unittest
from pkg_resources import resource_filename

import numpy as np
import astropy.units as q
import astropy.constants as ac
try:
    import batman
except ImportError:
    print("Could not import `batman` package. Functionality limited.")

from awesimsoss import TSO, BlackbodyTSO, TestTSO


class TestTSO(unittest.TestCase):
    """Tests for the TSO class"""
    def setUp(self):
        """Setup for the tests"""
        # Make star data
        star_file = resource_filename('awesimsoss', 'files/scaled_spectrum.txt')
        star = np.genfromtxt(star_file, unpack=True)
        self.star = [star[0]*q.um, (star[1]*q.W/q.m**2/q.um).to(q.erg/q.s/q.cm**2/q.AA)]

        # Make planet data
        planet_file = resource_filename('awesimsoss', '/files/WASP107b_pandexo_input_spectrum.dat')
        self.planet = np.genfromtxt(planet_file, unpack=True)

    def test_init(self):
        """Test that the TSO class is generated properly"""
        # Initialize the 256 subarray with two groups and two integrations
        # and the CLEAR filter
        tso256clear = TSO(ngrps=2, nints=2, star=self.star)

        self.assertEqual(tso256clear.ngrps, 2)
        self.assertEqual(tso256clear.nints, 2)
        self.assertEqual(tso256clear.nframes, 4)
        self.assertEqual(tso256clear.dims, (2, 2, 256, 2048))
        self.assertEqual(tso256clear.subarray, 'SUBSTRIP256')
        self.assertEqual(tso256clear.filter, 'CLEAR')

        # Initialize the 96 subarray with two groups and two integrations
        # and the CLEAR filter
        tso96clear = TSO(ngrps=2, nints=2, star=self.star, subarray='SUBSTRIP96')

        self.assertEqual(tso96clear.ngrps, 2)
        self.assertEqual(tso96clear.nints, 2)
        self.assertEqual(tso96clear.nframes, 4)
        self.assertEqual(tso96clear.dims, (2, 2, 96, 2048))
        self.assertEqual(tso96clear.subarray, 'SUBSTRIP96')
        self.assertEqual(tso96clear.filter, 'CLEAR')

        # Initialize the 256 subarray with two groups and two integrations
        # and the F277W filter
        tso256f277w = TSO(ngrps=2, nints=2, star=self.star, filter='F277W')

        self.assertEqual(tso256f277w.ngrps, 2)
        self.assertEqual(tso256f277w.nints, 2)
        self.assertEqual(tso256f277w.nframes, 4)
        self.assertEqual(tso256f277w.dims, (2, 2, 256, 2048))
        self.assertEqual(tso256f277w.subarray, 'SUBSTRIP256')
        self.assertEqual(tso256f277w.filter, 'F277W')

        # Initialize the 96 subarray with two groups and two integrations
        # and the F277W filter
        tso96f277w = TSO(ngrps=2, nints=2, star=self.star, subarray='SUBSTRIP96', filter='F277W')

        self.assertEqual(tso96f277w.ngrps, 2)
        self.assertEqual(tso96f277w.nints, 2)
        self.assertEqual(tso96f277w.nframes, 4)
        self.assertEqual(tso96f277w.dims, (2, 2, 96, 2048))
        self.assertEqual(tso96f277w.subarray, 'SUBSTRIP96')
        self.assertEqual(tso96f277w.filter, 'F277W')

    def test_run_no_planet(self):
        """A test of run_simulation() with no planet"""
        # Make the TSO object
        tso256 = TSO(ngrps=2, nints=2, star=self.star)

        # Run the CLEAR simulation
        tso256.run_simulation()

    def test_run_with_planet(self):
        """A test of run_simulation() with a planet"""
        # Make the TSO object
        tso256clear = TSO(ngrps=2, nints=2, star=self.star)

        # Make the parameters
        try:
            params = batman.TransitParams()
            params.t0 = 0.
            params.per = 5.7214742
            params.a = 0.0558*q.AU.to(ac.R_sun)*0.66
            params.inc = 89.8
            params.ecc = 0.
            params.w = 90.
            params.limb_dark = 'quadratic'
            params.u = [0.1, 0.1]
            tmodel = batman.TransitModel(params, tso256clear.time)
            tmodel.teff = 3500
            tmodel.logg = 5
            tmodel.feh = 0

            # Run the simulation
            tso256clear.run_simulation(planet=self.planet, tmodel=tmodel)

        except:
            pass

    def test_lookup(self):
        """Test that coordinates are looked up if given a name"""
        # Make the TSO object
        no_targ = TSO(ngrps=2, nints=2, star=self.star)
        targ = TSO(ngrps=2, nints=2, star=self.star, target='trappist-1')

        # Check target name
        self.assertNotEqual(targ.target, no_targ.target)

        # Check coordinates
        self.assertNotEqual(targ.ra, no_targ.ra)
        self.assertNotEqual(targ.dec, no_targ.dec)

def test_TestTSO():
    """A test of the test instance!"""
    x = TestTSO()

def test_BlackbodyTSO():
    """A test of the BlackbodyTSO class"""
    x = BlackbodyTSO(teff=2000)
