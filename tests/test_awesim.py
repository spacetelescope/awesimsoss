#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for `awesim` module."""

from copy import copy
import unittest
from pkg_resources import resource_filename

import numpy as np
import astropy.units as q
import astropy.constants as ac
import batman

from awesimsoss import TSO, BlackbodyTSO, TestTSO, STAR_DATA, PLANET_DATA


class test_BlackbodyTSO(unittest.TestCase):
    """A test of the BlackbodyTSO class"""
    def setUp(self):
        pass

    def test_run_no_planet(self):
        """A test of the BlackbodyTSO class with no planet"""
        tso = BlackbodyTSO()

    def test_run_with_planet(self):
        """A test of the BlackbodyTSO class with a planet"""
        tso = BlackbodyTSO(add_planet=True)


class test_TestTSO(unittest.TestCase):
    """A test of the TestTSO class"""
    def setUp(self):
        pass

    def test_run_no_planet(self):
        """A test of the TestTSO class with no planet"""
        tso = TestTSO()

    def test_run_with_planet(self):
        """A test of the TestTSO class with a planet"""
        tso = TestTSO(add_planet=True)


class test_TSO(unittest.TestCase):
    """Tests for the TSO class"""
    def setUp(self):
        """Setup for the tests"""
        # Get data
        self.star = STAR_DATA
        self.planet = PLANET_DATA

    def test_export(self):
        """Test the export method"""
        # Make the TSO object and save
        test_tso = TSO(ngrps=2, nints=2, star=self.star, subarray='SUBSTRIP256')
        test_tso.simulate()
        try:
            test_tso.export('outfile.fits')
        except NameError:
            pass

    def test_init(self):
        """Test that the TSO class is generated properly"""
        # Initialize the FULL frame with two groups and two integrations
        # and the CLEAR filter
        tso2048clear = TSO(ngrps=2, nints=2, star=self.star, subarray='FULL')

        self.assertEqual(tso2048clear.ngrps, 2)
        self.assertEqual(tso2048clear.nints, 2)
        self.assertEqual(tso2048clear.nframes, 4)
        self.assertEqual(tso2048clear.dims, (2, 2, 2048, 2048))
        self.assertEqual(tso2048clear.subarray, 'FULL')
        self.assertEqual(tso2048clear.filter, 'CLEAR')

        # Initialize the 256 subarray with two groups and two integrations
        # and the CLEAR filter
        tso256clear = TSO(ngrps=2, nints=2, star=self.star, subarray='SUBSTRIP256')

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

        # Initialize the FULL frame with two groups and two integrations
        # and the F277W filter
        tso2048f277w = TSO(ngrps=2, nints=2, star=self.star, subarray='FULL', filter='F277W')

        self.assertEqual(tso2048f277w.ngrps, 2)
        self.assertEqual(tso2048f277w.nints, 2)
        self.assertEqual(tso2048f277w.nframes, 4)
        self.assertEqual(tso2048f277w.dims, (2, 2, 2048, 2048))
        self.assertEqual(tso2048f277w.subarray, 'FULL')
        self.assertEqual(tso2048f277w.filter, 'F277W')

        # Initialize the 256 subarray with two groups and two integrations
        # and the F277W filter
        tso256f277w = TSO(ngrps=2, nints=2, star=self.star, subarray='SUBSTRIP256', filter='F277W')

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
        """A test of simulate() with no planet"""
        # Make the TSO object
        tso = TSO(ngrps=2, nints=2, star=self.star)
        tso.simulate()
        tso.subarray = 'SUBSTRIP96'
        tso.simulate()
        tso.subarray = 'FULL'
        tso.simulate()

    def test_run_with_planet(self):
        """A test of simulate() with a planet"""
        # Make the TSO object
        tso = TSO(ngrps=2, nints=2, star=self.star)

        # Make orbital params
        params = batman.TransitParams()
        params.t0 = 0.
        params.per = 5.7214742
        params.a = 0.0558*q.AU.to(ac.R_sun)*0.66
        params.inc = 89.8
        params.ecc = 0.
        params.w = 90.
        params.limb_dark = 'quadratic'
        params.u = [0.1, 0.1]
        params.rp = 0.
        tmodel = batman.TransitModel(params, tso.time)
        tmodel.teff = 3500
        tmodel.logg = 5
        tmodel.feh = 0

        # Run the simulation
        tso.simulate(planet=self.planet, tmodel=tmodel)
        tso.subarray = 'SUBSTRIP96'
        tso.simulate(planet=self.planet, tmodel=tmodel)
        tso.subarray = 'FULL'
        tso.simulate(planet=self.planet, tmodel=tmodel)

    def test_lookup(self):
        """Test that coordinates are looked up if given a name"""
        # Make the TSO object
        targ = TSO(ngrps=2, nints=2, star=self.star, target='trappist-1')
        no_targ = TSO(ngrps=2, nints=2, star=self.star)

        # Check target name
        self.assertNotEqual(targ.target, no_targ.target)

        # Check coordinates
        self.assertNotEqual(targ.ra, no_targ.ra)
        self.assertNotEqual(targ.dec, no_targ.dec)

    def test_star(self):
        """Test that errors are thrown for bas star input"""
        # Test that non wavelength units fail
        bad_wave_star = copy(self.star)
        bad_wave_star[0] *= q.Jy
        kwargs = {'nints':2, 'ngrps':2, 'star':bad_wave_star}
        self.assertRaises(ValueError, TSO, **kwargs)

        # Test that non flux density units fail
        bad_flux_star = copy(self.star)
        bad_flux_star[1] *= q.K
        kwargs = {'nints':2, 'ngrps':2, 'star':bad_flux_star}
        self.assertRaises(ValueError, TSO, **kwargs)

        # Test that no units fail
        bad_unit_star = copy(self.star)
        bad_unit_star[0] = bad_unit_star[0].value
        kwargs = {'nints':2, 'ngrps':2, 'star':bad_unit_star}
        self.assertRaises(ValueError, TSO, **kwargs)

        # Test that spectrum shape
        bad_size_star = [self.star[0]]
        kwargs = {'nints':2, 'ngrps':2, 'star':bad_size_star}
        self.assertRaises(ValueError, TSO, **kwargs)

    def test_bad_attrs(self):
        """Test that invalid attributes throw an error"""
        # Make the TSO object
        tso = TSO(ngrps=2, nints=2, star=self.star)

        # Bad fiilter
        self.assertRaises(ValueError, setattr, tso, 'filter', 'foo')

        # Bad ncols
        self.assertRaises(TypeError, setattr, tso, 'ncols', 3)

        # Bad nrows
        self.assertRaises(TypeError, setattr, tso, 'nrows', 3)

        # Bad nints
        self.assertRaises(TypeError, setattr, tso, 'nints', 'three')

        # Bad ngrps
        self.assertRaises(TypeError, setattr, tso, 'ngrps', 'three')

        # Bad nresets
        self.assertRaises(TypeError, setattr, tso, 'nresets', 'three')

        # Bad orders
        tso.orders = 1
        self.assertRaises(ValueError, setattr, tso, 'orders', 'three')

        # Bad subarray
        self.assertRaises(ValueError, setattr, tso, 'subarray', 'three')

        # Bad t0
        self.assertRaises(ValueError, setattr, tso, 't0', 'three')

        # Bad target
        self.assertRaises(TypeError, setattr, tso, 'target', 3)

    def test_ldcs(self):
        """Test the limb darkening coefficients"""
        # Create instance
        tso = TSO(ngrps=2, nints=2, star=self.star)

        # Set manually
        ldcs = tso.ld_coeffs
        tso.ld_coeffs = np.ones((3, 2048, 2))

        # Bad LDCs
        self.assertRaises(TypeError, setattr, tso, 'ld_coeffs', 'foo')

    def test_plot(self):
        """Test plot method"""
        # Make the TSO object
        tso = TSO(ngrps=2, nints=2, star=self.star)

        # Test plot with no data
        plt = tso.plot(draw=False)

        # Run simulation
        tso.simulate()

        # Test bad ptype
        kwargs = {'ptype':'foo', 'draw':False}
        self.assertRaises(ValueError, tso.plot, **kwargs)

        # Standard plot with traces
        plt = tso.plot(traces=True)

        # Standard plot with one order
        plt = tso.plot(order=1, draw=False)

        # No noise plot
        plt = tso.plot(noise=False, draw=False)

        # Log plot
        plt = tso.plot(scale='log', draw=False)

    def test_plot_slice(self):
        """Test plot_slice method"""
        # Make the TSO object
        tso = TSO(ngrps=2, nints=2, star=self.star)
        tso.simulate()

        # Standard plot with traces
        plt = tso.plot_slice(500, traces=True)

        # Standard plot with one order
        plt = tso.plot_slice(500, order=1, draw=False)

        # Plot with noise
        plt = tso.plot_slice(500, noise=True, draw=False)

        # Log plot
        plt = tso.plot_slice(500, scale='log', draw=False)

        # List of slices
        plt = tso.plot_slice([500, 1000], draw=False)

    def test_plot_ramp(self):
        """Test plot_ramp method"""
        # Make the TSO object
        tso = TSO(ngrps=2, nints=2, star=self.star)
        tso.simulate()

        # Standard plot
        plt = tso.plot_ramp(draw=False)
        tso.plot_ramp()

    def test_plot_lightcurve(self):
        """Test plot_lightcurve method"""
        # Make the TSO object
        tso = TSO(ngrps=2, nints=2, star=self.star)
        tso.simulate()

        # Test bad units
        kwargs = {'column':500, 'time_unit':'foo', 'draw':False}
        self.assertRaises(ValueError, tso.plot_lightcurve, **kwargs)

        # Standard plot
        plt = tso.plot_lightcurve(500)

        # Wavelength
        plt = tso.plot_lightcurve(1.6, draw=False)

        # Neither
        plt = tso.plot_lightcurve('foo', draw=False)

        # List of lightcurves
        plt = tso.plot_lightcurve([500, 1000], draw=False)

    def test_plot_spectrum(self):
        """Test plot_spectrum method"""
        # Make the TSO object
        tso = TSO(ngrps=2, nints=2, star=self.star)
        tso.simulate()

        # Standard plot
        plt = tso.plot_spectrum()

        # Standard plot with one order
        plt = tso.plot_spectrum(order=1, draw=False)

        # Log plot
        plt = tso.plot_spectrum(scale='log', draw=False)

        # No noise plot
        plt = tso.plot_spectrum(noise=True, draw=False)

        # Specific order
        plt = tso.plot_spectrum(order=1, draw=False)
