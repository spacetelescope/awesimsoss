# -*- coding: utf-8 -*-
"""
A module to of utilities to assist awesim.py

Authors: Joe Filippazzo, Kevin Volk, Jonathan Fraine, Michael Wolfe
"""

import itertools

from bokeh.palettes import Category20


def color_gen():
    yield from itertools.cycle(Category20[20])
COLORS = color_gen()


def subarray(arr):
    """
    Get the pixel information for a NIRISS subarray
    
    The returned dictionary defines the extent ('x' and 'y'),
    the starting pixel ('xloc' and 'yloc'), and the number 
    of reference pixels at each subarray edge ('x1', 'x2',
    'y1', 'y2) as defined by SSB/DMS coordinates shown below:
        ___________________________________
       |               y2                  |
       |                                   |
       |                                   |
       | x1                             x2 |
       |                                   |
       |               y1                  |
       |___________________________________|
    (1,1)
    
    Parameters
    ----------
    arr: str
        The FITS header SUBARRAY value
    
    Returns
    -------
    dict
        The dictionary of the specified subarray
        or a nested dictionary of all subarrays
    """
    pix = {'FULL': {'xloc':1, 'x':2048, 'x1':4, 'x2':4,
                    'yloc':1, 'y':2048, 'y1':4, 'y2':4,
                    'tfrm':10.737, 'tgrp':10.737},
           'SUBSTRIP96' : {'xloc':1, 'x':2048, 'x1':4, 'x2':4,
                           'yloc':1803, 'y':96, 'y1':0, 'y2':0,
                           'tfrm':2.213, 'tgrp':2.213},
           'SUBSTRIP256' : {'xloc':1, 'x':2048, 'x1':4, 'x2':4,
                            'yloc':1793, 'y':256, 'y1':0, 'y2':4,
                            'tfrm':5.491, 'tgrp':5.491},
           'SUB80' : {'xloc':None, 'x':80, 'x1':0, 'x2':0,
                      'yloc':None, 'y':80, 'y1':4, 'y2':0},
           'SUB64' : {'xloc':None, 'x':64, 'x1':0, 'x2':4,
                      'yloc':None, 'y':64, 'y1':0, 'y2':4},
           'SUB128' : {'xloc':None, 'x':128, 'x1':0, 'x2':4,
                       'yloc':None, 'y':128, 'y1':0, 'y2':4},
           'SUB256' : {'xloc':None, 'x':256, 'x1':0, 'x2':4,
                       'yloc':None, 'y':256, 'y1':0, 'y2':4},
           'SUBAMPCAL' : {'xloc':None, 'x':512, 'x1':4, 'x2':0,
                          'yloc':None, 'y':1792, 'y1':4, 'y2':0},
           'WFSS64R' : {'xloc':None, 'x':64, 'x1':0, 'x2':4,
                        'yloc':1, 'y':2048, 'y1':4, 'y2':0},
           'WFSS64C' : {'xloc':1, 'x':2048, 'x1':4, 'x2':0,
                        'yloc':None, 'y':64, 'y1':0, 'y2':4},
           'WFSS128R' : {'xloc':None, 'x':128, 'x1':0, 'x2':4,
                         'yloc':1, 'y':2048, 'y1':4, 'y2':0},
           'WFSS128C' : {'xloc':1, 'x':2048, 'x1':4, 'x2':0,
                         'yloc':None, 'y':128, 'y1':0, 'y2':4},
           'SUBTASOSS' : {'xloc':None, 'x':64, 'x1':0, 'x2':0,
                          'yloc':None, 'y':64, 'y1':0, 'y2':0},
           'SUBTAAMI' : {'xloc':None, 'x':64, 'x1':0, 'x2':0,
                         'yloc':None, 'y':64, 'y1':0, 'y2':0}}

    return pix[arr]
