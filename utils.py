#!/usr/bin/env python

"""Utilities for the project"""

__all__ = ['timing']
__author__ = 'Anna Kukleva'
__date__ = 'October 2018'

import numpy as np
import time
from collections import defaultdict
import os


def timing(f):
    """Wrapper for functions to measure time"""
    def wrap(*args, **kwargs):
        time1 = time.time()
        ret = f(*args, **kwargs)
        time2 = time.time()
        print('%s took %0.3f ms ~ %0.3f min ~ %0.3f sec'
              % (f, (time2-time1)*1000.0, (time2-time1)/60.0, (time2-time1)))
        return ret
    return wrap

