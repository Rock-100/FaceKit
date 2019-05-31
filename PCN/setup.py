#!/usr/bin/python3

from distutils.core import setup
from distutils.extension import Extension
from distutils.core import setup, Extension

setup ( name = 'Python PCN wrapper',
        scripts=['PyPCN.py'],
        description = 'This package wraps PCN detector')

