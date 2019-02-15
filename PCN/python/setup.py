#!/usr/bin/python3

from distutils.core import setup
from distutils.extension import Extension
from distutils.core import setup, Extension

setup (
        name = 'Python PCN wrapper',
        packages = ['PyPCN'],
        #scripts=['PyPCN.py'],
        description = 'This package wraps PCN detector')

#module1 = Extension('demo',
#        sources = ['../PCN.cppdemo.c'])
#
#
#modules = [Extension../PCN.cpp("libPCNAPI.so",
#    ["../PCN.cpp"],
#    language = "c++",
#    extra_compile_args=["-fopenmp"],
#    extra_link_args=["-fopenmp"])]

