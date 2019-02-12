#!/usr/bin/python3

from distutils.core import setup
from distutils.extension import Extension
#from Cython.Build import cythonize
#from Cython.Distutils import build_ext

from distutils.core import setup, Extension

module1 = Extension('demo',
                    define_macros = [('MAJOR_VERSION', '1'),
                                     ('MINOR_VERSION', '0')],
                    include_dirs = ['/usr/local/include','/usr/local/include'],
                    library_dirs = ['/usr/lib/x86_64-linux-gnu/'],
                    libraries = ["opencv_core ","opencv_highgui","opencv_imgcodecs ","opencv_imgproc ","opencv_video ","opencv_videoio ","caffe ","glog ","boost_system ","protobuf" ],
		    extra_compile_args = ["-DCPU_ONLY"],
                    sources = ['src/PCN.cpp'])

setup (name = 'PackageName',
       version = '1.0',
       description = 'This is a demo package',
       author = 'Martin v. Loewis',
       author_email = 'martin@v.loewis.de',
       url = 'https://docs.python.org/extending/building',
       long_description = ''' This is really just a demo package.  ''',
       ext_modules = [module1],
       install_requires=['distribute'])

#module1 = Extension('demo',
#        sources = ['../PCN.cppdemo.c'])
#
#
#modules = [Extension../PCN.cpp("libPCNAPI.so",
#    ["../PCN.cpp"],
#    language = "c++",
#    extra_compile_args=["-fopenmp"],
#    extra_link_args=["-fopenmp"])]

