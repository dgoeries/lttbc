#!/usr/bin/env python
import os.path as osp
import sys

from setuptools import Extension, setup


def get_script_path():
    return osp.dirname(osp.realpath(sys.argv[0]))


class numpy_get_include:
    def __str__(self):
        import numpy
        return numpy.get_include()


lttbc_py = Extension("lttbc", sources=["src/lttbc/lttbc.c"],
                     define_macros=[
                         ("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
                     include_dirs=[numpy_get_include(),
                                   get_script_path()],
                     )

setup(
    ext_modules=[lttbc_py],
)
