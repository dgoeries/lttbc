#!/usr/bin/env python
from setuptools import Extension, setup


class numpy_get_include:
    def __str__(self):
        import numpy
        return numpy.get_include()


lttbc_py = Extension("lttbc", sources=["lttbc.c"],
                     define_macros=[
                         ("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
                     include_dirs=[numpy_get_include()],
                     )
setup(
    ext_modules=[lttbc_py],
)
