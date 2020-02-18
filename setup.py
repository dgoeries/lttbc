#!/usr/bin/env python
import os.path as osp
import sys

from setuptools import setup, Extension

import numpy


def get_script_path():
    return osp.dirname(osp.realpath(sys.argv[0]))


def read(*parts):
    return open(osp.join(get_script_path(), *parts)).read()


lttb_module = Extension('lttbc', sources=['lttbc.c'],
                        define_macros=[
                            ('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')],
                        include_dirs=[numpy.get_include(),
                                      get_script_path()],
                        )

setup(
    name="lttbc",
    author="European XFEL GmbH",
    version="0.1.7",
    include_dirs=[numpy.get_include(), get_script_path()],
    ext_modules=[lttb_module],
    author_email="dennis.goeries@xfel.eu",
    maintainer="Dennis Goeries",
    url="https://github.com/dgoeries/lttbc/",
    description="Largest triangle three buckets module for Python written in C",
    license="MIT",
    install_requires=[
        'numpy'],
)
