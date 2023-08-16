#!/usr/bin/env python
# -*- coding: utf-8 -*-

# LIBTwinSVM: A Library for Twin Support Vector Machines
# Developers: Mir, A. and Mahdi Rahbar
# License: GNU General Public License v3.0

"""
Installation of LIBTwinSVM library.
"""

from libtsvm import __version__
from pkg_resources import parse_version
from shutil import copy
# from setuptools import find_packages, setup, Command
import os
import subprocess
import sys
import traceback


# Package meta-data.
NAME = "LIBTwinSVM"
DESCRIPTION = "LIBTwinSVM: A Library for Twin Support Vector Machines."
URL = 'https://github.com/mir-am/LIBTwinSVM'
EMAIL = "mir-am@hotmail.com, mahdirahbar@gmail.com"
AUTHOR = "Mir, A. and Mahdi Rahbar"
REQUIRES_PYTHON = '>=3.5'
VERSION = '%s' % __version__
REQ_PACKAGES = ["cython", "numpy", "matplotlib", "pyQt5", "scikit-learn", "pandas",
                "xlsxwriter", "joblib", "numpydoc==0.7.0"]
PKG_DATA = {}
DATA_FILES = []

NUMPY_MIN_VERSION = '1.14.0'
CYTHON_MIN_VERSION = '0.28'

here = os.path.abspath(os.path.dirname(__file__))

LONG_DESCRIPTION = ("Please check out the project's `GitHub page "
                    "<https://github.com/mir-am/LIBTwinSVM>`_ for installation"
                    " guide and documentation.")

# using setuptool features
SETUPTOOLS_COMMANDS = set([
    'develop', 'release', 'install', 'bdist_egg', 'bdist_rpm',
    'bdist_wininst', 'install_egg_info', 'build_sphinx',
    'egg_info', 'easy_install', 'upload', 'bdist_wheel',
    '--single-version-externally-managed',
])
if SETUPTOOLS_COMMANDS.intersection(sys.argv):
    import setuptools

    class BinaryDistribution(setuptools.Distribution):
        def has_ext_modules(foo):
            return True

    if 'bdist_wheel' in sys.argv or 'install' in sys.argv:

        if sys.platform == 'win32':

            DATA_FILES = [(os.path.join('libtsvm', 'optimizer'),
                         [os.path.join('libtsvm', 'optimizer',
                                       'lapack_win64_MT.dll'),
             os.path.join('libtsvm', 'optimizer', 'blas_win64_MT.dll')])]

            print("Added LAPACK and BLAS DLLs to the wheel for Windows "
                  "platform.")

    extra_setuptools_args = dict(
        zip_safe=False,  # the package can run out of an .egg file
        include_package_data=True,
        distclass=BinaryDistribution,
        data_files=DATA_FILES,
        extras_require={
            'alldeps': (
                'numpy >= {0}'.format(NUMPY_MIN_VERSION),
                'cython >= {0}'.format(CYTHON_MIN_VERSION),
            ),
        },
    )

else:
    extra_setuptools_args = dict()


### Utility functinos for installation #######################################    
def cp_libs_win():
    """
    Copies external libraries for installing package on Windows.
    """

    src_1 = os.path.join('libtsvm', 'optimizer', 'armadillo-code', 'lib_win64',
                         'lapack_win64_MT.dll')
    src_2 = os.path.join('libtsvm', 'optimizer', 'armadillo-code', 'lib_win64',
                         'blas_win64_MT.dll')

    dst = os.path.join('libtsvm', 'optimizer')

    copy(src_1, dst)
    copy(src_2, dst)

    print("Copied the LAPACK and BLAS libraries to source distribution...")


def get_numpy_status():
    """
    It checks whether numpy is installed. If so, checks whether it is
    up-to-date.
    """

    np_status = {}

    try:

        import numpy

        np_status['version'] = numpy.__version__
        np_status['up_to_date'] = parse_version(np_status['version']) >=  \
        parse_version(NUMPY_MIN_VERSION)

    except ImportError:

        traceback.print_exc()

        np_status['version'] = ""
        np_status['up_to_date'] = False

    return np_status


def get_cython_status():
    """
    It checks whether Cython is installed. If so, checks whether it is
    up-to-date.
    """

    cy_status = {}

    try:

        import cython

        cy_status['version'] = cython.__version__
        cy_status['up_to_date'] = parse_version(cy_status['version']) >=  \
        parse_version(CYTHON_MIN_VERSION)

    except ImportError:

        traceback.print_exc()

        cy_status['version'] = ""
        cy_status['up_to_date'] = False

    return cy_status


def check_install_updated(pkg_name, pkg_status, req_str, help_instr):
    """
    It checks whether a package is installed and up-to-date.
    """

    if pkg_status['version']:

        if pkg_status['up_to_date'] is False:

            raise ImportError("Your current version of %s %s is out-of-date. %s %s" %  \
                              (pkg_name, pkg_status['version'], req_str,
                               help_instr))

        else:

            print("Found %s %s" % (pkg_name, pkg_status['version']))

    else:

        raise ImportError("%s is not installed. %s %s "
                          % (pkg_name, req_str, help_instr))


# This function is written with inspiration from NumPy's setup.py
def check_arma_submodule():
    """
    Verify that Armadillo submodule is initialized and checked out.
    """

    if not os.path.exists('.git'):
        return  # Not a git reposiroty

    help_instr = ("Please use the following command to initialize Armadillo"
                  " submodule:\ngit submodule update --init")

    # Check submodule is not missing
    with open('.gitmodules') as f:
        for l in f:
            if 'path' in l:
                p = l.split('=')[-1].strip()
                if not os.path.exists(p):
                    raise ValueError('Submodule %s missing.\n%s' % (p,
                                     help_instr))

    # Check submodule is clean
    proc = subprocess.Popen(['git', 'submodule', 'status'],
                            stdout=subprocess.PIPE)
    status, _ = proc.communicate()
    status = status.decode("ascii", "replace")
    for line in status.splitlines():
        if line.startswith('-') or line.startswith('+'):
            raise ValueError('Submodule not clean: %s\n%s' % (line,
                                                              help_instr))

    print("Found Armadillo submodule.")


def detect_anaconda_dist():
    """
    It detects Anaconda distribution.
    """

    return True if 'anaconda' in sys.version.lower() else False


def configuration(parent_package='', top_path=None):

    from numpy.distutils.misc_util import Configuration

    config = Configuration(None, parent_package, top_path)

    config.set_options(ignore_setup_xxx_py=True,
                       assume_default_configuration=True,
                       delegate_options_to_subpackages=True)
                       # quiet=True)

    config.add_subpackage('libtsvm')

    return config


def setup_package():

    global PKG_DATA

    # Check dependencies and requirements ####################################
    help_instr = ("Please check out the installation guide of the LIBTwinSVM:\n"
                  "https://libtwinsvm.readthedocs.io/en/latest/")
    np_req_str = "LIBTwinSVM requires NumPy >= %s.\n" % NUMPY_MIN_VERSION
    cy_req_str = ("Please install cython with a version >= %s in order to"
                  " build the LIBTwinSVM library." % CYTHON_MIN_VERSION)

    check_arma_submodule()

    # Check numpy status and its version on users' system
    np_status = get_numpy_status()
    cy_status = get_cython_status()

    check_install_updated('Numerical Python (NumPy)', np_status, np_req_str,
                          help_instr)
    check_install_updated('Cython', cy_status, cy_req_str, help_instr)
    ##########################################################################

    # Platform-dependent options
    if sys.platform == 'win32':

        cp_libs_win()

        PKG_DATA = {'libtsvm': [os.path.join('libtsvm', 'optimizer', '*.dll')]}

    # Remove PyQT5 from dependencies for Anaconda distribution
    if detect_anaconda_dist():
        # TODO: Check whether PyQT is installed.
        print("Detected Anaconda distribution...")
        REQ_PACKAGES.remove("pyQt5")

    metadata = dict(
        name=NAME,
        version=VERSION,
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        long_description_content_type='text/markdown',
        author=AUTHOR,
        author_email=EMAIL,
        python_requires=REQUIRES_PYTHON,
        url=URL,
        package_data=PKG_DATA,
        #packages=find_packages(exclude=["tests", "*.tests", "*.tests.*", "tests.*"]),
        # If your package is a single module, use this instead of 'packages':
        # py_modules=['mypackage'],

        # entry_points={
        #     'console_scripts': ['mycli=mymodule:cli'],
        # },
        install_requires=REQ_PACKAGES,
        license='GNU General Public License v3.0',
        classifiers=[
            # Trove classifiers
            # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
            'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
            'Programming Language :: C++',
            'Programming Language :: Python',
            'Programming Language :: Python :: 3.5',
            'Programming Language :: Python :: 3.6',
            'Programming Language :: Python :: 3.7',
            'Programming Language :: Python :: Implementation :: CPython',
            'Operating System :: Microsoft :: Windows',
            'Operating System :: POSIX',
            'Operating System :: Unix',
            'Intended Audience :: Education',
            'Intended Audience :: Science/Research',
            'Intended Audience :: Developers',
            'Topic :: Scientific/Engineering :: Artificial Intelligence'
        ],
        **extra_setuptools_args)

    from numpy.distutils.core import setup

    metadata['configuration'] = configuration

    setup(**metadata)


if __name__ == '__main__':
    setup_package()
