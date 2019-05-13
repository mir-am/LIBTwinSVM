#!/usr/bin/env python
# -*- coding: utf-8 -*-

# LIBTwinSVM
# Developers: Mir, A. and Mahdi Rahbar
# Version: 0.1 - 2019-03-20
# License: GNU General Public License v3.0

"""
LIBTwinSVM Program - A Library for Twin Support Vector Machines

"""

from libtsvm import __version__
from pkg_resources import parse_version
from shutil import copy
#from setuptools import find_packages, setup, Command
import io
import os
import subprocess
import sys
import traceback


# Package meta-data.
NAME = "LIBTwinSVM"
DESCRIPTION = "LIBTwinSVM Program -A Library for Twin Support Vector Machines with an easy-to-use Graphical User Interface."
URL = 'https://github.com/mir-am/LIBTwinSVM'
EMAIL = "mir-am@hotmail.com | MahdiRahbar@Gmail.com"
AUTHOR = "Mir, A. and Mahdi Rahbar"
REQUIRES_PYTHON = '>=3.5'
VERSION = '%s' % __version__
REQ_PACKAGES = ["cython", "numpy", "matplotlib", "pyQt5", "sklearn","pandas",
            "xlsxwriter", "joblib","numpydoc==0.7.0"]
PKG_DATA = {}

NUMPY_MIN_VERSION = '1.14.0'
CYTHON_MIN_VERSION = '0.28'

here = os.path.abspath(os.path.dirname(__file__))


# Import the README and use it as the long-description.
# Note: this will only work if 'README.md' is present in your MANIFEST.in file!
try:
    with io.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
        long_description = '\n' + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION



# Load the package's __version__.py module as a dictionary.
#about = {}
#if not VERSION:
#    project_slug = NAME.lower().replace("-", "_").replace(" ", "_")
#    with open(os.path.join(here, project_slug, '__version__.py')) as f:
#        exec(f.read(), about)
#else:
#    about['__version__'] = VERSION


#def build_ext_clipdcd():
#    """It builds the clipdcd extension module"""
#    
#    from libtsvm.optimizer.setup import ext_clipdcd
#    from Cython.Build import cythonize    
#    
#    cythonize(ext_clipdcd)

# class UploadCommand(Command):
#     """Support setup.py upload."""

#     description = 'Build and publish the package.'
#     user_options = []

#     @staticmethod
#     def status(s):
#         """Prints things in bold."""
#         print('\033[1m{0}\033[0m'.format(s))

#     def initialize_options(self):
#         pass

#     def finalize_options(self):
#         pass

#     def run(self):
#         try:
#             self.status('Removing previous builds…')
#             rmtree(os.path.join(here, 'dist'))
#         except OSError:
#             pass

#         self.status('Building Source and Wheel (universal) distribution…')
#         os.system('{0} setup.py sdist bdist_wheel --universal'.format(sys.executable))

#         self.status('Uploading the package to PyPI via Twine…')
#         os.system('twine upload dist/*')

#         self.status('Pushing git tags…')
#         os.system('git tag v{0}'.format(about['__version__']))
#         os.system('git push --tags')
        
#         sys.exit()
    
# using setuptool features
SETUPTOOLS_COMMANDS = set([
    'develop', 'release', 'bdist_egg', 'bdist_rpm',
    'bdist_wininst', 'install_egg_info', 'build_sphinx',
    'egg_info', 'easy_install', 'upload', 'bdist_wheel',
    '--single-version-externally-managed',
])
if SETUPTOOLS_COMMANDS.intersection(sys.argv):
    import setuptools
    
    class BinaryDistribution(setuptools.Distribution):
        def has_ext_modules(foo):
            return True
        
    if 'bdist_wheel' in sys.argv:
        
        data_files = [(os.path.join('libtsvm', 'optimizer'),
        [os.path.join('libtsvm', 'optimizer', 'lapack_win64_MT.dll'),
         os.path.join('libtsvm', 'optimizer', 'blas_win64_MT.dll')])]
    
        print("Added LAPACK and BLAS DLLs to the wheel.")

    else:
        
        data_files = []

    extra_setuptools_args = dict(
        zip_safe=False,  # the package can run out of an .egg file
        include_package_data=True,
        distclass=BinaryDistribution,
        data_files=data_files,
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
    It checks whether numpy is installed. If so, checks whether it is up-to-date.
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
    It checks whether Cython is installed. If so, checks whether it is up-to-date.
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
        return # Not a git reposiroty
    
    help_instr = ("Please use the following command to initialize Armadillo"
                  " submodule:\ngit submodule update --init")
    
    # Check submodule is not missing
    with open('.gitmodules') as f:
        for l in f:
            if 'path' in l:
                p = l.split('=')[-1].strip()
                if not os.path.exists(p):
                    raise ValueError('Submodule %s missing.\n%s' % (p, help_instr))
    
    # Check submodule is clean                
    proc = subprocess.Popen(['git', 'submodule', 'status'],
                            stdout=subprocess.PIPE)
    status, _ = proc.communicate()
    status = status.decode("ascii", "replace")
    for line in status.splitlines():
        if line.startswith('-') or line.startswith('+'):
            raise ValueError('Submodule not clean: %s\n%s' % (line, help_instr))
    
    print("Found Armadillo submodule.")
    
# This code is borrowed from Matthew Brett's windows-wheel-builder
#def add_dll_wheel(dist_path='dist'):
#    """
#    Adds runtime DLLs to pre-built wheels for Windows.
#    """
#    
#    from glob import glob
#    # delocate package is only needed for building wheels for Windows systems.
#    from delocate import wheeltools
#    from zipfile import ZipFile
#    
#    def my_zip2dir(zip_fname, out_dir):
#        """ Removes need for 'unzip' binary in PATH during `wheeltools.zip2dir`
#        """
#        with open(zip_fname, 'rb') as fobj:
#            zip = ZipFile(fobj)
#            zip.extractall(path=out_dir)
#
#    # Monkeypatch wheeltools
#    wheeltools.zip2dir = my_zip2dir
#    
#    lapack_dll = os.path.join('libtsvm', 'optimizer', 'lapack_win64_MT.dll')
#    blas_dll = os.path.join('libtsvm', 'optimizer', 'blas_win64_MT.dll')
#    
#    wheel_fnames = glob(os.path.join(dist_path, '*.whl'))
#    
#    print(wheel_fnames)
#    
#    for fname in wheel_fnames:
#        print('Processing', fname)
#        with wheeltools.InWheel(fname, fname):
#            print(os.path.abspath())
#            
#            copy2(lapack_dll, os.path.join('libtsvm', 'optimizer'))
#            copy2(blas_dll, os.path.join('libtsvm', 'optimizer'))
#            
#    print("Added LAPACK and BLAS DLLs to the wheel.")
    
##############################################################################
    
def configuration(parent_package='', top_path=None):
    
    from numpy.distutils.misc_util import Configuration
    
    config = Configuration(None, parent_package, top_path)
    
    config.set_options(ignore_setup_xxx_py=True,
                       assume_default_configuration=True,
                       delegate_options_to_subpackages=True) #,
                       #quiet=True)
                       
    config.add_subpackage('libtsvm')
    
    return config

def setup_package():
    
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

    metadata = dict(
        name=NAME,
        version=VERSION,
        description=DESCRIPTION,
        long_description=long_description,
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
        test_suite='tests',
        license='GNU General Public License v3.0',
        classifiers=[
            # Trove classifiers
            # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
            'License :: OSI Approved :: GNU General Public License v3.0',
            'Programming Language :: Python',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.5',
            'Programming Language :: Python :: 3.6',
            'Programming Language :: Python :: 3.7',
            'Programming Language :: Python :: Implementation :: CPython',
        ],
        **extra_setuptools_args)
    
    from numpy.distutils.core import setup
    
    metadata['configuration'] = configuration
    
    setup(**metadata)
    
    
if __name__ == '__main__':
    setup_package()
    