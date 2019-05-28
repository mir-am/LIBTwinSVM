# -*- coding: utf-8 -*-

# LIBTwinSVM: A Library for Twin Support Vector Machines
# Developers: Mir, A. and Mahdi Rahbar
# License: GNU General Public License v3.0

"""
This module is a setup file for the libtsvm subpackage.
"""

def configuration(parent_package='', top_path=None):
    
    from numpy.distutils.misc_util import Configuration
    
    config = Configuration('libtsvm', parent_package, top_path)
    
    config.add_subpackage('ui')
    # Optimizer subpackage which needs to be compiled.
    config.add_subpackage('optimizer')
    
    return config


if __name__ == '__main__':
    
    from numpy.distutils.core import setup
    
    setup(**configuration(top_path='').todict())
