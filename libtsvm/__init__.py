# LIBTwinSVM
# Developers: Mir, A. and Mahdi Rahbar
# Version: 0.1 - 2019-03-20
# License: GNU General Public License v3.0


# The device that does the major computation
__DEVICE = 'CPU'
__GPU_enabled = False

# Check whether CuPy can imported. Otherwise GPU version will be disabled.
try:
    
    import cupy as cp
    
    __GPU_enabled = True
    
except Exception as err:
    
    print("Could not import CuPy package. Therefore, GPU version is disabled.\n"
          "Error message: %s" % err)


def get_current_device():
    """
    Gets the current device.
    """
    global __DEVICE
    return __DEVICE


def set_device_GPU():
    """
    Sets the current device to the GPU
    """
    
    global __DEVICE
    global __GPU_enabled
    
    if __GPU_enabled:
    
        __DEVICE = 'GPU'


def set_device_CPU():
    """
    Sets the current device to the GPU
    """
    
    global __DEVICE
    __DEVICE = 'CPU'
    
    
def set_dev_GPU_tests():
    """
    This is intented to use for only running unit tests.
    """
    
    global __DEVICE
    global __GPU_enabled
    
    if __GPU_enabled:
    
        __DEVICE = 'GPU'
        
    else:
        
        from unittest import SkipTest
        
        print("Cannot run estimators unit tests on the GPU.")
        
        raise SkipTest("GPU version is disabled. GPU tests will be ignored.")
    
    