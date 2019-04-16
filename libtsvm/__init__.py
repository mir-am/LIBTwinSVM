# LIBTwinSVM
# Developers: Mir, A. and Mahdi Rahbar
# Version: 0.1 - 2019-03-20
# License: GNU General Public License v3.0


# The device that does the major computation
__DEVICE = 'CPU'


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
    __DEVICE = 'GPU'


def set_device_CPU():
    """
    Sets the current device to the GPU
    """
    
    global __DEVICE
    __DEVICE = 'CPU'
    