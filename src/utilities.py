"""
file: utilities.py
Generic reading, writing and display functions
"""

import sirf.Reg as uReg


def readNiftiImageData(filename, verbose=True):
    
    if verbose: print(filename)

    image = uReg.ImageData(filename)

    if verbose: print(image.dimensions())

    return image
