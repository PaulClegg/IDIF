"""
file: utilities.py
Generic reading, writing and display functions
"""

import numpy as np
from matplotlib import pyplot as plt

import sirf.Reg as uReg


def readNiftiImageData(filename, verbose=True):
    
    if verbose: print("\n" + filename)

    image = uReg.ImageData(filename)

    if verbose: print(image.dimensions())

    return image

def imshow(image, aspect=None, title=''):
    """Display an image with a colourbar, returning the plot handle.

    Arguments:
    image -- a 2D array of numbers
    limits -- colourscale limits as [min,max]. An empty [] uses the full range
    title -- a string for the title of the plot (default "")
    """
    plt.title(title)
    bitmap=plt.imshow(image, aspect=aspect)
    limits=[np.nanmin(image),np.nanmax(image)]

    plt.clim(limits[0], limits[1])
    plt.colorbar(shrink=.6)
    plt.axis('off')
    return bitmap

def displayRegImageData(image_data, aspect=None, title=""):
    image_array = image_data.as_array()
    
    image_shape = image_array.shape
    z = image_shape[2] // 2
    plt.figure()
    imshow(image_array[:, :, z], aspect, title)
    plt.show()

 
