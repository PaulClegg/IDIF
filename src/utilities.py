"""
file: utilities.py
Generic reading, writing and display functions
"""

import os
import numpy as np
import nibabel as nib
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

def makeNiftiImageReal(filename):
    img = nib.load(filename)
    image_real = np.abs(img.get_fdata())
    img.header.set_data_dtype(image_real.dtype)
    outImg = nib.Nifti1Image(image_real, 
        affine=img.header.get_best_affine(), header=img.header)
    out_name = filename.split(".")[0] + "_real.nii"
    nib.save(outImg, out_name)

    return out_name
 
def saveGadgetronImageAsRegNifti(image_data, data_stem, filename):
    assert filename.split(".")[1] == "nii", "utilities, *.nii filename only"
    
    mri_template_image = "starVIBE_twelve_1_real.nii"
    cvs_path = "/home/pclegg/devel/SIRF-SuperBuild/docker/devel/cvs"
    path = os.path.join(cvs_path, mri_template_image)

    assert os.path.isfile(path), "utilities: template mri file doesn't exist"
    cvs_image = readNiftiImageData(path)

    image_arr = image_data.as_array()
    reg_out = cvs_image.clone()
    reg_out.fill(image_arr)
    reg_out.write(os.path.join(data_stem, filename))

