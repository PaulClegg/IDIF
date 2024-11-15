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

def convertNiftiFilesToMovie(data_stem, stem, nFrames, verbose=True):
    filename = os.path.join(data_stem, stem + "0.nii")
    img1 = nib.load(filename)
    if verbose: print(img1.header)

    # create header for movie
    header = img1.header
    if verbose: print("\n\n")
    if verbose: print(header["dim"])
    dim = header["dim"]
    dim[0] = 4
    dim[4] = nFrames
    header["dim"] = dim
    if verbose: print(header["dim"])

    if verbose: print(header["pixdim"])
    pixdim = header["pixdim"]
    pixdim[4] = 1
    header["pixdim"] = pixdim
    if verbose: print(header["pixdim"])

    if verbose: print(header["xyzt_units"])
    xyzt_units = header["xyzt_units"]
    xyzt_units = 10
    header["xyzt_units"] = xyzt_units
    if verbose: print(header["xyzt_units"])

    if verbose: print(header["slice_duration"])
    slice_duration = header["slice_duration"]
    slice_duration = 1.0
    header["slice_duration"] = slice_duration
    if verbose: print(header["slice_duration"])

    if verbose: print(header["slice_code"])
    slice_code = header["slice_code"]
    slice_code = 1
    header["slice_code"] = slice_code
    if verbose: print(header["slice_code"])

    frame1 = np.rot90(img1.get_fdata(), k=3, axes=(0, 1))

    iShape = frame1.shape
    movie = np.zeros((iShape[0], iShape[1], iShape[2], nFrames))
    movie[:, :, :, 0] = frame1
    for i in range(1, nFrames, 1):
        filename = os.path.join(data_stem, stem + str(i) + ".nii")
        img = nib.load(filename)
        frame = np.rot90(img.get_fdata(), k=3, axes=(0, 1))
        movie[:, :, :, i] = frame

    outMovie = nib.Nifti1Image(movie, affine=img1.header.get_best_affine(), header=header)
    out_name = os.path.join(data_stem, "test_movie.nii")
    nib.save(outMovie, out_name)
        

