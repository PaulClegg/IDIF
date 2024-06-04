"""
file: PET_tools.py
To generate and reconstruct raw PET data from digital phantom
"""

import os
import numpy as np
import scipy.ndimage as sn
from matplotlib import pyplot as plt

import sirf.STIR as pPET

def create3Dtemplate(data_stem):
    template = pPET.AcquisitionData('Siemens_mMR', span=11, 
        max_ring_diff=60, view_mash_factor=1)
    out_path = os.path.join(data_stem, "template3D.hs")
    template.write(out_path)

def imageToSinogram(image_data, template, verbose=True):

    im_pet = pPET.ImageData(template)
    if verbose: 
        print("\nComparison")
        print(f"From template: {im_pet.dimensions()}")
        print(f"From phantom: {image_data.dimensions()}")

    acq_mod = pPET.AcquisitionModelUsingRayTracingMatrix()
    acq_mod.set_num_tangential_LORs(10)
    acq_mod.set_up(template, im_pet)

    reshaped = reshapePhantomData(image_data, im_pet)
    if verbose: 
        print(f"From reshaped phantom: {reshaped.dimensions()}")

    raw_pet = acq_mod.forward(reshaped)
    return raw_pet

def reshapePhantomData(original, target, verbose=True):

    orig_arr = original.as_array()
    reshaped_arr = np.zeros((orig_arr.shape[2], orig_arr.shape[0], 
        orig_arr.shape[1]))
    for z in range(orig_arr.shape[2]):
        reshaped_arr[z, :, :] = orig_arr[:, :, z]
    ln = orig_arr.shape[0]
    lslice = orig_arr.shape[2]
    ln_out = target.dimensions()[1]
    lslice_out = target.dimensions()[0]
    x_new = np.linspace(0, (ln-1), ln_out)
    y_new = np.linspace(0, (ln-1), ln_out)
    z_new = np.linspace(0, (lslice-1), lslice_out)

    z = np.zeros(lslice_out * ln_out * ln_out)
    x = np.zeros(lslice_out * ln_out * ln_out)
    y = np.zeros(lslice_out * ln_out * ln_out)
    a = 0; b=0
    for i in range(lslice_out * ln_out * ln_out):
        c = i % ln_out
        z[i] = z_new[a]
        x[i] = x_new[b]
        y[i] = y_new[c]
        if c == (ln_out - 1):
            b += 1
        if b == ln_out:
            a += 1
            b = 0
    points_out = np.array([z, x, y])

    new_arr = sn.map_coordinates(reshaped_arr, points_out)
    new_arr = new_arr.reshape((lslice_out, ln_out, ln_out))
    z_mid = new_arr.shape[0] // 2
    if verbose:
        plt.figure()
        plt.imshow(new_arr[z_mid, :, :])
        plt.show()

    # Forward
    reshaped = target.clone()
    reshaped.fill(new_arr)

    return reshaped
