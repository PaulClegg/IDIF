"""
file: test_Figure1.py
Functional tests leading to the components of Figure 1
"""

import pytest
import os
from matplotlib import pyplot as plt
import numpy as np

import PET_tools as tf1PT

import sirf.Reg as tf1Reg
import sirf.STIR as tf1PET
import utilities as tf1U

@pytest.mark.skip()
def test_displaySagittalUmapCut():
    #xy_size = 420.0 / 256.0 #mm
    #z_size = 3.0 # mm
    xy_size = 2.08626 #mm
    z_size = 2.03125 # mm
    aspect = z_size / xy_size
    data_stem = "/home/pclegg/devel/SIRF-SuperBuild/docker/devel/IDIF/data"
    uMap_name = "uMap_phantom.nii"
    path = os.path.join(data_stem, uMap_name)

    uMap_image = tf1Reg.ImageData(path)

    template_path = os.path.join(data_stem, "template3D.hs")
    template = tf1PET.AcquisitionData(template_path)
    im_pet = tf1PET.ImageData(template)

    print(uMap_image.dimensions())
    uMap_reshaped = tf1PT.reshapePhantomData(uMap_image, im_pet)

    uMap_arr = uMap_reshaped.as_array()
    #uMap_arr = uMap_image.as_array()
    image_shape = uMap_arr.shape
    print(image_shape)

    # display
    title = "Sagittal cut through u-map"
    y = image_shape[2] // 2
    plt.figure()
    tf1U.imshow(np.flip(uMap_arr[:, :, y], axis=0), aspect, title)
    plt.show()

    assert True

@pytest.mark.skip()
def test_displayProjectedMRI():
    data_stem = "/home/pclegg/devel/SIRF-SuperBuild/docker/devel/IDIF/data"
    mri_name = "T1_motion1_projected.nii"
    #mri_name = "T1_motion1_image.nii"
    path = os.path.join(data_stem, mri_name)

    mri_image = tf1Reg.ImageData(path)
    mri_arr = mri_image.as_array()
    image_shape = mri_arr.shape
    print(image_shape)
    print(np.abs(mri_arr.max()))
    print(np.abs(mri_arr.min()))

    # display
    title = "Transverse cut through T1 mri"
    z = image_shape[0] // 2
    z = 21 # For portal vein
    print(z)
    plt.figure()
    tf1U.imshow(mri_arr[z, :, :], aspect=None, title=title)
    plt.show()

    assert True

#@pytest.mark.skip()
def test_displayReconstructedPET():
    data_stem = "/home/pclegg/devel/SIRF-SuperBuild/docker/devel/IDIF/data"

    #PET_image_name = "pet_motion1_image.hv"
    PET_image_name = "pet_static_frame_16_image.hv"
    PET_image_path = os.path.join(data_stem, PET_image_name)

    PET_image = tf1PET.ImageData(PET_image_path)
    PET_arr = PET_image.as_array()
    image_shape = PET_arr.shape
    print(image_shape)

    # display
    title = "Transverse cut through PET"
    z = image_shape[0] // 2
    z = 69
    print(z)
    plt.figure()
    tf1U.imshow(PET_arr[z, :, :], None, title)
    plt.show()

    #for z in range(image_shape[0]):
    #    plt.figure()
    #    tf1U.imshow(PET_arr[z, :, :], None, title)
    #plt.show()

    assert True

