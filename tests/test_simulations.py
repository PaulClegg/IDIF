"""
file:test_simulations.py
To test creation of image data from phantom data
"""

import pytest
import os

import utilities as tsU
import uMap as tsP
import MRI as tsM

@pytest.mark.skip()
def test_readingNiftiImage():
    filename = "phantom_motion1.nii"
    data_stem = "/home/pclegg/devel/SIRF-SuperBuild/docker/devel/IDIF/data"
    path = os.path.join(data_stem, filename)
    image = tsU.readNiftiImageData(path)

    assert True

@pytest.mark.skip()
def test_displayNiftiImage():
    filename = "phantom_motion1.nii"
    data_stem = "/home/pclegg/devel/SIRF-SuperBuild/docker/devel/IDIF/data"
    path = os.path.join(data_stem, filename)
    image = tsU.readNiftiImageData(path)

    tsU.displayRegImageData(image, title="Initial phantom data")

    assert True

@pytest.mark.skip()
def test_convertToUmap():
    filename = "phantom_motion1.nii"
    data_stem = "/home/pclegg/devel/SIRF-SuperBuild/docker/devel/IDIF/data"
    path = os.path.join(data_stem, filename)
    image = tsU.readNiftiImageData(path)

    uMap_image = tsP.convertPhantomToUmap(image)
    uMap_name = "uMap_phantom.nii"
    path = os.path.join(data_stem, uMap_name)
    uMap_image.write(path)

    tsU.displayRegImageData(uMap_image, title="uMap of phantom data")

    assert True

@pytest.mark.skip()
def test_readTissueProperties():
    filename = "tissueproperty.m"
    data_stem = "/home/pclegg/devel/SIRF-SuperBuild/docker/devel/IDIF/data"
    path = os.path.join(data_stem, filename)

    properties = tsM.readTissueProperties(path)

    assert True

@pytest.mark.skip()
def test_convertPhantomToT1Values():
    filename = "phantom_motion1.nii"
    data_stem = "/home/pclegg/devel/SIRF-SuperBuild/docker/devel/IDIF/data"
    path = os.path.join(data_stem, filename)
    image = tsU.readNiftiImageData(path)

    prop_name = "tissueproperty.m"
    path = os.path.join(data_stem, prop_name)
    properties = tsM.readTissueProperties(path, verbose=False)

    T1_image = tsM.convertPhantomToT1Values(image, properties)
    tsU.displayRegImageData(image, title="Initial phantom data")
    tsU.displayRegImageData(T1_image, title="T1 weighted phantom data")
    outname = "T1_" + filename.split("_")[1].split(".nii")[0] + "_image.nii"
    path = os.path.join(data_stem, outname)
    T1_image.write(path)

    assert True

#@pytest.mark.skip()
def test_forwardProjectRadialMRI():
    filename = "T1_motion1_image.nii"
    data_stem = "/home/pclegg/devel/SIRF-SuperBuild/docker/devel/IDIF/data"
    path = os.path.join(data_stem, filename)
    T1_image = tsU.readNiftiImageData(path)

    raw_mri = tsM.forwardProjectRadialMRI(T1_image, data_stem, verbose=True)

    assert True

