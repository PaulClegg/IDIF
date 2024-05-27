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

#@pytest.mark.skip()
def test_readTissueProperties():
    filename = "tissueproperty.m"
    data_stem = "/home/pclegg/devel/SIRF-SuperBuild/docker/devel/IDIF/data"
    path = os.path.join(data_stem, filename)

    properties = tsM.readTissueProperties(path)

    assert True

