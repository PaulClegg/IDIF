"""
file: test_Figure1.py
Functional tests leading to the components of Figure 1
"""

import pytest
import os
from matplotlib import pyplot as plt
import numpy as np

import sirf.Reg as tf1Reg
import utilities as tf1U

#@pytest.mark.skip()
def test_displaySagittalUmapCut():
    xy_size = 420.0 / 256.0 #mm
    z_size = 3.0 # mm
    aspect = z_size / xy_size
    data_stem = "/home/pclegg/devel/SIRF-SuperBuild/docker/devel/IDIF/data"
    uMap_name = "uMap_phantom.nii"
    path = os.path.join(data_stem, uMap_name)

    uMap_image = tf1Reg.ImageData(path)
    uMap_arr = uMap_image.as_array()
    image_shape = uMap_arr.shape
    print(image_shape)

    # display
    title = "Sagittal cut through u-map"
    y = image_shape[1] // 2
    plt.figure()
    tf1U.imshow(np.rot90(uMap_arr[:, y, :], axes=(0,1)), aspect, title)
    plt.show()

    assert True

