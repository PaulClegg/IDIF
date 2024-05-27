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
    data_stem = "/home/pclegg/devel/SIRF-SuperBuild/docker/devel/IDIF/data"
    uMap_name = "uMap_phantom.nii"
    path = os.path.join(data_stem, uMap_name)

    uMap_image = tf1Reg.ImageData(path)
    uMap_arr = uMap_image.as_array()

    # double the length along S-I axis
    image_shape = uMap_arr.shape
    print(image_shape)
    new_shape = (image_shape[0], image_shape[1], 2 * image_shape[2])
    new_image = np.zeros(new_shape)
    for z in range(new_shape[2]):
        half_int = z // 2
        new_image[:, :, z] = uMap_arr[:, :, half_int]

    # display
    title = "Sagittal cut through u-map"
    y = image_shape[1] // 2
    plt.figure()
    tf1U.imshow(np.rot90(new_image[:, y, :], axes=(0,1)), title)
    plt.show()

    assert True

