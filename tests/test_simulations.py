"""
file:test_simulations.py
To test creation of image data from phantom data
"""

import pytest
import os
import numpy as np

import utilities as tsU
import uMap as tsP
import MRI as tsM
import PET_tools as tsPT

import sirf.STIR as tsPET
import sirf.Reg as tsReg

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
def test_forwardProjectStarvibeMRI():
    filename = "T1_motion1_image.nii"
    data_stem = "/home/pclegg/devel/SIRF-SuperBuild/docker/devel/IDIF/data"
    path = os.path.join(data_stem, filename)
    T1_image = tsU.readNiftiImageData(path)

    raw_name = "starVIBE.dat"
    raw_path = "/home/pclegg/devel/SIRF-SuperBuild/docker/devel/cvs"
    raw_full = os.path.join(raw_path, raw_name)

    #raw_mri = tsM.forwardProjectStarvibeMRI(T1_image, raw_path, raw_full,
    bwd_mr = tsM.forwardProjectStarvibeMRI(T1_image, raw_path, raw_full,
        verbose=True)
    bwd_arr = np.abs(bwd_mr.as_array())
    print(bwd_arr.shape)
    path = os.path.join(raw_path, "starVIBE_twelve_1_real.nii")
    cvs_image = tsU.readNiftiImageData(path)
    T1_out = cvs_image.clone()
    print(T1_out.dimensions())
    T1_out.fill(bwd_arr)
    T1_out.write(os.path.join(data_stem, "T1_motion1_projected.nii"))

    assert True

@pytest.mark.skip()
def test_create3Dtemplate():
    data_stem = "/home/pclegg/devel/SIRF-SuperBuild/docker/devel/IDIF/data"
    tsPT.create3Dtemplate(data_stem)

    assert True

@pytest.mark.skip()
def test_imageToSinogram():
    filename = "phantom_motion1.nii"
    data_stem = "/home/pclegg/devel/SIRF-SuperBuild/docker/devel/IDIF/data"
    path = os.path.join(data_stem, filename)
    phantom_data = tsU.readNiftiImageData(path)

    image_data = tsPT.convertPhantomToActivity(phantom_data)

    template_path = os.path.join(data_stem, "template3D.hs")
    template = tsPET.AcquisitionData(template_path)
    im_pet = tsPET.ImageData(template)

    uMap_name = "uMap_phantom.nii"
    uMap_path = os.path.join(data_stem, uMap_name)
    uMap_image = tsReg.ImageData(uMap_path)
    print(uMap_image.dimensions())
    uMap_reshaped = tsPT.reshapePhantomData(uMap_image, im_pet)
    print(uMap_reshaped.dimensions())

    path1 = "/home/pclegg/devel/SIRF-SuperBuild/docker/devel/cvs/"
    path2 = "PPA_Anon_CAR12/Data_for_Paul/"
    cvs_path = os.path.join(path1, path2)
    name = "30001Tho_PetAcquisition_Raw_Data/attempt2a.n.hdr"
    norm_file = os.path.join(cvs_path, name)

    raw_pet = tsPT.imageToSinogram(image_data, template, uMap_reshaped, norm_file, verbose=True)
    out_name = "raw_pet_motion1.hs"
    out_path = os.path.join(data_stem, out_name)
    raw_pet.write(out_path)

    assert True

@pytest.mark.skip()
def test_reconstructRawPhantomPET():
    data_stem = "/home/pclegg/devel/SIRF-SuperBuild/docker/devel/IDIF/data"
    uMap_name = "uMap_phantom.nii"
    uMap_path = os.path.join(data_stem, uMap_name)
    uMap_image = tsReg.ImageData(uMap_path)

    template_path = os.path.join(data_stem, "template3D.hs")
    template = tsPET.AcquisitionData(template_path)
    im_pet = tsPET.ImageData(template)

    print(uMap_image.dimensions())
    uMap_reshaped = tsPT.reshapePhantomData(uMap_image, im_pet)
    print(uMap_reshaped.dimensions())

    path1 = "/home/pclegg/devel/SIRF-SuperBuild/docker/devel/cvs/"
    path2 = "PPA_Anon_CAR12/Data_for_Paul/"
    cvs_path = os.path.join(path1, path2)
    name = "30001Tho_PetAcquisition_Raw_Data/attempt2a.n.hdr"
    norm_file = os.path.join(cvs_path, name)

    PET_name = "raw_pet_motion1.hs"
    PET_path = os.path.join(data_stem, PET_name)
    raw_pet = tsPET.AcquisitionData(PET_path)

    PET_image = tsPT.reconstructRawPhantomPET(raw_pet, template, uMap_reshaped, norm_file)
    PET_out_name = "pet_motion1_image.hv"
    PET_out_path = os.path.join(data_stem, PET_out_name)
    PET_image.write(PET_out_path)

    assert True

