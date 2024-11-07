"""
file:test_simulations.py
To test creation of image data from phantom data
"""

import pytest
import os
import numpy as np
from matplotlib import pyplot as plt

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

@pytest.mark.skip()
def test_forwardProjectStarvibeMRI():
    filename = "T1_motion1_image.nii"
    data_stem = "/home/pclegg/devel/SIRF-SuperBuild/docker/devel/IDIF/data"
    path = os.path.join(data_stem, filename)
    T1_image = tsU.readNiftiImageData(path)

    raw_name = "phantom_template.h5"
    raw_full = os.path.join(data_stem, raw_name)

    bwd_mr = tsM.forwardProjectStarvibeMRI(T1_image, raw_full, verbose=True)
    #tsU.saveGadgetronImageAsRegNifti(bwd_mr, data_stem, "T1_motion1_projected.nii")

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
    print("\n im_pet.dimensions() \n", im_pet.dimensions(), "\n")
    voxels_PET = im_pet.voxel_sizes()
    print("\n im_pet.voxel_sizes() \n", voxels_PET, "\n")

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

@pytest.mark.skip()
def test_creationOfBloodCurvesForPET():

    time = np.linspace(0.0, 3600.0, 3601)

    feng1, feng2 = tsPT.createBloodCurves(time)

    assert True

#@pytest.mark.skip()
def test_creationOfLiverCurveForPET():
    time = np.linspace(0.0, 3600.0, 3601)
    feng1, feng2 = tsPT.createBloodCurves(time)

    liver = tsPT.createLiverCurve(feng1, feng2, time)

    assert True

@pytest.mark.skip()
def test_isolatePortalVein():
    filename = "separate_veins_10.nii"
    data_stem = "/home/pclegg/devel/SIRF-SuperBuild/docker/devel/IDIF/data"
    path = os.path.join(data_stem, filename)
    phantom_data = tsU.readNiftiImageData(path)

    portal_data = tsPT.isolatePortalVein(phantom_data)

    assert True

@pytest.mark.skip()
def test_isolateHepaticArtery():
    filename = "hepatic_motion_1.nii"
    data_stem = "/home/pclegg/devel/SIRF-SuperBuild/docker/devel/IDIF/data"
    path = os.path.join(data_stem, filename)
    phantom_data = tsU.readNiftiImageData(path)

    hepatic_data = tsPT.isolateHepaticArtery(phantom_data)

    assert True

@pytest.mark.skip()
def test_creatingFrames():
    
    times, durations = tsPT.returnFrameTimes()
    print(times)
    feng1_framed, feng2_framed = tsPT.createBloodCurves(times, verbose=False)

    time = np.linspace(0.0, 3600.0, 3601)
    feng1_full, feng2_full = tsPT.createBloodCurves(time, verbose=False)

    plt.figure()
    plt.semilogx(time, feng1_full, color="r", label="Artery")
    plt.semilogx(time, feng2_full, color="b", label="Portal Vein")
    plt.scatter(times, feng1_framed, marker="o", color="r", label="Artery framed")
    plt.scatter(times, feng2_framed, marker="o", color="b", label="Portal Vein framed")
    plt.legend()
    plt.xlabel("Time (sec)")
    plt.ylabel("Activity conc. (Bq/mL)")
    plt.show()

    filename = "separate_veins_1.nii"
    data_stem = "/home/pclegg/devel/SIRF-SuperBuild/docker/devel/IDIF/data"
    path = os.path.join(data_stem, filename)
    phantom_data = tsU.readNiftiImageData(path)
    portal_data = tsPT.isolatePortalVein(phantom_data, verbose = False)

    frame_activities = tsPT.returnFrameValues(time, feng2_full,
        times, durations)
    print(frame_activities)

    frame_dim = portal_data.dimensions()
    print(frame_dim)
    dynamic_data = np.zeros((len(frame_activities), frame_dim[0], frame_dim[1]))
    i = 0
    for activity in frame_activities:
        frame = tsPT.changeActivityInSingleRegion(portal_data, 43, activity)
        dynamic_data[i, :, :] = frame.as_array()[:, :, 37]
        i += 1

    dynamic_data += 1e-6 # for log intensity scale
    c=dynamic_data[0]
    c=c.reshape(frame_dim[0], frame_dim[1]) # this is the size of my pictures
    im=plt.imshow(c, norm="log", vmin=0.01, vmax=225.0)
    for row in dynamic_data:
        row=row.reshape(frame_dim[0], frame_dim[1]) # this is the size of my pictures
        im.set_data(row)
        plt.pause(0.5)
    plt.show()

    assert True

@pytest.mark.skip()
def test_averagingAcrossMotionStates():
    times, durations = tsPT.returnFrameTimes()
    feng1_framed, feng2_framed = tsPT.createBloodCurves(times, verbose=False)
    time = np.linspace(0.0, 3600.0, 3601)
    feng1_full, feng2_full = tsPT.createBloodCurves(time, verbose=False)
    frame_activities = tsPT.returnFrameValues(time, feng2_full,
        times, durations)

    stem = "separate_veins_"
    data_stem = "/home/pclegg/devel/SIRF-SuperBuild/docker/devel/IDIF/data"
    states = 10

    for j, activity in enumerate(frame_activities):
        for i in range(1, states + 1, 1):
            filename = stem + str(i) + ".nii"
            path = os.path.join(data_stem, filename)
            phantom_data = tsU.readNiftiImageData(path)
            portal_data = tsPT.isolatePortalVein(phantom_data, verbose = False)
            if i == 1:
                frame = tsPT.changeActivityInSingleRegion(portal_data, 43, activity)
                curr_arr = frame.as_array() / states
                frame.fill(curr_arr)
            else:
                working = tsPT.changeActivityInSingleRegion(portal_data, 43, activity)
                curr_arr = frame.as_array()
                curr_arr += working.as_array() / states
                frame.fill(curr_arr)
        out_name = os.path.join(data_stem, "frame_" + str(j) + ".nii")
        frame.write(out_name)

    assert True

@pytest.mark.skip()
def test_displayNiftiAverageFrame():
    filename = "frame_2.nii"
    data_stem = "/home/pclegg/devel/SIRF-SuperBuild/docker/devel/IDIF/data"
    path = os.path.join(data_stem, filename)
    image = tsU.readNiftiImageData(path)

    tsU.displayRegImageData(image, title="Time averaged kinetic data")

    assert True
    
@pytest.mark.skip()
def test_displayMovieOfAverageFrames():
    frames = int(20)
    frame_dim = int(256)
    data_stem = "/home/pclegg/devel/SIRF-SuperBuild/docker/devel/IDIF/data"
    dynamic_data = np.zeros((frames, frame_dim, frame_dim))
    for im in range(frames):
        filename = "frame_" + str(im) + ".nii"
        path = os.path.join(data_stem, filename)
        image = tsU.readNiftiImageData(path)
        dynamic_data[im, :, :] = image.as_array()[:, :, 36]

    dynamic_data += 1e-6 # for log intensity scale
    c=dynamic_data[0]
    c=c.reshape(frame_dim, frame_dim) # this is the size of my pictures
    im=plt.imshow(c, norm="log", vmin=0.01, vmax=225.0)
    for row in dynamic_data:
        row=row.reshape(frame_dim, frame_dim) # this is the size of my pictures
        im.set_data(row)
        plt.pause(0.5)
    plt.show()

    assert True
    
