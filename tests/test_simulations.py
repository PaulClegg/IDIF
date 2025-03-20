"""
file:test_simulations.py
To test creation of image data from phantom data
"""

import pytest
import os
import sys
import numpy as np
from matplotlib import pyplot as plt

sys.path.insert(0, '/home/jovyan/IDIF/src')
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
def test_convertNiftiFilesToMovie():
    stem = "static_frame_"
    data_stem = "/home/pclegg/devel/SIRF-SuperBuild/docker/devel/IDIF/data"
    tsU.convertNiftiFilesToMovie(data_stem, stem, 20)

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
    #filename = "T1_motion1_image.nii"
    filename = "reconstructed_mri_1.nii"
    data_stem = "/home/pclegg/devel/SIRF-SuperBuild/docker/devel/IDIF/data"
    datapath2 = os.path.join(data_stem, "static")
    path = os.path.join(datapath2, filename)
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
    filename = "static_frame_2.nii"
    data_stem = "/home/pclegg/devel/SIRF-SuperBuild/docker/devel/IDIF/data"
    path = os.path.join(data_stem, filename)
    phantom_data = tsU.readNiftiImageData(path)

    # For a phantom I need conversion to activity
    #image_data = tsPT.convertPhantomToActivity(phantom_data)
    # For a dynamic simulation - I don't
    image_data = phantom_data

    template_path = os.path.join(data_stem, "template3D.hs")
    template = tsPET.AcquisitionData(template_path)
    im_pet = tsPET.ImageData(template)
    print("\n im_pet.dimensions() \n", im_pet.dimensions(), "\n")
    voxels_PET = im_pet.voxel_sizes()
    print("\n im_pet.voxel_sizes() \n", voxels_PET, "\n")

    datapath2 = os.path.join(data_stem, "static")
    uMap_name = "uMap_phantom.nii"
    uMap_path = os.path.join(datapath2, uMap_name)
    uMap_image = tsReg.ImageData(uMap_path)
    print(uMap_image.dimensions())
    uMap_reshaped = tsPT.reshapePhantomData(uMap_image, im_pet)
    print(uMap_reshaped.dimensions())

    path1 = "/home/pclegg/devel/SIRF-SuperBuild/docker/devel/cvs/"
    path2 = "PPA_Anon_CAR12/Data_for_Paul/"
    cvs_path = os.path.join(path1, path2)
    name = "30001Tho_PetAcquisition_Raw_Data/attempt2a.n.hdr"
    norm_file = os.path.join(cvs_path, name)

    raw_pet = tsPT.imageToSinogram(image_data, template, uMap_reshaped, 
        norm_file, verbose=True)
    #out_name = "raw_pet_motion1.hs"
    out_name = "raw_" + filename.split(".")[0] + ".hs"
    out_path = os.path.join(data_stem, out_name)
    raw_pet.write(out_path)

    assert True

@pytest.mark.skip()
def test_reconstructRawPhantomPET():
    data_stem = "/home/pclegg/devel/SIRF-SuperBuild/docker/devel/IDIF/data"
    uMap_name = "uMap_phantom.nii"
    datapath2 = os.path.join(data_stem, "static")
    uMap_path = os.path.join(datapath2, uMap_name)
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

    #PET_name = "raw_pet_motion1.hs"
    stem = "static_frame_2"
    PET_name = "raw_" + stem + ".hs"
    PET_path = os.path.join(data_stem, PET_name)
    raw_pet = tsPET.AcquisitionData(PET_path)

    PET_image = tsPT.reconstructRawPhantomPET(raw_pet, template, uMap_reshaped, norm_file)
    #PET_out_name = "pet_motion1_image.hv"
    PET_out_name = "pet_" + stem + "_image.hv"
    PET_out_path = os.path.join(data_stem, PET_out_name)
    PET_image.write(PET_out_path)

    assert True

@pytest.mark.skip()
def test_creationOfBloodCurvesForPET():

    time = np.linspace(0.0, 3600.0, 3601)

    feng1, feng2 = tsPT.createBloodCurves(time)

    assert True

@pytest.mark.skip()
def test_creationOfLiverCurveForPET():
    time = np.linspace(0.0, 3600.0, 3601)
    feng1, feng2 = tsPT.createBloodCurves(time)

    liver = tsPT.createLiverCurve(feng1, feng2, time)

    assert True

@pytest.mark.skip()
def test_isolateLiverVessels():
    filename = "separate_veins_10.nii"
    data_stem = "/home/pclegg/devel/SIRF-SuperBuild/docker/devel/IDIF/data"
    path = os.path.join(data_stem, filename)
    phantom_data = tsU.readNiftiImageData(path)

    portal_data = tsPT.isolateLiverVessels(phantom_data)

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
    liver_full = tsPT.createLiverCurve(feng1_full, feng2_full, time)

    plt.figure()
    plt.semilogx(time, feng1_full, color="r", label="Artery")
    plt.semilogx(time, feng2_full, color="b", label="Portal Vein")
    plt.scatter(times, feng1_framed, marker="o", color="r", label="Artery framed")
    plt.scatter(times, feng2_framed, marker="o", color="b", label="Portal Vein framed")
    plt.legend()
    plt.xlabel("Time (sec)")
    plt.ylabel("Activity conc. (Bq/mL)")
    plt.show()

    #filename = "separate_veins_1.nii"
    filename = "hepatic_motion_1.nii"
    data_stem = "/home/pclegg/devel/SIRF-SuperBuild/docker/devel/IDIF/data"
    path = os.path.join(data_stem, filename)
    phantom_data = tsU.readNiftiImageData(path)
    # Next line might be redundant
    portal_data = tsPT.isolateLiverVessels(phantom_data, verbose = False)

    vein_activities = tsPT.returnFrameValues(time, feng2_full,
        times, durations)
    print(vein_activities)
    artery_activities = tsPT.returnFrameValues(time, feng1_full,
        times, durations)
    print(artery_activities)
    liver_activities = tsPT.returnFrameValues(time, liver_full,
        times, durations)
    print(liver_activities)
    start = 0.0; stop = 1200.0
    other_activities = tsPT.otherOrganRampValues(start, stop, times)
    #print(other_activities)

    frame_dim = portal_data.dimensions()
    print(frame_dim)
    dynamic_data = np.zeros((len(vein_activities), frame_dim[0], frame_dim[1]))
    i = 0
    for cnt, activity in enumerate(vein_activities):
        portal_data1 = tsPT.changeActivityInSingleRegion(portal_data, 
            105, artery_activities[cnt])
        portal_data2 = tsPT.changeActivityInSingleRegion(portal_data1, 
            7, liver_activities[cnt])
        portal_data3 = tsPT.changeRemainingActivities(portal_data2, cnt, other_activities)
        frame = tsPT.changeActivityInSingleRegion(portal_data3, 43, activity)
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
    liver_full = tsPT.createLiverCurve(feng1_full, feng2_full, time)
    vein_activities = tsPT.returnFrameValues(time, feng2_full,
        times, durations)
    artery_activities = tsPT.returnFrameValues(time, feng1_full,
        times, durations)
    liver_activities = tsPT.returnFrameValues(time, liver_full,
        times, durations)
    start = 0.0; stop = 1200.0
    other_activities = tsPT.otherOrganRampValues(start, stop, times)

    #stem = "separate_veins_"
    stem = "hepatic_motion_"
    data_stem = "/home/pclegg/devel/SIRF-SuperBuild/docker/devel/IDIF/data"
    states = 10

    for cnt, activity in enumerate(vein_activities):
        for i in range(1, states + 1, 1):
            filename = stem + str(i) + ".nii"
            path = os.path.join(data_stem, filename)
            phantom_data = tsU.readNiftiImageData(path)
            portal_data = tsPT.isolateLiverVessels(phantom_data, verbose = False)
            if i == 1:
                portal_data1 = tsPT.changeActivityInSingleRegion(portal_data, 
                    105, artery_activities[cnt])
                portal_data2 = tsPT.changeActivityInSingleRegion(portal_data1, 
                    7, liver_activities[cnt])
                portal_data3 = tsPT.changeRemainingActivities(portal_data2, 
                    cnt, other_activities)
                frame = tsPT.changeActivityInSingleRegion(portal_data3, 43, activity)
                curr_arr = frame.as_array() / states
                frame.fill(curr_arr)
            else:
                portal_data1 = tsPT.changeActivityInSingleRegion(portal_data, 
                    105, artery_activities[cnt])
                portal_data2 = tsPT.changeActivityInSingleRegion(portal_data1, 
                    7, liver_activities[cnt])
                portal_data3 = tsPT.changeRemainingActivities(portal_data2, 
                    cnt, other_activities)
                working = tsPT.changeActivityInSingleRegion(portal_data3, 43, activity)
                curr_arr = frame.as_array()
                curr_arr += working.as_array() / states
                frame.fill(curr_arr)
        out_name = os.path.join(data_stem, "frame_" + str(cnt) + ".nii")
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
    
@pytest.mark.skip()
def test_creatingMotionFreeFrames():
    
    times, durations = tsPT.returnFrameTimes()
    print(times)
    feng1_framed, feng2_framed = tsPT.createBloodCurves(times, verbose=False)

    time = np.linspace(0.0, 3600.0, 3601)
    feng1_full, feng2_full = tsPT.createBloodCurves(time, verbose=False)
    liver_full = tsPT.createLiverCurve(feng1_full, feng2_full, time)

    plt.figure()
    plt.semilogx(time, feng1_full, color="r", label="Artery")
    plt.semilogx(time, feng2_full, color="b", label="Portal Vein")
    plt.scatter(times, feng1_framed, marker="o", color="r", label="Artery framed")
    plt.scatter(times, feng2_framed, marker="o", color="b", label="Portal Vein framed")
    plt.legend()
    plt.xlabel("Time (sec)")
    plt.ylabel("Activity conc. (Bq/mL)")
    plt.show()

    filename = "hepatic_motion_1.nii"
    data_stem = "/home/pclegg/devel/SIRF-SuperBuild/docker/devel/IDIF/data"
    path = os.path.join(data_stem, filename)
    phantom_data = tsU.readNiftiImageData(path)
    # Next line might be redundant
    portal_data = tsPT.isolateLiverVessels(phantom_data, verbose = False)

    vein_activities = tsPT.returnFrameValues(time, feng2_full,
        times, durations)
    ###vein_activities = np.flip(vein_activities) # for navigation
    artery_activities = tsPT.returnFrameValues(time, feng1_full,
        times, durations)
    liver_activities = tsPT.returnFrameValues(time, liver_full,
        times, durations)
    ###liver_activities /= 10.0 # for navigation
    start = 0.0; stop = 1200.0
    other_activities = tsPT.otherOrganRampValues(start, stop, times)

    frame_dim = portal_data.dimensions()
    print(frame_dim)
    dynamic_data = np.zeros((len(vein_activities), frame_dim[0], 
        frame_dim[1]))
    i = 0
    for cnt, activity in enumerate(vein_activities):
        portal_data1 = tsPT.changeActivityInSingleRegion(portal_data, 
            105, artery_activities[cnt])
        portal_data2 = tsPT.changeActivityInSingleRegion(portal_data1, 
            7, liver_activities[cnt])
        portal_data3 = tsPT.changeRemainingActivities(portal_data2, 
            cnt, other_activities)
        frame = tsPT.changeActivityInSingleRegion(portal_data3, 43, activity)
        dynamic_data[i, :, :] = frame.as_array()[:, :, 37]
        i += 1
        out_name = os.path.join(data_stem, 
            "static_frame_" + str(cnt) + ".nii")
        frame.write(out_name)

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
def test_readExcelTAC():
    filename = "MotionStudy1_ROIs.xlsx"
    data_stem = "/home/pclegg/devel/SIRF-SuperBuild/docker/devel/IDIF/data"

    times, durations = tsPT.returnFrameTimes()
    liver_results = tsU.readExcelTAC(os.path.join(data_stem, filename), "Liver")
    pv_results = tsU.readExcelTAC(os.path.join(data_stem, filename), "PV2")
    portal_results = tsU.readExcelTAC(os.path.join(data_stem, filename), "Portal Vein")
    artery_results = tsU.readExcelTAC(os.path.join(data_stem, filename), "Artery")

    time = np.linspace(0.0, 3600.0, 3601)
    feng1_full, feng2_full = tsPT.createBloodCurves(time, verbose=False)
    liver_full = tsPT.createLiverCurve(feng1_full, feng2_full, time)
    liver_activities = tsPT.returnFrameValues(time, liver_full,
        times, durations)
    vein_activities = tsPT.returnFrameValues(time, feng2_full,
        times, durations)
    artery_activities = tsPT.returnFrameValues(time, feng1_full,
        times, durations)

    plt.figure()
    plt.title("Liver comparison")
    plt.scatter(times, liver_results[0, :], marker="o", color="red", label="Amide")
    plt.scatter(times, liver_activities, marker="+", color="blue", label="Simulated")
    plt.xlabel("Time (sec)")
    plt.ylabel("Activity conc. (Bq/mL)")
    plt.legend()
    plt.show()

    plt.figure()
    plt.title("Hepatic artery comparison")
    plt.scatter(times, artery_results[0, :], marker="o", color="red", label="Amide")
    plt.scatter(times, artery_activities, marker="+", color="blue", label="Simulated")
    plt.xlabel("Time (sec)")
    plt.ylabel("Activity conc. (Bq/mL)")
    plt.legend()
    plt.show()

    plt.figure()
    plt.title("Portal vein comparison")
    plt.scatter(times, pv_results[0, :], marker="o", color="red", label="Amide 2")
    plt.scatter(times, portal_results[0, :], marker="s", color="green", label="Amide 1")
    plt.scatter(times, vein_activities, marker="+", color="blue", label="Simulated")
    plt.xlabel("Time (sec)")
    plt.ylabel("Activity conc. (Bq/mL)")
    plt.legend()
    plt.show()

    assert True

@pytest.mark.skip()
def test_registerNifti():
    #N = 2
    path = "/home/pclegg/devel/SIRF-SuperBuild/docker/devel/IDIF/data/motion"
    ref_name = "motion_for_registration_1.nii"
    ref_file = os.path.join(path, ref_name)
    for N in range(2, 25, 1):
        flo_name = "motion_for_registration_" + str(N) + ".nii"
        flo_file = os.path.join(path, flo_name)
        

        TM = tsM.registerNifti(ref_file, flo_file, verbose=False)
        out_name = "TM_" + str(N) + "_to_1.npy"
        out_file = os.path.join(path, out_name)
        np.save(out_file, TM)

    assert True

#@pytest.mark.skip()
def test_simulatingFrameTwo():
    times, durations = tsPT.returnFrameTimes()
    print(times)
    feng1_framed, feng2_framed = tsPT.createBloodCurves(times, verbose=False)

    resp_ms = 500 / 3000.0 # length of respiratory motion state in seconds
    resp_cycle = 4 # total duration of respiratory cycle in seconds
    segment = 300.0
    samples = int((5.0 * segment / resp_ms) + 1) # 5 samples per motion state

    time = np.linspace(0.0, segment, samples) # High resolution for respiration
    feng1_full, feng2_full = tsPT.createBloodCurves(time, verbose=False)
    liver_full = tsPT.createLiverCurve(feng1_full, feng2_full, time)

    print(f"Frame 2 starts: {durations[0]}")
    print(f"Frame 2 centre: {times[1]}")
    print(f"Frame 2 duration: {durations[1]}")
    print(f"Frame 2 ends: {np.sum(durations[0:2])}")

    # I have about 5 time / activity points within each respiratory motion state
    # For the purposes of getting activity values - I just need to know
    # when respiratory motion states start and stop - I don't need the phase
    resp_times = []
    resp_dura = []
    cycles = int(segment / resp_ms)
    for i in range(cycles):
        resp_times.append((float(i) + 0.5) * resp_ms)
        resp_dura.append(resp_ms)

    liver_activities = tsPT.returnMotionStateValues(time, liver_full,
        resp_times, resp_dura)
    vein_activities = tsPT.returnMotionStateValues(time, feng2_full,
        resp_times, resp_dura)
    artery_activities = tsPT.returnMotionStateValues(time, feng1_full,
        resp_times, resp_dura)

    # I will need to know the "i" values where frame 2 begins and ends
    f2_start = durations[0]
    f2_stop = np.sum(durations[0:2])
    x = resp_times > f2_start
    if2_start = x.argmax()
    y = resp_times > f2_stop
    if2_stop = y.argmax()

    f2_liver = liver_activities[if2_start:if2_stop]
    f2_vein = vein_activities[if2_start:if2_stop]
    f2_artery = artery_activities[if2_start:if2_stop]
    f2_times = resp_times[if2_start:if2_stop]
    print(f"Motion states in frame 2: {len(f2_times)}")

    # I can create 0.5 sec sinograms covering the frame
    # Each sinogram will contain three respiratory motion states

    # At short times (first 4/5 minutes) these are corrected using MRI
    # At long times (last 45/46 minutes) these are corrected via pca on PET itself

    assert True
