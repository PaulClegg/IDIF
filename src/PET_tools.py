"""
file: PET_tools.py
To generate and reconstruct raw PET data from digital phantom
"""

import os
import numpy as np
import scipy.ndimage as sn
from matplotlib import pyplot as plt

import sirf.STIR as pPET
import sirf.Reg as pReg

def create3Dtemplate(data_stem):
    template = pPET.AcquisitionData('Siemens_mMR', span=11, 
        max_ring_diff=60, view_mash_factor=1)
    out_path = os.path.join(data_stem, "template3D.hs")
    template.write(out_path)

def convertPhantomToActivity(phantom_data, verbose=True):
    # Data from Christensen et al. Molecular Imaging, 16, 1 (2017)
    activities = [["Bone", 1, 2.4], # 11-C Palmitate - fatty acid metabolism tracer
                    ["Fat", 2, 0.0],
                    ["Skin", 3, 1.0],
                    ["Colon", 4, 1.6],
                    ["Gastro", 5, 2.2],
                    ["Pancreas", 6, 3.0], 
                    ["Liver", 7, 27.5],
                    ["Muscle", 8, 2.8],
                    ["Gallbladder", 9, 4.9],
                    ["Adrenalgland", 10, 3.4],
                    ["Vein", 11, 0.0],
                    ["Kidneys", 12, 5.0],
                    ["Spleen", 13, 4.2],
                    ["Artery", 14, 0.0],
                    ["Ureter", 15, 1.5]]
    image_data = phantom_data.clone()
    phantom_arr = phantom_data.as_array()
    image_arr = np.zeros(phantom_arr.shape)

    z_mid = phantom_arr.shape[0] // 2
    if verbose:
        plt.figure()
        plt.imshow(phantom_arr[z_mid, :, :])
        plt.show()

    for i in range(len(activities)):
        image_arr[phantom_arr == (i + 1)] = activities[i][2]

    if verbose:
        plt.figure()
        plt.imshow(image_arr[z_mid, :, :])
        plt.show()

    image_data.fill(image_arr)

    return image_data

def otherOrganRampValues(start, stop, times):
    activities = [["Bone", 1, 2.4],
                    ["Fat", 2, 0.0],
                    ["Skin", 3, 1.0],
                    ["Colon", 4, 1.6],
                    ["Gastro", 5, 2.2],
                    ["Pancreas", 6, 3.0], 
                    ["Liver", 7, 27.5],
                    ["Muscle", 8, 2.8],
                    ["Gallbladder", 9, 4.9],
                    ["Adrenalgland", 10, 3.4],
                    ["Vein", 11, 0.0],
                    ["Kidneys", 12, 5.0],
                    ["Spleen", 13, 4.2],
                    ["Artery", 14, 0.0],
                    ["Ureter", 15, 1.5]]
    other_activities = np.zeros((len(activities), len(times)))
    for iTime, time in enumerate(times):
        fraction = abs((time - start) / (stop - start))
        for iOrgan in range(len(activities)):
            if time < stop:
                other_activities[iOrgan, iTime] = fraction
            else:
                other_activities[iOrgan, iTime] = 1.0
            other_activities[iOrgan, iTime] *= activities[iOrgan][2]

    return other_activities

def changeRemainingActivities(phantom_data, iTime, other_activities):
    phantom_arr = phantom_data.as_array()
    for iOrgan in range(len(other_activities)):
        phantom_arr[phantom_arr == (iOrgan + 1)] = other_activities[iOrgan, iTime]

    active = phantom_data.clone()
    active.fill(phantom_arr)
    return active

def isolateLiverVessels(phantom_data, verbose=True):
    phantom_arr = phantom_data.as_array()
    print(f"Max value = {phantom_arr.max()}")
    phantom_arr[phantom_arr == 33] = -5
    phantom_arr[phantom_arr == 105] = -10
    phantom_arr[phantom_arr > 100] = 14 # Original artery label
    phantom_arr[phantom_arr > 20] = 11 # Original vein label
    phantom_arr[phantom_arr == -5] = 43
    phantom_arr[phantom_arr == -10] = 105
    ### Region 43 appears to be the portal vein ###

    if verbose:
        for i in range(30, 40, 1):
            plt.figure()
            plt.imshow(phantom_arr[:, :, i])
            
        plt.show()

    portal_data = phantom_data.clone()
    portal_data.fill(phantom_arr)

    return portal_data

def isolateHepaticArtery(phantom_data, verbose=True):
    phantom_arr = phantom_data.as_array()
    print(f"Max value = {phantom_arr.max()}")
    phantom_arr[phantom_arr == 105] = -5
    phantom_arr[phantom_arr > 100] = 14 # Original artery label
    phantom_arr[phantom_arr == -5] = 105
    ### Region 105 appears to be the hepatic artery + larger structures ###

    if verbose:
        for i in range(30, 40, 1):
            plt.figure()
            plt.imshow(phantom_arr[:, :, i])
            
        plt.show()

    hepatic_data = phantom_data.clone()

    return hepatic_data

def changeActivityInSingleRegion(phantom_data, region, newActivity):
    phantom_arr = phantom_data.as_array()
    phantom_arr[phantom_arr == region] = newActivity

    new_phantom = phantom_data.clone()
    new_phantom.fill(phantom_arr)

    return new_phantom

def recordSinogram(image_data, data_stem, seg_no, frame_no):
    separate = data_stem.split("/")[1:-1]
    cvs_path = "/" + separate[0] + "/" + separate[1] + "/" +\
            separate[2] + "/" + separate[3]
    template_path = os.path.join(cvs_path, "template3D.hs")
    template = pPET.AcquisitionData(template_path)
    im_pet = pPET.ImageData(template)
    print("\n im_pet.dimensions() \n", im_pet.dimensions(), "\n")
    voxels_PET = im_pet.voxel_sizes()
    print("\n im_pet.voxel_sizes() \n", voxels_PET, "\n")

    datapath2 = os.path.join(cvs_path, "static")
    uMap_name = "uMap_phantom.nii"
    uMap_path = os.path.join(datapath2, uMap_name)
    uMap_image = pReg.ImageData(uMap_path)
    print(uMap_image.dimensions())
    uMap_reshaped = reshapePhantomData(uMap_image, im_pet)
    print(uMap_reshaped.dimensions())

    name = "attempt2a.n.hdr"
    norm_file = os.path.join(cvs_path, name)

    raw_pet = imageToSinogram(image_data, template, uMap_reshaped,
        norm_file, verbose=True)
    out_name = "raw_frame" + str(frame_no) + "_seg" + str(seg_no) + ".hs"
    out_path = os.path.join(data_stem, out_name)
    raw_pet.write(out_path)

def imageToSinogram(image_data, template, attn_image, norm_file, verbose=True):

    im_pet = pPET.ImageData(template)
    if verbose: 
        print("\nComparison")
        print(f"From template: {im_pet.dimensions()}")
        print(f"From phantom: {image_data.dimensions()}")

    acq_model = pPET.AcquisitionModelUsingRayTracingMatrix()
    acq_model.set_num_tangential_LORs(10)
    acq_model.set_up(template, im_pet)

    asm_norm = pPET.AcquisitionSensitivityModel(norm_file)
    asm_norm.set_up(template)
    acq_model.set_acquisition_sensitivity(asm_norm)

    # create attenuation factors
    asm_attn = pPET.AcquisitionSensitivityModel(attn_image, acq_model)
    # converting attenuation image into attenuation factors
    # (one for every bin)
    asm_attn.set_up(template)
    ac_factors = template.get_uniform_copy(value=1)
    print('applying attenuation (please wait, may take a while)...')
    asm_attn.unnormalise(ac_factors)
    asm_attn = pPET.AcquisitionSensitivityModel(ac_factors)

    # chain attenuation and ECAT8 normalisation
    asm = pPET.AcquisitionSensitivityModel(asm_norm, asm_attn)
    asm.set_up(template)

    acq_model.set_acquisition_sensitivity(asm)

    reshaped = reshapePhantomData(image_data, im_pet)
    if verbose: 
        print(f"From reshaped phantom: {reshaped.dimensions()}")

    raw_pet = acq_model.forward(reshaped)
    return raw_pet

def reshapePhantomData(original, target, verbose=True):

    orig_arr = original.as_array()
    # Here pad data with zeros - to get to correct size
    padded_arr = np.zeros((362, 362, 86)) # to give correct sizes
    x_start = 53; y_start = 53; z_start = 11
    Dx = orig_arr.shape[0]; Dy = orig_arr.shape[1]; Dz = orig_arr.shape[2]
    padded_arr[x_start:(Dx + x_start), y_start:(Dy + y_start), z_start:(Dz + z_start)] =\
        orig_arr[:, :, :]

    reshaped_arr = np.zeros((padded_arr.shape[2], padded_arr.shape[0], 
        padded_arr.shape[1]))
    for z in range(padded_arr.shape[2]):
        reshaped_arr[z, :, :] = padded_arr[:, :, z]
    ln = padded_arr.shape[0]
    lslice = padded_arr.shape[2]
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

def reconstructRawPhantomPET(acq_data, template, attn_image, norm_file):
    nxny = (285,285)
    image = acq_data.create_uniform_image(1.0, nxny)

    acq_model = pPET.AcquisitionModelUsingRayTracingMatrix()
    acq_model.set_num_tangential_LORs(10)

    asm_norm = pPET.AcquisitionSensitivityModel(norm_file)
    asm_norm.set_up(acq_data)
    acq_model.set_acquisition_sensitivity(asm_norm)

    # create attenuation factors
    asm_attn = pPET.AcquisitionSensitivityModel(attn_image, acq_model)
    # converting attenuation image into attenuation factors
    # (one for every bin)
    asm_attn.set_up(acq_data)
    ac_factors = acq_data.get_uniform_copy(value=1)
    print('applying attenuation (please wait, may take a while)...')
    asm_attn.unnormalise(ac_factors)
    asm_attn = pPET.AcquisitionSensitivityModel(ac_factors)

    # chain attenuation and ECAT8 normalisation
    asm = pPET.AcquisitionSensitivityModel(asm_norm, asm_attn)
    asm.set_up(acq_data)

    acq_model.set_acquisition_sensitivity(asm)

    # define objective function to be maximized as
    # Poisson logarithmic likelihood (with linear model for mean)
    obj_fun = pPET.make_Poisson_loglikelihood(acq_data)
    obj_fun.set_acquisition_model(acq_model)

    # select Ordered Subsets Maximum A-Posteriori One Step Late as the
    # reconstruction algorithm (since we are not using a penalty,
    # or prior, in this example, we actually run OSEM);
    # this algorithm does not converge to the maximum of the objective
    # function but is used in practice to speed-up calculations
    # See the reconstruction demos for more complicated examples
    num_subsets = 1 #21 1
    num_subiterations = 12 #3 
    recon = pPET.OSMAPOSLReconstructor()
    recon.set_objective_function(obj_fun)
    recon.set_num_subsets(num_subsets)
    recon.set_num_subiterations(num_subiterations)

    # set up the reconstructor based on a sample image
    # (checks the validity of parameters, sets up objective function
    # and other objects involved in the reconstruction, which involves
    # computing/reading sensitivity image etc etc.)
    print('setting up, please wait...')
    recon.set_up(image)

    # set the initial image estimate
    recon.set_current_estimate(image)

    # reconstruct
    print('reconstructing, please wait...')
    recon.process()
    out = recon.get_output()

    return out

def createBloodCurves(time, verbose=True):

    t0 = 20.0
    A1 = 0.1
    A2 = 3.0
    A3 = 1.0
    l1 = 0.0001
    l2 = 0.01
    l3 = 0.1 
    l4 = 0.5
    assert (A1 + A2 + A3) * l4 - A1 * l1 - A2 * l2 - A3 * l3 > 0.0, "Unphysiological!"
    feng1 = A1 * np.exp(-l1 * (time - t0)) + A2 * np.exp(-l2 * (time - t0))
    feng1 += A3 * np.exp(-l3 * (time - t0)) -(A1 + A2 + A3) * np.exp(-l4 * (time - t0))
    feng1[time < t0] = 0.0
    feng1 *= 120.0

    A1 = 0.1
    A2 = 3.0
    A3 = 1.0
    l1 = 0.0001
    l2 = 0.01
    l3 = 0.1 
    l4 = 0.07
    assert (A1 + A2 + A3) * l4 - A1 * l1 - A2 * l2 - A3 * l3 > 0.0, "Unphysiological!"
    feng2 = A1 * np.exp(-l1 * (time - t0)) + A2 * np.exp(-l2 * (time - t0))
    feng2 += A3 * np.exp(-l3 * (time - t0)) -(A1 + A2 + A3) * np.exp(-l4 * (time - t0))
    feng2[time < t0] = 0.0
    feng2 *= 120.0

    if verbose:
        plt.figure()
        plt.plot(time, feng1, color="r", label="Artery")
        plt.plot(time, feng2, color="b", label="Portal Vein")
        plt.xlim([0.0, 600.0])
        plt.legend()
        plt.xlabel("Time (sec)")
        plt.ylabel("Activity conc. (Bq/mL)")
        plt.show()

    return feng1, feng2

def createLiverCurve(feng1, feng2, time, verbose = True):
    liver = np.zeros(len(feng1))

    blood = 0.75 * feng2 + 0.25 * feng1
    A = 0.9; B = 0.0005; C = 0.1; D = 1E-4
    residue = A * np.exp(-B * time) + C * np.exp(-D * time)
    long_blood = np.zeros(3 * len(blood))
    long_blood[len(blood):(2 * len(blood))] = blood

    liver = np.convolve(residue, long_blood, mode="valid")[0:len(time)]
    liver *= 200.0 / liver.max()

    if verbose:
        print("\n")
        print(f"Length liver: {len(liver)}")
        print(f"Length time: {len(time)}")

        plt.figure()
        plt.plot(time, blood, color="b", label="Blood")
        plt.plot(time, residue * (blood.max() / residue.max()), 
            color="r", label="Residue")
        plt.plot(time, liver, color="g", label="Liver")
        plt.legend()
        plt.xlabel("Time (sec)")
        plt.ylabel("Activity conc. (Bq/mL)")
        plt.show()

    return liver

def returnFrameTimes():

    times = np.zeros(20)
    durations = np.zeros(20)
    # Framing from: Iozzo et al GASTROENTEROLOGY 2010;139:846–856
    durations[0:8] = 15.0
    durations[8:10] = 30.0
    durations[10:12] = 120.0
    durations[12] = 180.0
    durations[13:19] = 300.0
    durations[19] = 600.0

    print(f"Scan duration: {sum(durations)}")
    
    previous = 0.0
    for i, full in enumerate(durations):
        times[i] = previous + full / 2.0
        previous += full

    return times, durations

def returnFrameValues(full_time, full_activity, frame_times, 
    frame_durations, verbose=True):

    assert (full_time[1] - full_time[0]) == 1.0, "Require one point every second"

    frame_activities = np.zeros(len(frame_times))
    for i, t in enumerate(frame_times):
        start = int(frame_times[i] - (frame_durations[i] / 2.0))
        stop = int(frame_times[i] + (frame_durations[i] / 2.0))
        frame_activities[i] = np.mean(full_activity[start:stop])
        

    if verbose:
        plt.figure()
        plt.semilogx(full_time, full_activity, color="r", label="Continuous")
        plt.scatter(frame_times, frame_activities, marker="o", color="r", label="Framed")
        plt.legend()
        plt.xlabel("Time (sec)")
        plt.ylabel("Activity conc. (Bq/mL)")
        plt.show()

    return frame_activities

def returnMotionStateValues(full_time, full_activity, resp_times, 
    resp_durations, verbose=True):

    resp_activities = np.zeros(len(resp_times))
    for i, t in enumerate(resp_times):
        start = t - (resp_durations[i] / 2.0)
        stop = t + (resp_durations[i] / 2.0)
        # need to find the start and stop indices
        x = np.array(full_time) > start
        istart = x.argmax()
        y = np.array(full_time) > stop
        istop = y.argmax()
        resp_activities[i] = np.mean(full_activity[istart:istop])

    if verbose:
        plt.figure()
        plt.plot(full_time, full_activity, color="r", label="Continuous")
        plt.scatter(resp_times, resp_activities, marker="o", color="r", label="Framed")
        plt.legend()
        plt.xlabel("Time (sec)")
        plt.ylabel("Activity conc. (Bq/mL)")
        plt.show()

    return resp_activities
