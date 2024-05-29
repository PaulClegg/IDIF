"""
file: MRI.py
To create and manipulate MRI data from the phantom images
"""

import os
import numpy as np
from matplotlib import pyplot as plt

import sirf.Gadgetron as mMR

def convertPhantomToT1Values(ph_image, properties, verbose=True):
    # from inspection of UoE StarVIBE data
    scale = 1.0E-6

    T1_image = ph_image.clone()
    image_arr = ph_image.as_array()
    [name, num, T1, T2, ADC, PDFF] = properties

    for i in range(len(num)):
        image_arr[image_arr == num[i]] = float(T1[i]) * scale
        
    T1_image.fill(image_arr)

    return T1_image

def readTissueProperties(filename, verbose=True):

    if verbose: print(filename)

    name = []; num = []; T1 = []; T2 = []; ADC = []; PDFF = []
    line_num = 0
    with open(filename) as file:
        for line in file:
            if line_num > 11: 
                vals = divideLineIntoChunks(line.rstrip(), verbose)
                name.append(vals[0])
                num.append(vals[1])
                T1.append(vals[2])
                T2.append(vals[3])
                ADC.append(vals[4])
                PDFF.append(vals[5])
            line_num += 1

    properties = [name, num, T1, T2, ADC, PDFF]
    return properties

def divideLineIntoChunks(line, verbose=True):
    # name
    initial = line.split("name'")[1]
    initial = initial.split(",'")[1]
    name = initial.split("'")[0]
    if verbose: print(name)
    # num
    initial = line.split("=")[0]
    initial = initial.split("(")[1]
    num = int(initial.split(")")[0])
    if verbose: print(num)
    # T1
    initial = line.split("'T1',")[1]
    T1 = int(initial.split(",")[0])
    if verbose: print(T1)
    # T2
    initial = line.split("'T2',")[1]
    T2 = int(initial.split(",")[0])
    if verbose: print(T2)
    # ADC
    initial = line.split("'ADC',")[1]
    ADC = float(initial.split("*10^-6")[0])
    ADC = ADC * 1E-6
    if verbose: print(ADC)
    # PDFF
    initial = line.split("'PDFF',")[1]
    PDFF = float(initial.split("/100")[0])
    if verbose: print(PDFF)

    return [name, num, T1, T2, ADC, PDFF]

def forwardProjectRadialMRI(MRI_image, data_path, verbose=True):
    filename = "3D_RPE_Lowres.h5"
    path = os.path.join(data_path, filename)
    mr_acq = mMR.AcquisitionData(path, False, 
        ignored=mMR.IgnoreMask(19))

    ### Following Christoph's approach 
    pe_ky = mr_acq.get_ISMRMRD_info('kspace_encode_step_1')
    print(len(pe_ky))
    print(np.max(pe_ky))
    print(np.where(pe_ky == (np.max(pe_ky)+1)//2))
    print("\n\n")
    ###
    # for radial
    processed_data = mr_acq
    if verbose:
        print(f"Is undersampled? {mr_acq.is_undersampled()}")
        print(mr_acq.dimensions())

    processed_data = mMR.set_radial2D_trajectory(processed_data)
    if verbose:
        traj = np.transpose(mMR.get_data_trajectory(processed_data))
        print("--- traj shape is {}".format(traj.shape))
        size = [2] * len(traj[0, 0:4608])
        plt.figure()
        plt.scatter(traj[0,0:4608], traj[1,0:4608], size, marker='.')
        # line 1 slice 1
        plt.scatter(traj[0,0:128], traj[1,0:128], marker='.') 
        # There are 72 radial lines per slice
        # line 1 slice 2
        plt.scatter(traj[0,9216:9344], traj[1,9216:9344], marker='.') 
        plt.axis("equal")
        plt.axis("off")
        plt.show()
    # Pass the radial trajectory in the shape (4608, 128, 2)
    red_traj = np.zeros((4608, 128, 2))
    j = 0
    last_i = 0
    for i in range(128, 589952, 128):
        red_traj[j, 0:128, 0] = traj[0, last_i:i]
        red_traj[j, 0:128, 1] = traj[1, last_i:i]
        j += 1
        last_i = i
    processed_data = mMR.set_radial2D_trajectory(processed_data, red_traj)

    # sort processed acquisition data
    print('---\n sorting acquisition data...')
    processed_data.sort()

    print("\nDimension problem")
    print(f"Length of trajectory: {traj.shape[1]}")
    product = processed_data.dimensions()[0] * processed_data.dimensions()[2]
    print(f"Product of no. spokes and spoke length: {product}")
    print(f"Trajectory - 128: {traj.shape[1] - 128}")
    print("\n\n")

    print('---\n computing coil sensitivity maps...')
    csms = mMR.CoilSensitivityData()
    csms.smoothness = 10
    csms.calculate(processed_data)
    # smooth coil sensitivity
    csms = csms.sqrt()
    csms = csms.sqrt()
    csms = csms.sqrt()

    raw_mri = mMR.AcquisitionData()
    return raw_mri
