"""
file: MRI.py
To create and manipulate MRI data from the phantom images
"""

import os
import numpy as np
import scipy.ndimage as sn
from matplotlib import pyplot as plt

import utilities as mU

import sirf.Gadgetron as mMR
import sirf.Reg as mReg

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

def forwardProjectStarvibeMRI(MRI_image, acq_file, verbose=True):
    if verbose:
        print("\nOriginal phantom image")
        orig_arr = MRI_image.as_array()
        print(orig_arr.shape)
        z = orig_arr.shape[2] // 2
        plt.figure()
        plt.imshow(orig_arr[:, :, z])
        plt.show()
    # I need an acquisition model for the StarVIBE data
    acq_data = mMR.AcquisitionData(acq_file, False,
        ignored=mMR.IgnoreMask(19))
    acq_data.sort_by_time()
   
    if verbose:
        print(acq_data.get_header())

    # Now we create the trajectory and set it
    ktraj = calc_rad_traj_golden(acq_data)
    mMR.set_radial2D_trajectory(acq_data, ktraj)

    print("\n\n")
    print(acq_data.dimensions())
    print(f"1600 x 38 = {1600*38}")
    print(acq_data.check_traj_type("radial"))

    csm = mMR.CoilSensitivityData()
    csm.smoothness = 100
    csm.calculate(acq_data)
    print("\nCoil sensitivity matrix")
    print(csm.dimensions())
    print(f"Real? {csm.is_real()}")

    acq_mod = mMR.AcquisitionModel(acqs=acq_data, imgs=csm)
    acq_mod.set_coil_sensitivity_maps(csm)

    print("\nPhantom image")
    print(MRI_image.dimensions())
    # Match dimensions of phantom image with that of the template data
    MRI_arr = MRI_image.as_array()
    rMRI_arr = np.zeros((MRI_arr.shape[2], MRI_arr.shape[0], 
        MRI_arr.shape[1]))
    for z in range(MRI_arr.shape[2]):
        rMRI_arr[z, :, :] = MRI_arr[:, :, z]
    ln = MRI_arr.shape[0]
    lslice = MRI_arr.shape[2]
    ln_out = csm.dimensions()[2]
    lslice_out = csm.dimensions()[1]
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

    new_arr = sn.map_coordinates(rMRI_arr, points_out)
    new_arr = new_arr.reshape((lslice_out, ln_out, ln_out))
    z_mid = new_arr.shape[0] // 2
    if verbose:
        print("\nResampled phantom image")
        print(new_arr.shape)
        plt.figure()
        plt.imshow(new_arr[z_mid, :, :])
        plt.show()

    # Now forward project new_arr to make raw acquisition data
    print('---\n Backward projection starVIBE data ...')
    recon_img = acq_mod.inverse(acq_data)

    # Forward
    im_out = recon_img.clone()
    im_out.fill(new_arr)
    print('---\n Forward projection phantom data ...')
    raw_mri = acq_mod.forward(im_out)
    if verbose: print(raw_mri.dimensions())

    # Backward - actually inverse
    print('---\n Backward projection phantom data ...')
    bwd_mr = acq_mod.inverse(raw_mri)

    if verbose:
        fig, axs = plt.subplots(3, 1)
        fig.set_size_inches(11.69, 8.27)
        fig.suptitle("MR plots from 3D", fontsize=16)

        z_mid = new_arr.shape[0] // 2
        acq_dim = raw_mri.dimensions()
        axs[0].set_title("Original")
        axs[0].imshow(new_arr[z_mid, :, :])
        axs[0].axis("off")
        axs[1].set_title("Raw")
        axs[1].imshow(np.log(np.abs(
            raw_mri.as_array()[15:acq_dim[0]:38, 3, :])))
        axs[1].axis("off")
        axs[2].set_title("Backwards")
        axs[2].imshow(np.abs(bwd_mr.as_array()[z_mid, :, :]))
        axs[2].axis("off")

        plt.show()

    return bwd_mr

def calc_rad_traj_golden(ad):
    # Trajectory for Golden angle radial acquisitions
    dims = ad.dimensions()
    kx = np.linspace(-dims[2]//2, dims[2]//2, dims[2])
    ky = ad.get_ISMRMRD_info('kspace_encode_step_1')
    
    # Define angle with Golden angle increment
    angRad = ky * np.pi * 0.618034

    # Normalise radial points between [-0.5 0.5]
    krad = kx
    krad = krad / (np.amax(np.abs(krad)) * 2)

    # Calculate trajectory
    rad_traj = np.zeros((dims[2], dims[0], 2), dtype=np.float32)
    rad_traj[:, :, 0] = -1.0 * krad.reshape(-1, 1) * np.sin(angRad)
    rad_traj[:, :, 1] = -1.0 * krad.reshape(-1, 1) * np.cos(angRad)
    rad_traj = np.moveaxis(rad_traj, 0, 1)
    return rad_traj

def registerNifti(ref_name, flo_name, verbose=True):
    if ref_name[-4:] == ".nii":
        ref_image = mReg.ImageData(ref_name)
        flo_image = mReg.ImageData(flo_name)
    else:
        assert 1 == 0, "This isn't going to work right now"
        #ref_image = mPET.ImageData(ref_name)
        #flo_image = mPET.ImageData(flo_name)

    if verbose:
        output_arr = flo_image.as_array()
        out_slice = int(output_arr.shape[0] / 2)
        ref_arr = ref_image.as_array()
        print(ref_arr.shape)
        ref_slice = int(ref_arr.shape[0] / 2)
        print(output_arr.shape)
        slices = [output_arr[out_slice, :, :],
                  ref_arr[ref_slice, :, :]]
        fig, axes = plt.subplots(1, len(slices))
        for i, thisSlice in enumerate(slices):
            axes[i].imshow(thisSlice)
        plt.title('Initial image, slice: %i' % out_slice)
        plt.show()
        ref_y = int(ref_arr.shape[1] / 2)
        slices = [output_arr[:, ref_y, :],
                  ref_arr[:, ref_y, :]]
        fig, axes = plt.subplots(1, len(slices))
        for i, thisSlice in enumerate(slices):
            axes[i].imshow(thisSlice)
        plt.title('Initial image, y: %i' % out_slice)
        plt.show()
        
    # Set to NiftyF3dSym for non-rigid
    algo = mReg.NiftyAladinSym()

    # Set images
    algo.set_reference_image(ref_image)
    algo.set_floating_image(flo_image)

    algo.set_parameter('SetPerformRigid','1')
    algo.set_parameter('SetPerformAffine','1')
    #algo.set_parameter('SetWarpedPaddingValue','0') # initially this was unset (NaN)

    algo.process()

    if verbose:
        output = algo.get_output()
        #output.write("StarVIBE_in_PETspace_affine.hv")
        output_arr = output.as_array()
        out_slice = int(output_arr.shape[0] / 2)
        ref_arr = ref_image.as_array()
        print(ref_arr.shape)
        ref_slice = int(ref_arr.shape[0] / 2)
        print(output_arr.shape)
        slices = [output_arr[out_slice, :, :],
                  ref_arr[ref_slice, :, :]]
        fig, axes = plt.subplots(1, len(slices))
        for i, thisSlice in enumerate(slices):
            axes[i].imshow(thisSlice)
        plt.title('Registered image, slice: %i' % out_slice)
        plt.show()
        ref_y = int(ref_arr.shape[1] / 2)
        slices = [output_arr[:, ref_y, :],
                  ref_arr[:, ref_y, :]]
        fig, axes = plt.subplots(1, len(slices))
        for i, thisSlice in enumerate(slices):
            axes[i].imshow(thisSlice)
        plt.title('Registered image, y: %i' % out_slice)
        plt.show()

    np.set_printoptions(precision=3,suppress=True)
    TM = algo.get_transformation_matrix_forward()
    if verbose: print(TM.as_array())

    return TM

