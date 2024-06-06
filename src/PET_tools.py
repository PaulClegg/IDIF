"""
file: PET_tools.py
To generate and reconstruct raw PET data from digital phantom
"""

import os
import numpy as np
import scipy.ndimage as sn
from matplotlib import pyplot as plt

import sirf.STIR as pPET

def create3Dtemplate(data_stem):
    template = pPET.AcquisitionData('Siemens_mMR', span=11, 
        max_ring_diff=60, view_mash_factor=1)
    out_path = os.path.join(data_stem, "template3D.hs")
    template.write(out_path)

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
    reshaped_arr = np.zeros((orig_arr.shape[2], orig_arr.shape[0], 
        orig_arr.shape[1]))
    for z in range(orig_arr.shape[2]):
        reshaped_arr[z, :, :] = orig_arr[:, :, z]
    ln = orig_arr.shape[0]
    lslice = orig_arr.shape[2]
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
