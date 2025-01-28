"""
file: uMap.py
To create and manipulate an attenuation map from the phantom data
"""


def convertPhantomToUmap(phantom_image, verbose=True):
    uMap_image = phantom_image.clone()
    if verbose: print(uMap_image.dimensions())

    working_arr = phantom_image.as_array()
    working_arr[working_arr < 1E-5] = 0.0 # background
    working_arr[working_arr == 1] = 0.16 # bone for spine and ribs
    working_arr[working_arr > 1] = 0.1 # remaining soft tissue

    uMap_image.fill(working_arr)

    return uMap_image
