"""
file: MRI.py
To create and manipulate MRI data from the phantom images
"""

def convertPhantomToT1Values(ph_image, properties, verbose=True):
    T1_image = ph_image.clone()
    image_arr = ph_image.as_array()
    [name, num, T1, T2, ADC, PDFF] = properties

    for i in range(len(num)):
        image_arr[image_arr == num[i]] = T1[i]
        
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
