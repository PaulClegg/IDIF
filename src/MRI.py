"""
file: MRI.py
To create and manipulate MRI data from the phantom images
"""


def readTissueProperties(filename, verbose=True):

    if verbose: print(filename)

    name = []; num = []; T1 = []; T2 = []; ADC = []; PDFF = []
    line_num = 0
    with open(filename) as file:
        for line in file:
            if line_num > 11: 
                print(line.rstrip())
            line_num += 1

    properties = [name, num, T1, T2, ADC, PDFF]
    return properties

def divideLineIntoChunks(line, verbose=True):
    initial = line.split("=")[0]
