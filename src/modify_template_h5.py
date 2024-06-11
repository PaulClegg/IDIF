"""
file: modify_template_h5.py
Script to modify matrix dimensions in StarVIBE MRI data
"""

import h5py as h5
import os
import csv
import numpy as np

def makeFilenameForStarVIBE():
    part1_path = "/home/pclegg/devel/SIRF-SuperBuild/docker/devel/"
    part2_path = "IDIF/data"
    data_path = os.path.join(part1_path, part2_path)
    name = "phantom_template.h5"
    filename = os.path.join(data_path, name)

    return filename

def changeMatrixSizeForStarVIBE():
    filename = makeFilenameForStarVIBE()

    f = h5.File(filename, "r+", libver="latest")
    print(f.name)
    print(list(f.keys()))
    print(f["dataset"])
    print(list(f["dataset"].keys()))
    print(f["dataset"]["data"])
    print(f["dataset"]["waveforms"])
    print(f["dataset"]["xml"])

    # Try to find the column headers within "xml"
    xml = f["dataset"]["xml"][0].decode('UTF-8')
    print(type(xml))

    # Print everything between <encoding> and <encodingLimits>

    lower = xml.split("<encoding>")[1]
    encoding = lower.split("<encodingLimits>")[0]
    print(encoding)

    # Change <z>88</z> to <z>38</z>
    print(xml.count("<z>88</z>"))
    xml_encoded = xml.replace("<z>88</z>", "<z>38</z>")

    # Change <z>80</z> to <z>38</z>
    print(xml_encoded.count("<z>80</z>"))
    xml_corrected = xml_encoded.replace("<z>80</z>", "<z>38</z>")

    # Check change - compare with original
    lower = xml_corrected.split("<encoding>")[1]
    encoding = lower.split("<encodingLimits>")[0]
    print(encoding)
    
    # Convert xml back to bytes
    xml = bytes(xml_corrected,'UTF-8')
    print(type(xml))

    # Return to dataset
    f["dataset"]["xml"][0] = xml
    print(f["dataset"]["xml"])

    # Save updated h5 file

    f.close()

def showDataHeadings():
    filename = makeFilenameForStarVIBE()

    f = h5.File(filename, "r+", libver="latest")
    print(f.name)
    print(list(f.keys()))
    print(f["dataset"])
    print(list(f["dataset"].keys()))
    print(f["dataset"]["data"])
    print(f["dataset"]["waveforms"])
    print(f["dataset"]["xml"])

    # Try to find the column headers within "xml"
    xml = f["dataset"]["xml"][0].decode('UTF-8')
    print(type(xml))

    # Exploration of timing information
    data = np.dtype(f["dataset"]["data"][0])
    print(data)

    f.close()

def findSampleTimeForStarVIBE():
    filename = makeFilenameForStarVIBE()

    f = h5.File(filename, "r+", libver="latest")
    print(f.name)
    print(list(f.keys()))
    print(f["dataset"])
    print(list(f["dataset"].keys()))
    print("\n\n")
    print(len(f["dataset"]["data"]))

    # Exploration of timing information
    dt = np.dtype(f["dataset"]["data"][0])
    print(dt)

    # Now try line 1
    print("\n\n")
    data0 = f["dataset"]["data"][0]["data"]
    print(data0)
    print(type(data0))
    print(len(data0))
    print("\n\n")
    head0 = f["dataset"]["data"][0]["head"]
    print(head0)
    print(type(head0))
    print(len(head0))

    # Try turning data into a np.array of dtype=dt
    print("\n\n")
    data = np.array(f["dataset"]["data"][0], dtype=dt)
    print(type(data))
    print(data["head"]["acquisition_time_stamp"])
    print(data["head"]["physiology_time_stamp"])

    lenData = len(f["dataset"]["data"])
    acq_time_stamps = np.zeros(lenData)
    phy_time_stamps = np.zeros(lenData)
    for i in range(lenData):
        data = np.array(f["dataset"]["data"][i], dtype=dt)
        acq_time_stamps[i] = data["head"]["acquisition_time_stamp"]
        phy_time_stamps[i] = data["head"]["physiology_time_stamp"][0]

    print(f"Earliest acquisition time stamp: {acq_time_stamps.min()}")
    print(f"Latest acquisition time stamp: {acq_time_stamps.max()}")
    print(f"Earliest physiology time stamp: {phy_time_stamps.min()}")
    print(f"Latest physiology time stamp: {phy_time_stamps.max()}")

    f.close()

def extractSpokeTimes():
    filename = os.path.join(data_path, name)

    f = h5.File(filename, "r+", libver="latest")
    print(f.name)
    print(list(f.keys()))
    print(f["dataset"])
    print(list(f["dataset"].keys()))
    print("\n\n")
    nSpokes = len(f["dataset"]["data"])
    spoke_num = np.zeros(nSpokes)
    spoke_time = np.zeros(nSpokes)

    dt = np.dtype(f["dataset"]["data"][0])
    print(dt)
    for i in range(nSpokes):
        data = np.array(f["dataset"]["data"][i], dtype=dt)
        spoke_num[i] = i
        spoke_time[i] = data["head"]["acquisition_time_stamp"]

    f.close()

    filename = "Spoke_times.csv"
    columns = ["Spoke number", "acquisition time"] 

    with open(filename, "w") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        for i in range(nSpokes):
            thisList = list([spoke_num[i], spoke_time[i]])
            writer.writerow(thisList)

def ECGforStarVIBE():
    filename = os.path.join(data_path, name)

    f = h5.File(filename, "r+", libver="latest")
    print(f.name)
    print(list(f.keys()))
    print(f["dataset"])
    print(list(f["dataset"].keys()))
    waveform_len = len(f["dataset"]["waveforms"])
    print(waveform_len)
    print(f["dataset"]["waveforms"])

    print("\n\n")
    dt = np.dtype(f["dataset"]["waveforms"][0])
    print(dt)

    # Now try line 1
    print("\n\n")
    data0 = f["dataset"]["waveforms"][0]["data"]
    print(data0)
    print(type(data0))
    print(len(data0))
    print("\n\n")
    head0 = f["dataset"]["waveforms"][0]["head"]
    print(head0)
    print(type(head0))
    print(len(head0))
    print("spokes " + str(head0[3]))
    print("time_stamp " + str(head0[4]))

    nsamples = int((waveform_len / 5) * 40)

    spoke_list = np.zeros(nsamples)
    given_time = np.zeros(nsamples)
    derived_times = np.zeros(nsamples)
    channel1 = np.zeros(nsamples)
    channel2 = np.zeros(nsamples)
    for fifth in range(5, waveform_len + 5, 5):
        thisData = f["dataset"]["waveforms"][fifth - 5]["data"]
        thisHead = f["dataset"]["waveforms"][fifth - 5]["head"]
        i = (fifth // 5 - 1) * 40
        spoke_list[i:(i + 40)] = thisHead[3]
        given_time[i:(i + 40)] = thisHead[4]
        channel1[i:(i + 40)] = thisData[0:40]
        channel2[i:(i + 40)] = thisData[40:80]
        derived_times[i:(i + 40)] = np.array([thisHead[4] + x for x in range(40)])

    f.close()

    filename = "ECGfromStarVibe.csv"
    columns = ["derived time", "given time", "spoke list", "channel 1", "channel 2"] 

    with open(filename, "w") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        for i in range(nsamples):
            thisList = list([derived_times[i], given_time[i], spoke_list[i], 
                channel1[i], channel2[i]])
            writer.writerow(thisList)


if __name__ == "__main__":
    #findSampleTimeForStarVIBE()
    #ECGforStarVIBE()
    #extractSpokeTimes()
    #changeMatrixSizeForStarVIBE()
    showDataHeadings()
