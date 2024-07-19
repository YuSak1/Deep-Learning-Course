#!/usr/bin/env python3
import numpy as np

if __name__ == "__main__":
    # Load data distribution, each data point on a line
    # Store in dictionary
    with open("numpy_entropy_data.txt", "r") as data:
        dataDict = {}
        countAll = 0
        for line in data:
            line = line.rstrip("\n")
            # TODO: process the line
            if line in dataDict:
                dataDict[line] += 1
            else:
                dataDict[line] = 1

            countAll += 1

    #print(dataDict)

    dataSorted = sorted(dataDict)     

    # TODO: Create a NumPy array containing the data distribution
    dataDist = np.zeros(len(dataDict))
    i = 0
    for item in dataSorted:
        dataDist[i] = dataDict[item]/countAll
        i += 1

    #print(dataDist)


    # Load model distribution, each line `word \t probability`, creating
    # a NumPy array containing the model distribution
    with open("numpy_entropy_model.txt", "r") as model:
        modelDict = {}    
        for line in model:
            line = line.rstrip("\n")
            # TODO: process the line
            # Store in dictionary
            split = line.split("\t")
            modelDict[split[0]] = split[1]
            
            
    # TODO: Compute and print entropy H(data distribution)
    entropy = 0
    i = 0
    for item in dataSorted:
        prob = dataDist[i]
        entropyEach = -prob * np.log(prob)
        entropy += entropyEach
        i += 1

    print("{:.2f}".format(entropy))


    # TODO: Compute and print cross-entropy H(data distribution, model distribution)
    # and KL-divergence D_KL(data distribution, model_distribution)

    # Compute cross-entropy
    cross_entropy = 0

    #print(modelDict)
    if '0' not in modelDict.values() and len(modelDict) >= len(dataDict):
        i = 0
        for item in dataSorted:
            prob = dataDist[i]
            entropyEach = -prob * np.log(float(modelDict[item]))
            cross_entropy += entropyEach
            i += 1
        
        print("{:.2f}".format(cross_entropy))
    else:
        print("inf")


    # Compute KL-divergence
    KL_divergence = 0


    if '0' not in modelDict.values() and len(modelDict) >= len(dataDict):
        i = 0
        for item in dataSorted:
            prob = dataDist[i]
            entropyEach = prob * (np.log(prob)-np.log(float(modelDict[item])))
            KL_divergence += entropyEach
            i += 1

        print("{:.2f}".format(KL_divergence))
    else:
        print("inf")