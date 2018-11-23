import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfinv
import csv

A = np.loadtxt(open("dataTest3.csv", "rb"), delimiter=",", skiprows=4)
time = A[:, 0]*1e-3        # converting to s
amplitude = A[:, 1]*1e-3   # converting to V
meanAmplitude = np.mean(amplitude)
stdAmplitude = np.std(amplitude)
B = np.loadtxt(open("laser_off.csv", "rb"), delimiter=",", skiprows=0)
timeElect = B[:, 0]        # data is already in s
amplitudeElect = B[:, 1]+meanAmplitude   # data is already in V

# Gray code generator
nGray = 3
grayCode = ["0", "1"]
aux1 = ["1", "0"]
for i in range(nGray-1):
    aux0 = grayCode + []
    aux1 = aux0 + []
    aux1.reverse()
    for j in range(0, len(aux0)):
        aux0[j] = '0'+aux0[j]
        aux1[j] = '1'+aux1[j]
    grayCode = aux0+aux1
    aux0 = grayCode+[]

# Bin limit calculator
binLimits = np.zeros([2**nGray-1])
N = 2**(nGray-1)
if nGray == 1:
    binLimits = meanAmplitude
else:
    for i in range(N):
        binLimits[N-i-1] = meanAmplitude + np.sqrt(2)*stdAmplitude*erfinv(-float(i)/N)
        if i != 0:
            binLimits[N+i-1] = meanAmplitude + np.sqrt(2)*stdAmplitude*erfinv(float(i)/N)

counts = np.zeros(len(grayCode))
countsElect = np.zeros(len(grayCode))
outBinary = []
for i in range(len(amplitude)):
    aux = amplitude[i] < binLimits
    outBinary.append(grayCode[len(grayCode)-sum(aux)-1])
    counts[len(grayCode)-sum(aux)-1] = counts[len(grayCode)-sum(aux)-1] + 1
    if i < len(amplitudeElect):
        auxElect = amplitudeElect[i] < binLimits
        countsElect[len(grayCode)-sum(auxElect)-1] = countsElect[len(grayCode)-sum(auxElect)-1] + 1

probability = counts / len(amplitude)
probabilityElect = countsElect / len(amplitudeElect)

entropyTotal = -np.sum(probability*np.log2(probability))
entropyElect = -np.sum(probabilityElect[probabilityElect > 0]*np.log2(probabilityElect[probabilityElect > 0]))

# with open('out.csv', 'w') as csvfile:
#     writer = csv.writer(csvfile, lineterminator='\n')
#     for val in outBinary:
#         writer.writerow([val])

print(counts)
print(countsElect)
print(entropyTotal)
print(entropyElect)

# with open('out.csv', 'w') as csvfile:
#     writer = csv.writer(csvfile, lineterminator='\n')
#     for val in outBinary:
#         writer.writerow([val])

# print(outBinary)
