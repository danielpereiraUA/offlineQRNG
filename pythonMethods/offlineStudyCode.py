import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfinv

A = np.loadtxt(open("../csvData/polaIndTest2.csv", "rb"), delimiter=",", skiprows=4)
time = A[:, 0]*1e-3        # converting to s
amplitude = A[0:len(A), 1]*1e-3   # converting to V
meanAmplitude = np.mean(amplitude)
stdAmplitude = np.std(amplitude)

N = 100
corr = np.zeros(N)
for i in range(N):
    corr[i] = np.corrcoef(amplitude[0:int(1e6)], amplitude[int(i):int(1e6+i)])[0, 1]

# plt.figure()
# plt.plot(corr)
# plt.show()

aux = amplitude[0:len(amplitude):30]
amplitude = aux

# # Gray code generator
nGray = 2
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

grayCodeValue = np.zeros([2**nGray])
for i in range(2**nGray):
    for j in range(nGray):
        grayCodeValue[i] = grayCodeValue[i]+2*int(grayCode[i][j])-1

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

outBinary = []
for i in range(len(amplitude)):
    aux = amplitude[i] < binLimits
    outBinary.append(grayCode[len(grayCode)-sum(aux)-1])

OutBinary = ''.join(outBinary)
#
# print(OutBinary[2:5])

# this code outputs the generated binary
with open('../randomnessTestSuite/randomnessTest/out.txt', 'w') as outFile:
    print(OutBinary, file=outFile)
