import numpy as np
import matplotlib.pyplot as plt

A = np.loadtxt(open("dataTest3.csv", "rb"), delimiter=",", skiprows=4)
time = A[:, 0]*1e-3        # converting to s
amplitude = A[:, 1]*1e-3   # converting to V

nGray = 3
bins = 2**nGray
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

width = (np.max(amplitude)-np.min(amplitude))/bins
binLimits = np.zeros(bins-1)

for i in range(bins-1):
    if i != 0:
        binLimits[i] = binLimits[i-1]+width
    else:
        binLimits[i] = np.min(amplitude)+width
print('binning now!')
print(len(amplitude))

n = 100000

outBinary = np.zeros(n)
for i in range(n):
    print(i)
    aux = amplitude[i] < binLimits
    outBinary[i] = grayCode[len(grayCode)-sum(aux)-1]


hist, binedges = np.histogram(outBinary, bins=bins)

plt.figure()
plt.plot(hist)
plt.show()

print('done!')
