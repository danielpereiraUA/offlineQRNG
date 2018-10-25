import numpy as np
import matplotlib.pyplot as plt


from os import listdir
from os.path import isfile, join
files = [f for f in listdir('../characterizationData/')
         if isfile(join('../characterizationData/', f))]

nFiles = 30
variance = np.zeros(nFiles)
for i in range(nFiles):
    A = np.loadtxt(open(join("../characterizationData/",
                             files[i]), "rb"), delimiter=",", skiprows=4)
    a = A[0:len(A):10, 1]-np.mean(A[:, 1])
    variance[i] = np.var(a)

with open('variance.txt', 'w') as outFile:
    print(np.transpose(variance), file=outFile)

plt.figure()
plt.plot(variance)
plt.show()
