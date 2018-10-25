import numpy as np
import matplotlib.pyplot as plt

A = np.loadtxt(open("../shotNoiseCharacterizations/characterizationData/electricNoise.csv",
                    "rb"), delimiter=",", skiprows=4)
a = A[0:len(A):10, 1]-np.mean(A[:, 1])
B = np.loadtxt(open("../shotNoiseCharacterizations/characterizationData/1199.csv",
                    "rb"), delimiter=",", skiprows=4)
b = B[0:len(B):10, 1]-np.mean(B[:, 1])

plt.figure()
plt.plot(b)
plt.plot(a)


plt.figure()
plt.hist(b, 20)
plt.hist(a, 20)
plt.show()
