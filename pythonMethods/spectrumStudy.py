import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfinv

laserSpectrum=np.loadtxt(open("../csvData/spectrumWithLaser.csv", "rb"), delimiter=",", skiprows=4)
thermalSpectrum=np.loadtxt(open("../csvData/spectrumWoutLaser.csv", "rb"), delimiter=",", skiprows=4)

plt.figure()
plt.plot(laserSpectrum[:,0],laserSpectrum[:,1])
plt.plot(thermalSpectrum[:,0],thermalSpectrum[:,1])

plt.figure()
plt.plot(laserSpectrum[:,0],laserSpectrum[:,1]-thermalSpectrum[:,1])

plt.show()
