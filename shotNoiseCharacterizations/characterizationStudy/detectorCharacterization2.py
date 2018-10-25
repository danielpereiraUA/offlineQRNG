import numpy as np
import matplotlib.pyplot as plt

variance = np.loadtxt(open("variance.txt", "rb"), delimiter=",")

plt.figure()
plt.plot(variance[:, 0], variance[:, 1], '.')
plt.show()
