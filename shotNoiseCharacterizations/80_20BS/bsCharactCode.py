import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

out80 = np.array([-3.02, -1.99, -.98, .00, .98, 1.97, 3.04, 4.01, 15.05, 40.55])*-1
out20 = np.array([3.43, 4.46, 5.50, 6.46, 7.44, 8.44, 9.51, 10.47, 21.51, 49.05])*-1

out80mW = 10**(out80/10)
out20mW = 10**(out20/10)

slope, intercept, r_value, p_value, std_err = stats.linregress(out20mW, out80mW)

x = np.array([0, 0.5])

print(slope, intercept)

plt.figure()
plt.plot(x, slope*x+intercept)
plt.plot(out20mW, out80mW, '*')
plt.show()
