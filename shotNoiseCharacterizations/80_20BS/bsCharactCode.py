import numpy as np
import matplotlib.pyplot as plt

out80=np.array([-3.02,-1.99,-.98,.00,.98,1.97,3.04,4.01,15.05,40.55])*-1
out20=np.array([3.43,4.46,5.50,6.46,7.44,8.44,9.51,10.47,21.51,49.05])*-1

print(out80)
plt.figure()
plt.plot(out20,out80,'*')
plt.show()
