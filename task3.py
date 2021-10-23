#Discrete  Fourier  Transform  (DFT)
import numpy as np
import matplotlib.pyplot as plt

n = 256
w = np.exp(-1j*2*np.pi/n)

J,K = np.meshgrid(np.arange(1,n),np.arange(1,n))
DFT = np.power(w,(J-1)*(K-1))
DFT = np.real(DFT)

plt.imshow(DFT)
plt.show()