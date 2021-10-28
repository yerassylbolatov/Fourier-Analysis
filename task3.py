#Discrete  Fourier  Transform  (DFT).
#Discrete  Fourier  transform  (DFT)  matrix  with  𝑛=256.  
#Real  part  of  DFT matrix.  
import numpy as np
import matplotlib.pyplot as plt

n = 256
w = np.exp(-1j*2*np.pi/n)

#Fast meshgrid algorithm
J,K = np.meshgrid(np.arange(1,n),np.arange(1,n))
DFT = np.power(w,(J-1)*(K-1))
DFT = np.real(DFT)

plt.imshow(DFT)
plt.show()