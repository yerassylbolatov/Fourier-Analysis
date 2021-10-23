import numpy as np
import matplotlib.pyplot as plt

#define domain
dx = 0.001
L = np.pi
x = L * np.arange(-1+dx,1+dx,dx)
n = len(x)
nquart = int(np.floor(n/4))

#define hat function
f = np.zeros_like(x)
f[nquart:2*nquart] = (4/n)*np.arange(1,nquart+1)
f[2*nquart:3*nquart] = 1 - (4/n)*np.arange(0,nquart)

fig, (ax1,ax2) = plt.subplots(1,2)
ax1.plot(x,f,'-',color='k',linewidth=2)

A0 = np.sum(f * np.ones_like(x)) * dx
fFS = A0/2

A = np.zeros(20)
B = np.zeros(20)
for k in range(20):
  A[k] = np.sum(f * np.cos(np.pi*(k+1)*x/L)) * dx
  B[k] = np.sum(f * np.sin(np.pi*(k+1)*x/L)) * dx
  fFS = fFS + A[k]*np.cos(np.pi*(k+1)*x/L) + B[k]*np.sin(np.pi*(k+1)*x/L)
  ax2.plot(x,fFS,'-')
plt.suptitle('Fourier series for a continuous hat function')
plt.show()