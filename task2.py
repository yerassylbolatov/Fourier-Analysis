#Task 2. Fourier Series for a Discontinuous Hat Signal.
import numpy as np
import matplotlib.pyplot as plt

dx = 0.01
L = 2*np.pi
x = np.arange(0,L+dx,dx)
n = len(x)
nquart = int(np.floor(n/4))

f = np.zeros_like(x)
f[nquart:3*nquart] = 1

A0 = np.sum(f*np.ones_like(x)) * dx * 2 / L
fFS = A0/2*np.ones_like(f)

for k in range(1,101):
    Ak = np.sum(f*np.cos(2*np.pi*k*x/L)) * 2 * dx / L
    Bk = np.sum(f*np.sin(2*np.pi*k*x/L)) * 2 * dx / L
    fFS = fFS + Ak*np.cos(2*np.pi*k*x/L) + Bk*np.sin(2*np.pi*k*x/L)

plt.plot(x,f,color='k',linewidth=1.2)
plt.plot(x,fFS,'-',color='r',linewidth=0.8)
plt.show()