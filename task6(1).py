from matplotlib.image import imread
from mpl_toolkits.mplot3d import axes3d
import numpy as np
import matplotlib.pyplot as plt
import os

A = imread(os.path.join('Cotik.jpg'))
B = np.mean(A, -1);

Bt = np.fft.fft2(B)
Btsort = np.sort(np.abs(Bt.reshape(-1)))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
X,Y = np.meshgrid(np.arange(1,np.shape(B)[1]+1),np.arange(1,np.shape(B)[0]+1))
ax.plot_surface(X[0::10,0::10],Y[0::10,0::10],256-B[0::10,0::10],cmap='viridis',edgecolor='none')
ax.set_title('Surface plot')
ax.mouse_init()
ax.view_init(200,270)
plt.show()