from matplotlib.image import imread
import numpy as np
import matplotlib.pyplot as plt
import os

A = imread(os.path.join('Cotik.jpg'))
B = np.mean(A, -1);

plt.figure()
plt.imshow(A)
plt.axis('off')
plt.show()

Bt = np.fft.fft2(B)
Btsort = np.sort(np.abs(Bt.reshape(-1)))

for keep in (1,0.1,0.05,0.01,0.002):
    thresh = Btsort[int(np.floor((1-keep)*len(Btsort)))]
    ind = np.abs(Bt)>thresh
    Atlow = Bt * ind
    Alow = np.fft.ifft2(Atlow).real
    plt.figure()
    plt.imshow(Alow, cmap = 'gray')
    plt.axis('off')
    plt.title('Compressed image: keep = ' + str(keep*100) + '%')

plt.show()

