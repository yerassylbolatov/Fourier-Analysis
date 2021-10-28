# Task 5 
# Two-dimensional Fourier transform via one-dimensional row-wise and column-wise fast Fourier transforms.
from matplotlib.image import imread
import numpy as np
import matplotlib.pyplot as plt
import os

A = imread(os.path.join('Cotik.jpg'))
B = np.mean(A, -1)

fig,axs = plt.subplots(2,2)

# Real image
axs[0,0].imshow(A)
axs[0,0].title.set_text('Image')
axs[0,0].axis('off')

# Grayscale image
axs[0,1].imshow(B, cmap='gray')
axs[0,1].title.set_text('Gray Image')
axs[0,1].axis('off')

# Compute row-wise FFT
Cshift = np.zeros_like(B,dtype='complex_')
C = np.zeros_like(B,dtype='complex_')
for j in range(B.shape[0]):
    Cshift[j,:] = np.fft.fftshift(np.fft.fft(B[j,:]))
    C[j,:] = np.fft.fft(B[j,:])

img = axs[1,0].imshow(np.log(np.abs(Cshift)))
img.set_cmap('gray')
axs[1,0].title.set_text('Row-wise FFT')
axs[1,0].axis('off')

# Compute column-wise FFT
D = np.zeros_like(C)
for j in range(C.shape[1]):
    D[:,j] = np.fft.fft(C[:,j])

img = axs[1,1].imshow(np.fft.fftshift(np.log(np.abs(D))))
img.set_cmap('gray')
axs[1,1].title.set_text('Column-wise FFT')
axs[1,1].axis('off')

plt.show()