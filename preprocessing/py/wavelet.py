import cv2
import pywt
import numpy as np
import matplotlib.pyplot as plt

# reduce noise with wavelet method
def apply_wavelet_denoising(path):

    # load image in gray scale
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    # applica la DWT (Discrete Wavelet Transform)
    coeffs = pywt.dwt2(img, 'haar')   # haar???
    cA, (cH, cV, cD) = coeffs

    # set the threshold for the details
    threshold = 30  # Soglia di esempio, puoi modificarla in base all'immagine
    cH = pywt.threshold(cH, threshold, mode='soft')
    cV = pywt.threshold(cV, threshold, mode='soft')
    cD = pywt.threshold(cD, threshold, mode='soft')

    # build the image
    coeffs = (cA, (cH, cV, cD))
    img_denoised = pywt.idwt2(coeffs, 'haar')

    return img_denoised


# comparison between original and denoised image
def show_difference(original, denoised):

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(original, cmap='gray')

    plt.subplot(1, 2, 2)
    plt.title("Denoised Image")
    plt.imshow(denoised, cmap='gray')
    plt.show()