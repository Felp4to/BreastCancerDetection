import cv2
import pywt
import numpy as np
import matplotlib.pyplot as plt

# Funzione per applicare DWT e ridurre il rumore su un singolo canale
def denoise_channel(channel):
    # Applica la DWT (Discrete Wavelet Transform) sul canale
    coeffs = pywt.dwt2(channel, 'haar')  # Puoi scegliere altre famiglie wavelet (haar, db, sym, ecc.)
    cA, (cH, cV, cD) = coeffs

    # Imposta la soglia per i dettagli (puoi modificarla per ottimizzare il risultato)
    threshold = 30
    cH = pywt.threshold(cH, threshold, mode='soft')
    cV = pywt.threshold(cV, threshold, mode='soft')
    cD = pywt.threshold(cD, threshold, mode='soft')

    # Ricostruisce il canale denoised
    coeffs = (cA, (cH, cV, cD))
    channel_denoised = pywt.idwt2(coeffs, 'haar')
    
    # Normalizza i valori del canale a 8-bit (0-255)
    channel_denoised = np.clip(channel_denoised, 0, 255).astype(np.uint8)
    
    return channel_denoised


def apply_wavelet_denoising(path, target_size, debug=0, show_result=0):
    # Carica l'immagine a colori (BGR)
    img = cv2.imread(path, cv2.IMREAD_COLOR)

    # Separa i canali BGR
    B, G, R = cv2.split(img)

    # Applica la riduzione del rumore a ciascun canale
    B_denoised = denoise_channel(B)
    G_denoised = denoise_channel(G)
    R_denoised = denoise_channel(R)

    # Combina i canali denoised in un'immagine BGR
    img_denoised = cv2.merge([B_denoised, G_denoised, R_denoised])

    # Ridimensiona l'immagine se necessario
    if target_size is not None:
        img_denoised = cv2.resize(img_denoised, (target_size[1], target_size[0]), interpolation=cv2.INTER_LINEAR)

    # Se debug è attivato, mostra gli istogrammi
    if debug:
        stamp_histogram(img)
        stamp_histogram(img_denoised)

    # Se show_result è attivato, mostra la differenza tra immagine originale e denoised
    if show_result:
        show_difference(img, img_denoised)

    return img_denoised


# comparison between original and equalized image
def show_difference(img, denoised):
    # Imposta la figura per mostrare le immagini affiancate
    plt.figure(figsize=(10, 5))

    # Mostra l'immagine originale
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Immagine Original')
    plt.axis('off')

    # Mostra l'immagine originale
    plt.subplot(1, 2, 2)
    plt.imshow(denoised, cmap='gray')
    plt.title('Denoised Image with Wavelet')
    plt.axis('off')


# shows histogram and cdf (cumulative distribution function)
def stamp_histogram(img):
    hist,bins = np.histogram(img.flatten(),256,[0,256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * float(hist.max()) / cdf.max()
    plt.plot(cdf_normalized, color = 'b')
    plt.hist(img.flatten(),256,[0,256], color = 'r')
    plt.xlim([0,256])
    plt.legend(('cdf','histogram'), loc = 'upper left')
    plt.show()
