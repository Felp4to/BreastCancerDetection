import cv2
import pywt
import numpy as np
import image_metrics as im
import matplotlib.pyplot as plt

# Funzione per applicare DWT e ridurre il rumore su un singolo canale
#def denoise_channel(channel):
    # Applica la DWT (Discrete Wavelet Transform) sul canale
 #   coeffs = pywt.dwt2(channel, 'haar')  # Puoi scegliere altre famiglie wavelet (haar, db, sym, ecc.)
 #   cA, (cH, cV, cD) = coeffs

    # Imposta la soglia per i dettagli (puoi modificarla per ottimizzare il risultato)
 #   threshold = 10
  #  cH = pywt.threshold(cH, threshold, mode='soft')
  #  cV = pywt.threshold(cV, threshold, mode='soft')
  #  cD = pywt.threshold(cD, threshold, mode='soft')

    # Ricostruisce il canale denoised
 #   coeffs = (cA, (cH, cV, cD))
 #   channel_denoised = pywt.idwt2(coeffs, 'haar')
  #  
    # Normalizza i valori del canale a 8-bit (0-255)
 #   channel_denoised = np.clip(channel_denoised, 0, 255).astype(np.uint8)
    
   # return channel_denoised

def denoise_channel(channel):
    # Applica la DWT (Discrete Wavelet Transform) sul canale
    coeffs = pywt.dwt2(channel, 'haar')  # Puoi scegliere altre famiglie wavelet (haar, db, sym, ecc.)
    cA, (cH, cV, cD) = coeffs

    # Stima della deviazione standard del rumore usando la mediana di cD
    sigma = np.median(np.abs(cD)) / 0.6745
    # Calcolo della soglia universale
    N = cH.size  # Numero di coefficienti
    threshold = sigma * np.sqrt(2 * np.log(N))

    # Applica la soglia ai coefficienti di dettaglio
    cH = pywt.threshold(cH, threshold, mode='soft')
    cV = pywt.threshold(cV, threshold, mode='soft')
    cD = pywt.threshold(cD, threshold, mode='soft')

    # Ricostruisce il canale denoised
    coeffs = (cA, (cH, cV, cD))
    channel_denoised = pywt.idwt2(coeffs, 'haar')
    
    # Normalizza i valori del canale a 8-bit (0-255)
    channel_denoised = np.clip(channel_denoised, 0, 255).astype(np.uint8)
    
    return channel_denoised


def apply_wavelet_denoising(path, target_size, debug=0, show_result=0, show_histogram=0, show_metrics=0):
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
    
    if (show_histogram):
        show_histograms_2(img, img_denoised)

    if(show_metrics):
        psnr, lpips, mse = im.calculate_metrics(img, img_denoised)
        print("psnr: ", psnr)
        print("lpips: ", lpips)
        print("mse: ", mse)

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


def show_histograms_2(img1, img2):
    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # First image histogram and CDF
    hist1, bins1 = np.histogram(img1.flatten(), 256, [0, 256])
    cdf1 = hist1.cumsum()
    cdf_normalized1 = cdf1 * float(hist1.max()) / cdf1.max()
    
    axes[0].plot(cdf_normalized1, color='b')
    axes[0].hist(img1.flatten(), 256, [0, 256], color='r')
    axes[0].set_xlim([0, 256])
    axes[0].set_title("Original Image")
    axes[0].legend(('CDF', 'Histogram'), loc='upper left')

    # Second image histogram and CDF
    hist2, bins2 = np.histogram(img2.flatten(), 256, [0, 256])
    cdf2 = hist2.cumsum()
    cdf_normalized2 = cdf2 * float(hist2.max()) / cdf2.max()

    axes[1].plot(cdf_normalized2, color='b')
    axes[1].hist(img2.flatten(), 256, [0, 256], color='r')
    axes[1].set_xlim([0, 256])
    axes[1].set_title("Wavelet Image")
    axes[1].legend(('CDF', 'Histogram'), loc='upper left')

    plt.show()


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
