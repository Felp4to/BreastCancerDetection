from skimage import io  
import cv2
import numpy as np
import matplotlib.pyplot as plt
import image_metrics as metrics
import image_metrics as im


# Funzione per applicare un filtro passa-basso a un singolo canale
def low_pass_filter_channel(channel):
    # Calcola la trasformata di Fourier
    f_transform = np.fft.fft2(channel)
    f_transform_shifted = np.fft.fftshift(f_transform)      # Shift per centrare lo zero

    # Calcola l'energia spettrale (modulo quadrato della trasformata di Fourier)
    energy = np.abs(f_transform_shifted) ** 2
    total_energy = np.sum(energy)                           # Energia totale

    # Calcola il 80% dell'energia totale
    target_energy = 0.80 * total_energy

    # Crea una matrice di zeri per la maschera
    rows, cols = channel.shape
    crow, ccol = rows // 2, cols // 2                       # Centro dell'immagine

    # Ordina le frequenze per energia decrescente e calcola il cutoff
    energy_sorted = np.sort(energy.flatten())               # Vettorizza l'energia
    cumulative_energy = np.cumsum(energy_sorted)            # Energia cumulativa
    cutoff_value = energy_sorted[np.searchsorted(cumulative_energy, target_energy)]  # Trova il valore di cutoff

    # Crea la maschera per il filtro passa-basso
    mask = np.zeros((rows, cols), dtype=np.float32)

    # Seleziona le frequenze che sono inferiori al valore di cutoff
    for i in range(rows):
        for j in range(cols):
            if energy[i, j] <= cutoff_value:
                mask[i, j] = 1

    # Applica il filtro alla trasformata di Fourier
    filtered_transform = f_transform_shifted * mask

    # Inversa della trasformata di Fourier
    filtered_transform_shifted = np.fft.ifftshift(filtered_transform)
    filtered_image = np.fft.ifft2(filtered_transform_shifted)
    filtered_image = np.abs(filtered_image)  # Prendi il valore assoluto

    return filtered_image


# Funzione per applicare un filtro passa-basso a un singolo canale
def low_pass_filter_channel_2(channel):
    # Calcola la trasformata di Fourier
    f_transform = np.fft.fft2(channel)
    f_transform_shifted = np.fft.fftshift(f_transform)      # Shift per centrare lo zero

    # Crea il filtro passa-basso
    rows, cols = channel.shape
    crow, ccol = rows // 2, cols // 2  # Centro dell'immagine

    # Crea una matrice di zeri
    mask = np.zeros((rows, cols), dtype=np.float32)

    cutoff = 35

    # Crea un cerchio (filtro) nel centro della matrice
    x, y = np.ogrid[-crow:rows-crow, -ccol:cols-ccol]
    mask_area = x**2 + y**2 <= cutoff**2
    mask[mask_area] = 1

    # Applica il filtro alla trasformata di Fourier
    filtered_transform = f_transform_shifted * mask

    # Inversa della trasformata di Fourier
    filtered_transform_shifted = np.fft.ifftshift(filtered_transform)
    filtered_image = np.fft.ifft2(filtered_transform_shifted)
    filtered_image = np.abs(filtered_image)  # Prendi il valore assoluto

    return filtered_image


# Funzione per filtrare l'immagine a colori
def low_pass_filter_color_image(image):
    # Inizializza un array per l'immagine filtrata
    filtered_image = np.zeros_like(image)

    # Applica il filtro passa-basso a ciascun canale
    for i in range(image.shape[2]):  # Itera su ogni canale (R, G, B)
        filtered_image[:, :, i] = low_pass_filter_channel_2(image[:, :, i])

    return filtered_image

# Funzione per applicare il denoising tramite il filtro passa-basso
def apply_fourier_denoising(path, target_size, histogram=0, show_result=0, show_histogram=0, show_metrics=0):
    image = io.imread(path)
    
    # Applica il filtro passa-basso all'immagine a colori con cutoff automatico
    filtered_image = low_pass_filter_color_image(image)

    # Resize the image if a target size is provided
    if target_size is not None:
        filtered_image = cv2.resize(filtered_image, (target_size[1], target_size[0]), interpolation=cv2.INTER_LINEAR)

    if(show_result):
        plot_filtered_image(image, filtered_image)

    if(show_histogram):
        show_histograms(image, filtered_image)

    if(show_metrics):
        psnr, lpips, mse = im.calculate_metrics(image, filtered_image)
        print("psnr: ", psnr)
        print("lpips: ", lpips)
        print("mse: ", mse)
    
    return filtered_image


def show_histograms(img1, img2): 
    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # First image histogram and CDF
    hist1, bins1 = np.histogram(img1.flatten(), 256, [0, 256])
    cdf1 = hist1.cumsum()
    cdf_normalized1 = cdf1 * float(hist1.max()) / cdf1.max()
    
    axes[0].plot(cdf_normalized1, color='b')
    axes[0].hist(img1.flatten(), 256, [0, 256], color='r')
    axes[0].set_xlim([0, 256])
    axes[0].set_title("Original image")
    axes[0].legend(('CDF', 'Histogram'), loc='upper left')

    # Second image histogram and CDF
    hist2, bins2 = np.histogram(img2.flatten(), 256, [0, 256])
    cdf2 = hist2.cumsum()
    cdf_normalized2 = cdf2 * float(hist2.max()) / cdf2.max()

    axes[1].plot(cdf_normalized2, color='b')
    axes[1].hist(img2.flatten(), 256, [0, 256], color='r')
    axes[1].set_xlim([0, 256])
    axes[1].set_title("Fuzzy Image")
    axes[1].legend(('CDF', 'Histogram'), loc='upper left')

    plt.show()

def plot_filtered_image(original_image, filtered_image):
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(original_image)
    plt.title('Immagine Originale')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(filtered_image)
    plt.title('Immagine Filtrata')
    plt.axis('off')

    plt.show()


























