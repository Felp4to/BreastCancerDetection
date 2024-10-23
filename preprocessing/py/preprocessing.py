import clahe as clahe
import fuzzy as fuzzy
import he as he
import wavelet as wavelet
import no_denoising as nd
import matplotlib.pyplot as plt


# mio_modulo.py
def show_comparison(images, titles):
    
    # Creiamo una figura con 1 riga e 5 colonne
    fig, axs = plt.subplots(1, 5, figsize=(15, 5))

    # Mostriamo ciascuna immagine nella rispettiva colonna
    for i in range(len(images)-1):
        axs[i].imshow(images[i], cmap='gray')  # Mostra l'immagine
        axs[i].set_title(titles[i])  # Aggiungi il titolo
        axs[i].axis('off')  # Rimuovi gli assi

    plt.tight_layout()
    plt.show()
        

# apply pre-processing to the image
def image_processor(image_path, target_size, type='no_denoising'):
    if type == 'no_denoising':
        return nd.no_denoising(image_path, target_size)
    elif type == 'clahe':
        return clahe.apply_equalization_clahe_Lab(image_path, target_size, 0, 0)
    elif type == 'he':
        return he.apply_histogram_equalization(image_path, target_size)
    elif type == 'fuzzy':
        return fuzzy.FuzzyContrastEnhance(image_path, target_size)
    elif type == 'wavelet':
        return wavelet.apply_wavelet_denoising(image_path, target_size)










