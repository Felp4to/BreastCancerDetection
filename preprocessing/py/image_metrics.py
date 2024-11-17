import cv2
import numpy as np
import lpips
import torch
import warnings
warnings.filterwarnings("ignore")

# Confronta la distorsione dell'immagine preprocessata rispetto all'immagine originale. La PSNR misura 
# la qualità dell'immagine in termini di errore quadratico medio (MSE). Un valore di PSNR più elevato 
# indica una maggiore somiglianza tra l'immagine preprocessata e l'immagine originale, con minore distorsione.
def calculate_psnr(original, denoised):
    # calculate mse
    mse = np.mean((original - denoised) ** 2)
    if mse == 0:
        return float('inf')  # PSNR = inf means that the images are identical
    
    # max value possible for the pixel
    max_pixel_value = 255.0
    
    # calculate PSNR
    psnr = 10 * np.log10((max_pixel_value ** 2) / mse)

    return psnr


# indica la discrepanza quadratica media fra i valori dei dati osservati ed i valori dei dati stimati.
def mean_squared_error(original, denoised):
    # calculate mse
    mse = np.mean((original - denoised) ** 2)
    return mse


# LPIPS è un indice basato su reti neurali profonde, che è stato addestrato per predire quanto due immagini siano 
# simili, tenendo conto della percezione umana. In particolare, LPIPS misura la dissimilarità tra due immagini 
# usando rappresentazioni intermediate (caratteristiche) di una rete neurale pre-addestrata
def calculate_lpips(original, denoised):

    # Ridimensiona le immagini se necessario (LPIPS richiede che le immagini siano delle stesse dimensioni)
    img1 = cv2.resize(original, (50, 50))
    img2 = cv2.resize(denoised, (50, 50))

    # Converti le immagini in formato float e normalizzale
    img1 = img1.astype(np.float32) / 255.0
    img2 = img2.astype(np.float32) / 255.0

    # Converti in tensori PyTorch
    img1_tensor = torch.from_numpy(img1).permute(2, 0, 1)  # Cambia l'ordine da HWC a CHW
    img2_tensor = torch.from_numpy(img2).permute(2, 0, 1)  # Cambia l'ordine da HWC a CHW

    # Aggiungi un batch dimension
    img1_tensor = img1_tensor.unsqueeze(0)  # Aggiunge una dimensione per il batch
    img2_tensor = img2_tensor.unsqueeze(0)  # Aggiunge una dimensione per il batch

    # Crea un modello LPIPS
    loss_fn = lpips.LPIPS(net='alex')  # Usa il modello "alex", puoi scegliere anche "vgg" o "squeeze"
    
    # Calcola LPIPS
    lpips_score = loss_fn(img1_tensor, img2_tensor)
    return lpips_score.item()  # Restituisce il punteggio come un numero float


def calculate_metrics(image_rgb, lab_image):
    psnr = calculate_psnr(image_rgb, lab_image)
    lpips = calculate_lpips(image_rgb, lab_image)
    mse = mean_squared_error(image_rgb, lab_image)
    return psnr, lpips, mse