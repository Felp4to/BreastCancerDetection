import cv2
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display, Markdown
from glob2 import glob
import time
import image_metrics as im


# Gaussian Function:
def G(x, mean, std):
    return np.exp(-0.5*np.square((x-mean)/std))

# Membership Functions:
def ExtremelyDark(x, M):
    return G(x, -50, M/6)

def VeryDark(x, M):
    return G(x, 0, M/6)

def Dark(x, M):
    return G(x, M/2, M/6)

def SlightlyDark(x, M):
    return G(x, 5*M/6, M/6)

def SlightlyBright(x, M):
    return G(x, M+(255-M)/6, (255-M)/6)

def Bright(x, M):
    return G(x, M+(255-M)/2, (255-M)/6)

def VeryBright(x, M):
    return G(x, 255, (255-M)/6)

def ExtremelyBright(x, M):
    return G(x, 305, (255-M)/6)

def OutputFuzzySet(x, f, M, thres):
    x = np.array(x)
    result = f(x, M)
    result[result > thres] = thres
    return result

def AggregateFuzzySets(fuzzy_sets):
    return np.max(np.stack(fuzzy_sets), axis=0)

def Infer(i, M, get_fuzzy_set=False):
    # Calculate degree of membership for each class
    VD = VeryDark(i, M)
    Da = Dark(i, M)
    SD = SlightlyDark(i, M)
    SB = SlightlyBright(i, M)
    Br = Bright(i, M)
    VB = VeryBright(i, M)
    
    # Fuzzy Inference:
    x = np.arange(-50, 306)
    Inferences = (
        OutputFuzzySet(x, ExtremelyDark, M, VD),
        OutputFuzzySet(x, VeryDark, M, Da),
        OutputFuzzySet(x, Dark, M, SD),
        OutputFuzzySet(x, Bright, M, SB),
        OutputFuzzySet(x, VeryBright, M, Br),
        OutputFuzzySet(x, ExtremelyBright, M, VB)
    )
    
    # Calculate AggregatedFuzzySet:
    fuzzy_output = AggregateFuzzySets(Inferences)
    
    # Calculate crisp value of centroid
    if get_fuzzy_set:
        return np.average(x, weights=fuzzy_output), fuzzy_output
        
    return np.average(x, weights=fuzzy_output)


# fuzzy method
def FuzzyContrastEnhance(path, target_size=(50, 50, 3), show_result=0, show_histogram=0, show_metrics=0):

    image = cv2.imread(path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Convert RGB to LAB
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    
    # Get L channel
    l = lab[:, :, 0]
    
    # Calculate M value
    M = np.mean(l)

    if M < 128:
        M = 127 - (127 - M)/2
    else:
        M = 128 + M/2
        
    # Precompute the fuzzy transform
    x = list(range(-50,306))
    #x = list(range(-150,406))

    FuzzyTransform = dict(zip(x,[Infer(np.array([i]), M) for i in x]))
    
    # Apply the transform to l channel
    u, inv = np.unique(l, return_inverse = True)
    l = np.array([FuzzyTransform[i] for i in u])[inv].reshape(l.shape)
    
    # Min-max scale the output L channel to fit (0, 255):
    Min = np.min(l)
    Max = np.max(l)
    lab[:, :, 0] = (l - Min)/(Max - Min) * 255

    # Convert LAB to RGB
    lab_image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    #if(target_size != None):
        #lab_image = cv2.resize(lab_image, (target_size[1], target_size[0]), interpolation=cv2.INTER_LINEAR)
       # lab_image = cv2.resize(lab_image, (width, height), interpolation=cv2.INTER_LINEAR)

    # show images
    if(show_result): 
        show_difference(image_rgb, lab_image)

    if(show_histogram):
        show_histograms(image_rgb, lab_image)

    if(show_metrics):
        psnr, lpips, mse = im.calculate_metrics(image_rgb, lab_image)
        print("psnr: ", psnr)
        print("lpips: ", lpips)
        print("mse: ", mse)

    lab_image = lab_image / 255.0

    return lab_image


# this function draws the curves of  the degrees of membership
def plot_fuzzy_membership(M, ExtremelyDark, VeryDark, Dark, SlightlyDark, SlightlyBright, Bright, VeryBright, ExtremelyBright):

    # Array di intensità dei pixel da -50 a 305
    x = np.arange(-50, 306)

    # computation of the degrees of memmbership for each pixel intensity x and the value M
    ED = ExtremelyDark(x, M)
    VD = VeryDark(x, M)
    Da = Dark(x, M)
    SD = SlightlyDark(x, M)
    SB = SlightlyBright(x, M)
    Br = Bright(x, M)
    VB = VeryBright(x, M)
    EB = ExtremelyBright(x, M)

    # create graphic
    plt.figure(figsize=(20,5))
    plt.plot(x, ED, 'k-.', label='ED', linewidth=1)
    plt.plot(x, VD, 'k-', label='VD', linewidth=2)
    plt.plot(x, Da, 'g-', label='Da', linewidth=2)
    plt.plot(x, SD, 'b-', label='SD', linewidth=2)
    plt.plot(x, SB, 'r-', label='SB', linewidth=2)
    plt.plot(x, Br, 'c-', label='Br', linewidth=2)
    plt.plot(x, VB, 'y-', label='VB', linewidth=2)
    plt.plot(x, EB, 'y-.', label='EB', linewidth=1)
    
    # Reference lines for M, minimum (0) and maximum (255) pixel intensity
    plt.plot((M, M), (0, 1), 'm--', label='M', linewidth=2)
    plt.plot((0, 0), (0, 1), 'k--', label='MinIntensity', linewidth=2)
    plt.plot((255, 255), (0, 1), 'k--', label='MaxIntensity', linewidth=2)

    # graphic settings
    plt.legend()
    plt.xlim(-50, 305)
    plt.ylim(0.0, 1.01)
    plt.xlabel('Pixel intensity')
    plt.ylabel('Degree of membership')
    plt.title(f'M={M}')
    plt.show()


# plot fuzzy sets for a list of pixel values and a constant M.
def plot_fuzzy_sets(pixels = [64, 96, 160, 192], M=128):

    x = np.arange(-50, 306)  # define axis x, from -50 to 305

    for pixel in pixels:
        # Calcola il centroide e il set fuzzy di output per il valore di pixel attuale
        centroid, output_fuzzy_set = Infer(np.array([pixel]), M, get_fuzzy_set=True)

        # Imposta la figura
        plt.figure(figsize=(20, 5))

        # Traccia il set fuzzy
        plt.plot(x, output_fuzzy_set, 'k-', label='FuzzySet', linewidth=2)

        # Linea tratteggiata per M
        plt.plot((M, M), (0, 1), 'm--', label='M', linewidth=2)

        # Linea tratteggiata per il pixel di input
        plt.plot((pixel, pixel), (0, 1), 'g--', label='Input', linewidth=2)

        # Linea tratteggiata per il centroide
        plt.plot((centroid, centroid), (0, 1), 'r--', label='Output', linewidth=2)

        # Area colorata sotto il set fuzzy
        plt.fill_between(x, np.zeros(len(x)), output_fuzzy_set, color=(.9, .9, .9, .9))

        # Aggiunge la legenda
        plt.legend()

        # Imposta i limiti dell'asse x e y
        plt.xlim(-50, 305)
        plt.ylim(0.0, 1.01)

        # Etichette e titolo
        plt.xlabel('Output pixel intensity')
        plt.ylabel('Degree of membership')
        plt.title(f'input_pixel_intensity = {pixel}\nM = {M}')

        # Mostra il grafico
        plt.show()


# plots the input-output mapping for different values of M
def plot_io_mapping(means=(64, 96, 128, 160, 192)):

    plt.figure(figsize=(25, 5))  # Imposta la dimensione della figura

    x = np.arange(256)  # Input range: 0-255

    for i in range(len(means)):
        M = means[i]  # Prende il valore di M dalla lista

        # Misura il tempo di esecuzione per il calcolo degli output
        start_time = time.time()
        
        # Calcola gli output usando la funzione Infer per ogni valore di x
        y = np.array([Infer(np.array([xi]), M) for xi in x])
        
        end_time = time.time()
        print(f"Tempo di esecuzione per M = {M}: {end_time - start_time:.2f} secondi")

        # Crea il subplot per ciascun valore di M
        plt.subplot(1, len(means), i+1)
        plt.plot(x, y, 'r-', label='IO mapping')  # Traccia la mappatura IO in rosso

        # Imposta i limiti degli assi
        plt.xlim(0, 256)
        plt.ylim(-50, 355)

        # Etichette degli assi
        plt.xlabel('Input Intensity')
        plt.ylabel('Output Intensity')

        # Titolo per ogni subplot
        plt.title(f'M = {M}')

    plt.tight_layout()  # Migliora la disposizione dei subplots
    plt.show()  # Mostra il grafico


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


# comparison between original and equalized image
def show_difference(img, fce):
    # Imposta la figura per mostrare le immagini affiancate
    plt.figure(figsize=(10, 5))

    # Mostra l'immagine originale
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    # Mostra l'immagine originale
    plt.subplot(1, 2, 2)
    plt.imshow(fce, cmap='gray')
    plt.title('Denoised Image with fuzzy')
    plt.axis('off')

