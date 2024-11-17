import cv2
import numpy as np
import matplotlib.pyplot as plt

#
# clipLimit : this parameter sets the threshold for contrast limiting
# tileGridSize : this parameter sets the number of tiles in the row and column (default 8 x 8)
#


# return equalized image with clahe method
def apply_equalization_clahe(path, target_size, histogram=0, show_result=1, clipLimit=2.0, tileGridSize=(8, 8)):

    # upload gray scale image
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    # create CLAHE object with limit contrast
    clahe = cv2.createCLAHE(clipLimit, tileGridSize)

    # apply CLAHE to the image
    clahe_img = clahe.apply(img)

    # ottieni le dimensioni dell'immagine
    height, width = img.shape

    # resize
    if(target_size == None):
        clahe_img = cv2.resize(clahe_img, (width, height))
    else:
        clahe_img = cv2.resize(clahe_img, (target_size[1], target_size[0]), interpolation=cv2.INTER_LINEAR)

    # converti l'immagine in un formato 224x224x3 replicando i valori su 3 canali
    clahe_img_rgb = cv2.cvtColor(clahe_img, cv2.COLOR_GRAY2RGB)

    if(histogram): 
        show_histogram(img, "original")
        show_histogram(clahe_img_rgb, "clahe")
    if(show_result):
        show_difference(img, clahe_img_rgb)

    return clahe_img_rgb


# Funzione per applicare CLAHE a un'immagine a colori nello spazio Lab
def apply_equalization_clahe_Lab(path, target_size, histogram=0, show_result=0, clipLimit=2.0, tileGridSize=(8, 8), threshold_high_light=255):

    # upload gray scale image
    image = cv2.imread(path)

    # Converti l'immagine dallo spazio RGB allo spazio Lab
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    
    # Dividi l'immagine nei tre canali: L, a, b
    l, a, b = cv2.split(lab_image)

    # Crea una maschera per le alte luci (ad es. luminanza > 180)
    # La maschera sarà bianca (255) nelle aree luminose e nera (0) nelle altre
    _, mask = cv2.threshold(l, threshold_high_light, 255, cv2.THRESH_BINARY)

    # Inverti la maschera, in modo da selezionare solo le aree non luminose
    mask_inv = cv2.bitwise_not(mask)

    # Applica CLAHE solo nelle aree più scure (usando la maschera inversa)
    clahe = cv2.createCLAHE(clipLimit, tileGridSize)
    l_clahe = clahe.apply(l)

    # Combina i risultati: CLAHE dove non ci sono alte luci, mantieni l'originale nelle alte luci
    l_combined = cv2.bitwise_and(l, l, mask=mask) + cv2.bitwise_and(l_clahe, l_clahe, mask=mask_inv)

    # Unisci i canali L, a, b con la nuova luminanza
    lab_clahe = cv2.merge((l_combined, a, b))

    # Converti indietro da Lab a BGR
    img_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_Lab2BGR)

    # Normalizza l'immagine nell'intervallo [0, 255]
    #clahe_img = cv2.normalize(clahe_img, None, 0, 255, cv2.NORM_MINMAX)
    #clahe_img = np.clip(clahe_img, 0, 255)

    #resize
    #clahe_img = cv2.resize(clahe_img, (target_size[1], target_size[0]))
    if(target_size != None):
        img_clahe = cv2.resize(img_clahe, (target_size[1], target_size[0]), interpolation=cv2.INTER_LINEAR)

    if(histogram): 
        #show_histogram(image, "original")
        #show_histogram(clahe_img, "clahe")
        show_histograms_2(image, img_clahe, "original", "clahe")
    if(show_result):
        show_difference(image, img_clahe)
    
    return img_clahe


# Funzione per applicare CLAHE a un'immagine a colori nello spazio Lab
def apply_equalization_clahe_HSV(path, target_size, histogram=0, show_result=1, clipLimit=2.0, tileGridSize=(8, 8)):
    # upload gray scale image
    image = cv2.imread(path)

    # Converti l'immagine da BGR a HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Dividi nei canali H, S, e V (valore = luminanza)
    h, s, v = cv2.split(hsv_image)

    # Applica CLAHE sul canale V
    clahe = cv2.createCLAHE(clipLimit, tileGridSize)
    v_clahe = clahe.apply(v)

    # Ricombina i canali
    hsv_clahe = cv2.merge((h, s, v_clahe))

    # Converti di nuovo in BGR
    final_img_hsv = cv2.cvtColor(hsv_clahe, cv2.COLOR_HSV2BGR)

    #resize
    #clahe_img = cv2.resize(clahe_img, (target_size[1], target_size[0]))
    if(target_size != None):
        final_img_hsv = cv2.resize(final_img_hsv, (target_size[1], target_size[0]), interpolation=cv2.INTER_LINEAR)

    if(histogram): 
        show_histogram(image, "original")
        show_histogram(final_img_hsv, "clahe")
    if(show_result):
        show_difference(image, final_img_hsv)

    return final_img_hsv



# show histogram
def show_histogram(img, label):
    hist,bins = np.histogram(img.flatten(),256,[0,256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * float(hist.max()) / cdf.max()
    plt.plot(cdf_normalized, color = 'b')
    plt.hist(img.flatten(),256,[0,256], color = 'r')
    plt.xlim([0,256])
    plt.legend((label,'histogram'), loc = 'upper left')
    plt.show()

def show_histograms_2(img1, img2, label1, label2):
    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # First image histogram and CDF
    hist1, bins1 = np.histogram(img1.flatten(), 256, [0, 256])
    cdf1 = hist1.cumsum()
    cdf_normalized1 = cdf1 * float(hist1.max()) / cdf1.max()
    
    axes[0].plot(cdf_normalized1, color='b')
    axes[0].hist(img1.flatten(), 256, [0, 256], color='r')
    axes[0].set_xlim([0, 256])
    axes[0].set_title(label1)
    axes[0].legend(('CDF', 'Histogram'), loc='upper left')

    # Second image histogram and CDF
    hist2, bins2 = np.histogram(img2.flatten(), 256, [0, 256])
    cdf2 = hist2.cumsum()
    cdf_normalized2 = cdf2 * float(hist2.max()) / cdf2.max()

    axes[1].plot(cdf_normalized2, color='b')
    axes[1].hist(img2.flatten(), 256, [0, 256], color='r')
    axes[1].set_xlim([0, 256])
    axes[1].set_title(label2)
    axes[1].legend(('CDF', 'Histogram'), loc='upper left')

    plt.show()


# comparison between original and equalized image
def show_difference(img, equalized):
    # Imposta la figura per mostrare le immagini affiancate
    plt.figure(figsize=(10, 5))

    # Mostra l'immagine originale
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    # Mostra l'immagine originale
    plt.subplot(1, 2, 2)
    plt.imshow(equalized, cmap='gray')
    plt.title('Equalized Image with clahe')
    plt.axis('off')















