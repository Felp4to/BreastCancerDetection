import cv2
import numpy as np
import matplotlib.pyplot as plt


# return equalized image with clahe method
def apply_equalization_clahe(path, histogram, clipLimit=2.0, tileGridSize=(8, 8)):

    # upload gray scale image
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    # create CLAHE object with limit contrast
    clahe = cv2.createCLAHE(clipLimit, tileGridSize)

    # apply CLAHE to the image
    clahe_img = clahe.apply(img)

    if(histogram == 1): 
        show_histogram(img, "original")
        show_histogram(clahe, "clahe")


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


# comparison between original and equalized image
def show_difference(img, equalized):
    # Imposta la figura per mostrare le immagini affiancate
    plt.figure(figsize=(10, 5))

    # Mostra l'immagine originale
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Immagine Original')
    plt.axis('off')

    # Mostra l'immagine originale
    plt.subplot(1, 2, 2)
    plt.imshow(equalized, cmap='gray')
    plt.title('Equalized with clahe')
    plt.axis('off')















