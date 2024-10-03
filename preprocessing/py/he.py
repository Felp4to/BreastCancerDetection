import cv2
import numpy as np
import matplotlib.pyplot as plt


# histogram equalization is a technique in image processing used to enhance the contrast 
# of an image by redistributing the intensity values. The goal is to spread out the most
# frequent intensity values over a wider range, making the image more clear and distinct
# in terms of features.
def apply_histogram_equalization(path, histogram, show_result):

    # read a image using imread
    img = cv2.imread(path,  cv2.IMREAD_GRAYSCALE)
    # equalize image
    equalized = cv2.equalizeHist(img)

    # shows histogram and cdf
    if(histogram == 1): 
        stamp_histogram(img)
        stamp_histogram(equalized)
    if(show_result):
        show_difference(img, equalized)

    return equalized


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
    plt.title('Immagine equalized')
    plt.axis('off')

















