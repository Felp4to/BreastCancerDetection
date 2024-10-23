import cv2
import numpy as np
import matplotlib.pyplot as plt


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
def apply_equalization_clahe_Lab(path, target_size, histogram=0, show_result=1, clipLimit=2.0, tileGridSize=(8, 8)):

    # upload gray scale image
    image = cv2.imread(path)

    # Converti l'immagine dallo spazio RGB allo spazio Lab
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    
    # Dividi l'immagine nei tre canali: L, a, b
    l_channel, a_channel, b_channel = cv2.split(lab_image)
    
    # Crea l'oggetto CLAHE
    clahe = cv2.createCLAHE(clipLimit, tileGridSize)
    
    # Applica CLAHE al canale di luminosità (L)
    l_channel_clahe = clahe.apply(l_channel)
    
    # Unisci i canali modificati per ricostruire l'immagine Lab
    lab_image_clahe = cv2.merge((l_channel_clahe, a_channel, b_channel))
    
    # Converti nuovamente l'immagine dallo spazio Lab allo spazio RGB
    clahe_img = cv2.cvtColor(lab_image_clahe, cv2.COLOR_Lab2BGR)

    # Ottieni le dimensioni dell'immagine
    height, width, channels = image.shape

    #resize
    #clahe_img = cv2.resize(clahe_img, (target_size[1], target_size[0]))
    if(target_size != None):
        clahe_img = cv2.resize(clahe_img, (target_size[1], target_size[0]), interpolation=cv2.INTER_LINEAR)

    if(histogram): 
        show_histogram(image, "original")
        show_histogram(clahe_img, "clahe")
    if(show_result):
        show_difference(image, clahe_img)
    
    return clahe_img


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















