import cv2
import numpy as np
import matplotlib.pyplot as plt


# histogram equalization is a technique in image processing used to enhance the contrast 
# of an image by redistributing the intensity values. The goal is to spread out the most
# frequent intensity values over a wider range, making the image more clear and distinct
# in terms of features.
def apply_histogram_equalization(path, target_size, histogram=0, show_result=0):

    # upload gray scale image
    image = cv2.imread(path)

    # Convert the image from BGR to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Split into H, S, and V channels (we will apply equalization to the V channel)
    h, s, v = cv2.split(hsv_image)
    
    # Apply global histogram equalization on the V (value/brightness) channel
    v_hist_eq = cv2.equalizeHist(v)

    # Merge the channels back
    hsv_hist_eq = cv2.merge((h, s, v_hist_eq))

    # Convert back to BGR color space
    final_img_hsv = cv2.cvtColor(hsv_hist_eq, cv2.COLOR_HSV2BGR)

    # Resize the image if a target size is provided
    if target_size is not None:
        final_img_hsv = cv2.resize(final_img_hsv, (target_size[1], target_size[0]), interpolation=cv2.INTER_LINEAR)

    # Optionally show histograms for comparison
    if histogram: 
        #stamp_histogram(image)
        #stamp_histogram(final_img_hsv)
        stamp_histogram_2(image, final_img_hsv, "Original", "Histogram Equalized")
    
    # Optionally show the original and processed image for comparison
    if show_result:
        show_difference(image, final_img_hsv)

    return final_img_hsv


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


def stamp_histogram_2(img1, img2, label1, label2):
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
    plt.title('Immagine Original')
    plt.axis('off')

    # Mostra l'immagine originale
    plt.subplot(1, 2, 2)
    plt.imshow(equalized, cmap='gray')
    plt.title('Immagine Equalized')
    plt.axis('off')

















