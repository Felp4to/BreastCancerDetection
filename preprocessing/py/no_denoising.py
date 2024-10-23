import cv2
import matplotlib.pyplot as plt


# no-denoising
def no_denoising(image_path, target_size):
    """Preprocess images for CNN model"""
    #print(image_path)
    # absolute_image_path = os.path.abspath(image_path)
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if(target_size != None):
        image = cv2.resize(image, (target_size[1], target_size[0]), interpolation=cv2.INTER_LINEAR)
    image_array = image / 255.0
    
    return image_array
