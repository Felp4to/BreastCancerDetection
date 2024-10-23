import os
import cv2
import numpy as np
import pandas as pd 
from tqdm import tqdm
import seaborn as sns
import tensorflow as tf
import constants as cns
import preprocessing2 as pre
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical  # type: ignore



# load csv file
def load_csv():
    df_meta = pd.read_csv(cns.META)
    df_metadata = pd.read_csv(cns.METADATA)
    df_dicom = pd.read_csv(cns.DICOM_INFO)
    df_mass_train = pd.read_csv(cns.MASS_CASE_DESCRIPTION_TRAIN_SET)
    df_mass_test = pd.read_csv(cns.MASS_CASE_DESCRIPTION_TEST_SET)
    df_calc_train = pd.read_csv(cns.CALC_CASE_DESCRIPTION_TRAIN_SET)
    df_calc_test = pd.read_csv(cns.CALC_CASE_DESCRIPTION_TEST_SET)

    return df_meta, df_metadata, df_dicom, df_mass_train, df_mass_test, df_calc_train, df_calc_test


# fix image paths
def fix_image_path(data, full_mammo_dict, cropped_images_dict):
    """correct dicom paths to correct image paths"""
    for index, img in enumerate(data.values):
        img_name = img[11].split("/")[2]
        data.iloc[index,11] = full_mammo_dict[img_name]
        img_name = img[12].split("/")[2]
        data.iloc[index,12] = cropped_images_dict[img_name]


def rename_columns(df_mass_train, df_mass_test):
    # rename columns
    df_mass_train = df_mass_train.rename(columns={'left or right breast': 'left_or_right_breast',
                                            'image view': 'image_view',
                                            'abnormality id': 'abnormality_id',
                                            'abnormality type': 'abnormality_type',
                                            'mass shape': 'mass_shape',
                                            'mass margins': 'mass_margins',
                                            'image file path': 'image_file_path',
                                            'cropped image file path': 'cropped_image_file_path',
                                            'ROI mask file path': 'ROI_mask_file_path'})
    
    df_mass_test = df_mass_test.rename(columns={'left or right breast': 'left_or_right_breast',
                                           'image view': 'image_view',
                                           'abnormality id': 'abnormality_id',
                                           'abnormality type': 'abnormality_type',
                                           'mass shape': 'mass_shape',
                                           'mass margins': 'mass_margins',
                                           'image file path': 'image_file_path',
                                           'cropped image file path': 'cropped_image_file_path',
                                           'ROI mask file path': 'ROI_mask_file_path'})
    
    return df_mass_train, df_mass_test


# draw distribution
def draw_distribution(df_mass_train):
    # pathology distributions
    value = df_mass_train['pathology'].value_counts()
    plt.figure(figsize=(8,6))
    plt.pie(value, labels=value.index, autopct='%1.1f%%')
    plt.title('Breast Cancer Mass Types', fontsize=14)
    plt.savefig('./pathology_distributions_red.png')
    plt.show()


# draw density
def draw_density(df_mass_train):
    # breast density against pathology
    plt.figure(figsize=(8,6))
    sns.countplot(df_mass_train, x='breast_density', hue='pathology')
    plt.title('Breast Density vs Pathology\n\n1: fatty || 2: Scattered Fibroglandular Density\n3: Heterogenously Dense || 4: Extremely Dense',
            fontsize=14)
    plt.xlabel('Density Grades')
    plt.ylabel('Count')
    plt.legend()
    plt.savefig('./density_pathology_red.png')
    plt.show()


# create function to display images
def display_images(column, number, df_mass_train):
    # Display some images
    import matplotlib.image as mpimg
    # create figure and axes
    number_to_visualize = number
    rows = 1
    cols = number_to_visualize
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5))
    
    # Loop through rows and display images
    for index, row in df_mass_train.head(number_to_visualize).iterrows():
        image_path = row[column]
        #display(image_path)
        image = mpimg.imread(image_path)
        ax = axes[index]
        ax.imshow(image, cmap='gray')
        ax.set_title(f"{row['pathology']}")
        ax.axis('off')
    plt.tight_layout()
    plt.show()


# return dataframe ready for the partitioning
def data_processing():
    # read csv files
    df_meta, df_metadata, df_dicom, df_mass_train, df_mass_test, df_calc_train, df_calc_test = load_csv()

    cropped_images = df_dicom[df_dicom.SeriesDescription=='cropped images'].image_path
    full_mammo = df_dicom[df_dicom.SeriesDescription=='full mammogram images'].image_path
    roi_img = df_dicom[df_dicom.SeriesDescription=='ROI mask images'].image_path

    # set correct image path for image types
    imdir = '../../dataset/'
    cropped_images = [imdir + s for s in cropped_images]
    full_mammo = [imdir + s for s in full_mammo]
    roi_img = [imdir + s for s in roi_img]

    #display(cropped_images)
    #display(full_mammo)
    #display(roi_img)

    # organize image paths
    full_mammo_dict = dict()
    cropped_images_dict = dict()
    roi_img_dict = dict()

    for dicom in full_mammo:
        key = dicom.split("/")[5]
        full_mammo_dict[key] = dicom
    for dicom in cropped_images:
        key = dicom.split("/")[5]
        cropped_images_dict[key] = dicom
    for dicom in roi_img:
        key = dicom.split("/")[5]
        roi_img_dict[key] = dicom

    # apply to datasets
    #fix_image_path(df_mass_train, full_mammo_dict, cropped_images_dict)
    #fix_image_path(df_mass_test, full_mammo_dict, cropped_images_dict)

    """correct dicom paths to correct image paths"""
    for index, img in enumerate(df_mass_train.values):
        img_name = img[11].split("/")[2]
        df_mass_train.iloc[index,11] = full_mammo_dict[img_name]
        img_name = img[12].split("/")[2]
        df_mass_train.iloc[index,12] = cropped_images_dict[img_name]

    """correct dicom paths to correct image paths"""
    for index, img in enumerate(df_mass_test.values):
        img_name = img[11].split("/")[2]
        df_mass_test.iloc[index,11] = full_mammo_dict[img_name]
        img_name = img[12].split("/")[2]
        df_mass_test.iloc[index,12] = cropped_images_dict[img_name]

    # rename columns
    df_mass_train, df_mass_test = rename_columns(df_mass_train, df_mass_test)
    
    # fill in missing values using the backwards fill method
    df_mass_train['mass_shape'] = df_mass_train['mass_shape'].bfill()
    df_mass_train['mass_margins'] = df_mass_train['mass_margins'].bfill()


    print(f'Shape of mass_train: {df_mass_train.shape}')
    print(f'Shape of mass_test: {df_mass_test.shape}')

    
    # fill in missing values using the backwards fill method
    df_mass_test['mass_margins'] = df_mass_test['mass_margins'].bfill()

    # draw graphics
    #draw_distribution(df_mass_train)
    #draw_density(df_mass_train)

    # display some images
    #print('Full Mammograms:\n')
    #display_images('image_file_path', 5, df_mass_train)
    #print('Cropped Mammograms:\n')
    #display_images('cropped_image_file_path', 5, df_mass_train)

    # Merge datasets
    full_mass = pd.concat([df_mass_train, df_mass_test], axis=0)

    return full_mass


# apply pre-processing to the image
def preprocessing(full_mass, target_size, type = 'A'):
    if type == 'A':
        print("Denoising")
    elif type == 'B':
        print("Clahe method")
    elif type == 'C':
        print("Hinstogram equalization")
    elif type == 'D':
        print("Fuzzy logic")
    elif type == 'E':
        print("Wavelet")

    # apply preprocessing
    processed_images = []
    for x in tqdm(full_mass['image_file_path']):
        processed_images.append(pre.image_processor(x, target_size, type))

    return processed_images


def generate_partitions(type = 'no_denoising'): 
    full_mass = data_processing()
    
    # Define the target size
    target_size = (224, 224, 3)

    # Apply preprocessor to train data
    #tqdm.pandas()
    #full_mass['processed_images'] = full_mass['image_file_path'].apply(lambda x: pre.image_processor(x, target_size, type))

    # Imposta tqdm per Pandas
    tqdm.pandas()

    # Applica la funzione con una barra di avanzamento
    full_mass['processed_images'] = full_mass['image_file_path'].progress_apply(
        lambda x: pre.image_processor(x, target_size, type)
    )

    #processed_images = preprocessing(full_mass, target_size, type)

    # Assegna il risultato
    #full_mass['processed_images'] = processed_images

    # Create a binary mapper
    class_mapper = {'MALIGNANT': 1, 'BENIGN': 0, 'BENIGN_WITHOUT_CALLBACK': 0}

    # Convert the processed_images column to an array
    X_resized = np.array(full_mass['processed_images'].tolist())

    # Set the option to opt-in to the future behavior
    pd.set_option('future.no_silent_downcasting', True)

    # Your original code
    full_mass['labels'] = full_mass['pathology'].replace(class_mapper)

    # Check the number of classes
    num_classes = len(full_mass['labels'].unique())

    X_train, X_test, y_train, y_test = train_test_split(X_resized, full_mass['labels'].values, test_size = 0.2, random_state = 42, stratify=full_mass['labels'].values)

    y_train = to_categorical(y_train, 2)
    y_test = to_categorical(y_test, 2)

    print('X_train shape : {}' .format(X_train.shape))
    print('X_test shape : {}' .format(X_test.shape))
    print('y_train shape : {}' .format(y_train.shape))
    print('y_test shape : {}' .format(y_test.shape))


    path = os.path.join('..', '..', 'partitions', type)
    if not os.path.exists(path):
        os.makedirs(path)
    path_X_train = os.path.join(path, 'X_train.npy')
    path_X_test = os.path.join(path, 'X_test.npy')
    path_y_train = os.path.join(path, 'y_train.npy')
    path_y_test = os.path.join(path, 'y_test.npy')

    # Salvare le singole partizioni
    np.save(path_X_train, X_train)
    np.save(path_X_test, X_test)
    np.save(path_y_train, y_train)
    np.save(path_y_test, y_test)

    return X_train, X_test, y_train, y_test


def generate_partitions(type = 'no_denoising'):
    full_mass = data_processing()
    
    # Define the target size
    target_size = (224, 224, 3)

    # Imposta tqdm per Pandas
    tqdm.pandas()

    # Applica la funzione con una barra di avanzamento
    full_mass['processed_images'] = full_mass['image_file_path'].progress_apply(
        lambda x: pre.image_processor(x, target_size, type)
    )

    # Create a binary mapper
    class_mapper = {'MALIGNANT': 1, 'BENIGN': 0, 'BENIGN_WITHOUT_CALLBACK': 0}

    # Convert the processed_images column to an array
    X_resized = np.array(full_mass['processed_images'].tolist())

    # Set the option to opt-in to the future behavior
    pd.set_option('future.no_silent_downcasting', True)

    # Label mapping
    full_mass['labels'] = full_mass['pathology'].replace(class_mapper)

    # First split: 80% training and 20% test+validation
    X_train, X_test_val, y_train, y_test_val = train_test_split(X_resized, full_mass['labels'].values, test_size=0.3, random_state=42)

    # Second split: 10% validation (which is half of the remaining 20%) and 10% test
    X_test, X_val, y_test, y_val = train_test_split(X_test_val, y_test_val, test_size=0.5, random_state=42) 

    # Convert labels to categorical
    y_train = to_categorical(y_train, 2)
    y_val = to_categorical(y_val, 2)
    y_test = to_categorical(y_test, 2)

    # Print shapes
    print('X_train shape : {}'.format(X_train.shape))
    print('X_val shape : {}'.format(X_val.shape))
    print('X_test shape : {}'.format(X_test.shape))
    print('y_train shape : {}'.format(y_train.shape))
    print('y_val shape : {}'.format(y_val.shape))
    print('y_test shape : {}'.format(y_test.shape))

    # Create directory if it doesn't exist
    path = os.path.join('..', '..', 'partitions', type)
    if not os.path.exists(path):
        os.makedirs(path)

    # Define paths for saving
    path_X_train = os.path.join(path, 'X_train.npy')
    path_X_val = os.path.join(path, 'X_val.npy')
    path_X_test = os.path.join(path, 'X_test.npy')
    path_y_train = os.path.join(path, 'y_train.npy')
    path_y_val = os.path.join(path, 'y_val.npy')
    path_y_test = os.path.join(path, 'y_test.npy')

    # Save partitions
    np.save(path_X_train, X_train)
    np.save(path_X_val, X_val)
    np.save(path_X_test, X_test)
    np.save(path_y_train, y_train)
    np.save(path_y_val, y_val)
    np.save(path_y_test, y_test)

    return X_train, X_val, X_test, y_train, y_val, y_test


def save_partitions(type, X_train, X_test, y_train, y_test):
    path = os.path.join('..', '..', 'partitions', type)
    if not os.path.exists(path):
        os.makedirs(path)
    path_X_train = os.path.join(path, 'X_train.npy')
    path_X_test = os.path.join(path, 'X_test.npy')
    path_y_train = os.path.join(path, 'y_train.npy')
    path_y_test = os.path.join(path, 'y_test.npy')

    # Salvare le singole partizioni
    np.save(path_X_train, X_train)
    np.save(path_X_test, X_test)
    np.save(path_y_train, y_train)
    np.save(path_y_test, y_test)


# main
def main():
    X_train, X_test, y_train, y_test = generate_partitions('A')


# main function execute
if __name__ == "__main__":
    
    main()



































