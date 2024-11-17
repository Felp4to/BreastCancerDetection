import pandas as pd
import numpy as np
import glob
import cv2
import os
import tqdm
import random
import tqdm as tqdm
import constants as cns
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import image_processing as pre



def load_csv():
    dicom_data = pd.read_csv(cns.PATH_DICOM_DATA)
    calc_case_df = pd.read_csv(cns.PATH_CALC_CASE_DF)
    mass_case_df = pd.read_csv(cns.PATH_MASS_CASE_DF)

    return dicom_data, calc_case_df, mass_case_df

def clean_csv(dicom_data, calc_case_df, mass_case_df):
     # delete some columns
    dicom_cleaned_data = dicom_data.copy()
    dicom_cleaned_data.drop(['PatientBirthDate','AccessionNumber','Columns','ContentDate',
                             'ContentTime','PatientSex','PatientBirthDate','ReferringPhysicianName',
                             'Rows','SOPClassUID','SOPInstanceUID','StudyDate','StudyID',
                             'StudyInstanceUID','StudyTime','InstanceNumber','SeriesInstanceUID',
                             'SeriesNumber'],axis =1, inplace=True)
    # fill empty field with next value
    dicom_cleaned_data['SeriesDescription'] = dicom_cleaned_data['SeriesDescription'].bfill()
    dicom_cleaned_data['Laterality'] = dicom_cleaned_data['Laterality'].bfill()

    # rename columns
    Data_cleaning_1 = calc_case_df.copy()
    Data_cleaning_1 = Data_cleaning_1.rename(columns={'calc type':'calc_type'})
    Data_cleaning_1 = Data_cleaning_1.rename(columns={'calc distribution':'calc_distribution'})
    Data_cleaning_1 = Data_cleaning_1.rename(columns={'image view':'image_view'})
    Data_cleaning_1 = Data_cleaning_1.rename(columns={'left or right breast':'left_or_right_breast'})
    Data_cleaning_1 = Data_cleaning_1.rename(columns={'breast density':'breast_density'})
    Data_cleaning_1 = Data_cleaning_1.rename(columns={'abnormality type':'abnormality_type'})
    Data_cleaning_1['pathology'] = Data_cleaning_1['pathology'].astype('category')
    Data_cleaning_1['calc_type'] = Data_cleaning_1['calc_type'].astype('category')
    Data_cleaning_1['calc_distribution'] = Data_cleaning_1['calc_distribution'].astype('category')
    Data_cleaning_1['abnormality_type'] = Data_cleaning_1['abnormality_type'].astype('category')
    Data_cleaning_1['image_view'] = Data_cleaning_1['image_view'].astype('category')
    Data_cleaning_1['left_or_right_breast'] = Data_cleaning_1['left_or_right_breast'].astype('category')
    Data_cleaning_1['calc_type'].fillna(method = 'bfill', axis = 0, inplace=True)
    Data_cleaning_1['calc_distribution'].fillna(method = 'bfill', axis = 0, inplace=True)

    # rename columns
    Data_cleaning_2 = mass_case_df.copy()
    Data_cleaning_2 = Data_cleaning_2.rename(columns={'mass shape':'mass_shape'})
    Data_cleaning_2 = Data_cleaning_2.rename(columns={'left or right breast':'left_or_right_breast'})
    Data_cleaning_2 = Data_cleaning_2.rename(columns={'mass margins':'mass_margins'})
    Data_cleaning_2 = Data_cleaning_2.rename(columns={'image view':'image_view'})
    Data_cleaning_2 = Data_cleaning_2.rename(columns={'abnormality type':'abnormality_type'})
    Data_cleaning_2['left_or_right_breast'] = Data_cleaning_2['left_or_right_breast'].astype('category')
    Data_cleaning_2['image_view'] = Data_cleaning_2['image_view'].astype('category')
    Data_cleaning_2['mass_margins'] = Data_cleaning_2['mass_margins'].astype('category')
    Data_cleaning_2['mass_shape'] = Data_cleaning_2['mass_shape'].astype('category')
    Data_cleaning_2['abnormality_type'] = Data_cleaning_2['abnormality_type'].astype('category')
    Data_cleaning_2['pathology'] = Data_cleaning_2['pathology'].astype('category')
    Data_cleaning_2['mass_shape'].fillna(method = 'bfill', axis = 0, inplace=True) 
    Data_cleaning_2['mass_margins'].fillna(method = 'bfill', axis = 0, inplace=True) 


def prep_and_split(preprocessing_method, perc_dataset):
    dicom_data, calc_case_df, mass_case_df = load_csv()
    cropped_images = dicom_data[dicom_data.SeriesDescription == 'cropped images'].image_path
    cropped_images = cropped_images.apply(lambda x: x.replace('CBIS-DDSM/jpeg', cns.PATH_IMAGES))
    full_mammogram_images = dicom_data[dicom_data.SeriesDescription == 'full mammogram images'].image_path
    full_mammogram_images = full_mammogram_images.apply(lambda x: x.replace('CBIS-DDSM/jpeg', cns.PATH_IMAGES))
    ROI_mask_images = dicom_data[dicom_data.SeriesDescription == 'ROI mask images'].image_path
    ROI_mask_images = ROI_mask_images.apply(lambda x: x.replace('CBIS-DDSM/jpeg', cns.PATH_IMAGES))
    
    # separate images in two array, cancer and non_cancer
    breast_imgs = glob.glob(cns.PATH_PNGS, recursive = True)
    non_cancer_imgs = []
    cancer_imgs = []
    for img in breast_imgs:
        if img[-5] == '0' :
            non_cancer_imgs.append(img)
        
        elif img[-5] == '1' :
            cancer_imgs.append(img)

    # print numbers of elements for each array
    non_cancer_num = len(non_cancer_imgs)  # No cancer
    cancer_num = len(cancer_imgs)   # Cancer 
    total_img_num = non_cancer_num + cancer_num 
    print('Number of Images of no cancer: {}' .format(non_cancer_num))   # images of Non cancer
    print('Number of Images of cancer : {}' .format(cancer_num))   # images of cancer 
    print('Total Number of Images : {}' .format(total_img_num))
    
    # Esegui il campionamento delle immagini
    some_non_img = random.sample(non_cancer_imgs, len(non_cancer_imgs))
    some_can_img = random.sample(cancer_imgs, len(cancer_imgs))
    
    # Inizializza array vuoti per memorizzare i dati delle immagini e le etichette
    non_img_arr = []  # Array per immagini non tumorali
    can_img_arr = []  # Array per immagini tumorali

    num1 = int(len(some_non_img) * float(perc_dataset))
    num2 = int(len(some_can_img) * float(perc_dataset))
    some_non_img = some_non_img[:num1]
    some_can_img = some_can_img[:num2]

    # Loop attraverso ciascuna immagine nella lista 'some_non_img'
    print("Caricamento immagini non tumorali...")
    for img in tqdm.tqdm(some_non_img, desc="Non-cancer images"):
        processed_img = pre.image_processor(img, (50, 50, 3), preprocessing_method)
        # Leggi l'immagine in modalità a colori
        #n_img = cv2.imread(img, cv2.IMREAD_COLOR)
        # Ridimensiona l'immagine a una dimensione fissa (50x50 pixel) utilizzando l'interpolazione lineare
        #n_img_size = cv2.resize(n_img, (50, 50), interpolation=cv2.INTER_LINEAR)
        # Aggiungi l'immagine ridimensionata e l'etichetta 0 (indicante non-tumore) all'array 'non_img_arr'
        #non_img_arr.append([n_img_size, 0])
        non_img_arr.append([processed_img, 0])

    # Loop attraverso ciascuna immagine nella lista 'some_can_img'
    print("Caricamento immagini tumorali...")
    for img in tqdm.tqdm(some_can_img, desc="Cancer images"):
        processed_img = pre.image_processor(img, (50, 50, 3), preprocessing_method)
        # Leggi l'immagine in modalità a colori
        #c_img = cv2.imread(img, cv2.IMREAD_COLOR)
        # Ridimensiona l'immagine a una dimensione fissa (50x50 pixel) utilizzando l'interpolazione lineare
        #c_img_size = cv2.resize(c_img, (50, 50), interpolation=cv2.INTER_LINEAR)
        # Aggiungi l'immagine ridimensionata e l'etichetta 1 (indicante tumore) all'array 'can_img_arr'
        #can_img_arr.append([c_img_size, 1])
        can_img_arr.append([processed_img, 1])


    # Mostra l'immagine
    plt.imshow(non_img_arr[0][0], cmap='gray')  # Usa 'gray' se l'immagine è in scala di grigi
    plt.axis('off')  # Nasconde gli assi
    plt.show()

    X = []  # List for image data
    y = []  # List for labels

    breast_img_arr = non_img_arr + can_img_arr

    # Shuffle the elements in the 'breast_img_arr' array randomly
    random.shuffle(breast_img_arr)

    # Loop through each element (feature, label) in the shuffled 'breast_img_arr'
    for feature, label in breast_img_arr:
        # Append the image data (feature) to the 'X' list
        X.append(feature)
        # Append the label to the 'y' list
        y.append(label)

    # Convert the lists 'X' and 'y' into NumPy arrays
    X = np.array(X)
    y = np.array(y)

    print(len(X))   

    # TESTING
    #X = np.array(X)[:1000]
    #y = np.array(y)[:1000]

    # Print the shape of the 'X' array
    print('X shape: {}'.format(X.shape))

    return splitting(X, y, 0.2, preprocessing_method)



def splitting(X, y, perc_test_val, preprocessing_method):
    # First, split the data into 80% train and 20% temporary (for validation and test sets)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=perc_test_val, stratify=y, random_state=42)

    # Now split the temporary set into validation (10% of total) and test (10% of total)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

    # Convert the categorical labels in 'y_train', 'y_val', and 'y_test' to one-hot encoded format
    y_train = to_categorical(y_train, 2)
    y_val = to_categorical(y_val, 2)
    y_test = to_categorical(y_test, 2)

    # create folder if is not exist
    root = os.path.join(cns.PATH_PARTITIONS_ROOT, preprocessing_method)
    os.makedirs(root, exist_ok=True)

    train_X_path = os.path.join(root, "X_train.npy")
    train_y_path = os.path.join(root, "y_train.npy")
    test_X_path = os.path.join(root, "X_test.npy")
    test_y_path = os.path.join(root, "y_test.npy")
    val_X_path = os.path.join(root, "X_val.npy")
    val_y_path = os.path.join(root, "y_val.npy")

    # Save X_test and y_test to a file
    np.save(train_X_path, X_train)
    np.save(train_y_path, y_train)
    np.save(test_X_path, X_test)
    np.save(test_y_path, y_test)
    np.save(val_X_path, X_val)
    np.save(val_y_path, y_val)

    # Output the shapes to confirm the splitting
    print('X_train shape : {}'.format(X_train.shape))
    print('X_val shape   : {}'.format(X_val.shape))
    print('X_test shape  : {}'.format(X_test.shape))
    print('y_train shape : {}'.format(y_train.shape))
    print('y_val shape   : {}'.format(y_val.shape))
    print('y_test shape  : {}'.format(y_test.shape))

    return X_train, X_test, X_val, y_train, y_test, y_val
