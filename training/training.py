import os
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
import constants as cns
import models as mod
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score



def load_partitions(preprocessing_method, colab=0):
    path = os.path.join(cns.PATH_PARTITIONS_COLAB if colab else cns.PATH_PARTITIONS_ROOT, preprocessing_method)
    
    # Define paths for all files
    paths = {
        'X_train': os.path.join(path, 'X_train.npy'),
        'X_test': os.path.join(path, 'X_test.npy'),
        'X_val': os.path.join(path, 'X_val.npy'),
        'y_train': os.path.join(path, 'y_train.npy'),
        'y_test': os.path.join(path, 'y_test.npy'),
        'y_val': os.path.join(path, 'y_val.npy')
    }

    # Try loading each file, handle errors if any file is missing
    try:
        X_train = np.load(paths['X_train'], allow_pickle=True)
        X_val = np.load(paths['X_val'], allow_pickle=True)
        X_test = np.load(paths['X_test'], allow_pickle=True)
        y_train = np.load(paths['y_train'], allow_pickle=True)
        y_test = np.load(paths['y_test'], allow_pickle=True)
        y_val = np.load(paths['y_val'], allow_pickle=True)
    except FileNotFoundError as e:
        print(f"Error loading files: {e}")
        return None  # or handle as needed
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def training(root, preprocessing_method="no_denoising", epochs = 25, batch_size = 75, colab=0):
    # load partitions
    X_train, X_val, X_test, y_train, y_val, y_test = load_partitions(preprocessing_method, colab)

    datasets = {
        "X_test": X_test,
        "y_test": y_test,
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val
    }

    for name, data in datasets.items():
        print(f"{name} shape: {data.shape}")
    
    # Data augmentation
    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # Create data generators for training and testing
    train_datagen = datagen.flow(X_train, y_train, batch_size=32)
    test_datagen = datagen.flow(X_test, y_test, batch_size=32, shuffle=False)
    val_datagen = datagen.flow(X_val, y_val, batch_size=32, shuffle=False)

    # Define an EarlyStopping callback
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='accuracy',          # Monitor the validation loss
        patience=5,                  # Number of epochs with no improvement after which training will be stopped
        min_delta=1e-7,              # Minimum change in the monitored quantity to be considered an improvement
        restore_best_weights=True,   # Restore model weights from the epoch with the best value of monitored quantity
    )

    # Define a ReduceLROnPlateau callback
    plateau = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',   # Monitor the validation loss
        factor=0.5,           # Factor by which the learning rate will be reduced (new_lr = lr * factor)
        patience=2,           # Number of epochs with no improvement after which learning rate will be reduced
        min_delta=1e-7,       # Minimum change in the monitored quantity to trigger a learning rate reduction
        cooldown=0,           # Number of epochs to wait before resuming normal operation after learning rate reduction
        verbose=1             # Verbosity mode (1: update messages, 0: no messages)
    )

    # create model
    model = mod.create_model_4()

    # training
    history = model.fit(train_datagen, 
                        validation_data = val_datagen,
                        epochs = epochs, 
                        batch_size = batch_size, 
                        callbacks=[early_stopping, plateau])
    
    # plot accuracy and loss function
    plot_acc_loss(history, preprocessing_method, colab)

    # valuate model
    accuracy, conf_matrix = valuate_model(model, X_test, y_test, test_datagen)

    # save results
    save_results(history, model, accuracy, conf_matrix, preprocessing_method, colab)

    return history, model


def save_results(history, model, accuracy, conf_matrix, preprocessing_method, colab):
    # define root
    root = os.path.join(cns.PATH_PARTITIONS_COLAB if colab else cns.PATH_MODELS, preprocessing_method)

    # show confusion matrix
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    path = os.path.join(root, "confusion_matrix.jpeg")
    plt.savefig(path, format='jpeg') 
    plt.close() 

    path = os.path.join(root, "accuracy.txt")
    with open(path, "w") as file:
        file.write(str(accuracy))
    
    # save model
    path = os.path.join(root, "model.keras")
    model.save(path)

    # save history
    path = os.path.join(root, "training_history.pkl")
    with open(path, 'wb') as f:
        pickle.dump(history.history, f)

    # convert chronology in dataframe
    history_df = pd.DataFrame(history.history)

    # save chronology in a csv file
    history_csv_path = os.path.join(root, "history.csv")
    history_df.to_csv(history_csv_path, index=False)


# evaluate model
def valuate_model(model, X_test, y_test, test_datagen):

    # evaluate test set
    test_loss, test_accuracy = model.evaluate(test_datagen)
    print(f"Test Loss: {test_loss}")
    print(f"Test Accuracy: {test_accuracy}")
    
    # prediction
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)     
    y_true = np.argmax(y_test, axis=1)              

    # calculate confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred_classes)

    # report
    print("Classification Report:")
    print(classification_report(y_true, y_pred_classes, zero_division=1))

    # accuracy
    accuracy = accuracy_score(y_true, y_pred_classes)
    print(f"Accuracy: {accuracy}")

    return accuracy, conf_matrix



# plot accuracy and loss function
def plot_acc_loss(history, preprocessing_method, colab):
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    train_accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']

    epochs_range = range(1, len(train_loss) + 1)

    if colab == 0:
        root = os.path.join(cns.PATH_MODELS, preprocessing_method)
    else:
        root = os.path.join(cns.PATH_PARTITIONS_COLAB, preprocessing_method)  

    # loss graphic
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    # accuracy graphic
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, train_accuracy, label='Training Accuracy')
    plt.plot(epochs_range, val_accuracy, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    # Define the path for saving the plot
    path = os.path.join(root, "acc_loss_plot.png")  # Change to PNG for better quality
    print("Saving plot to:", path)  # Check the path

    # Save and display the plot
    plt.savefig(path)   
    plt.show()

    plt.close()


if __name__ == "__main__":
    training()

















