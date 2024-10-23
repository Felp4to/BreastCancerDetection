import os
import pickle
import json
import numpy as np
import tensorflow as tf
import constants as cns
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score



def load_partitions(preprocessing_method):
    path = os.path.join(cns.PATH_PARTITIONS_ROOT, preprocessing_method)
    # Caricare i file salvati
    path_X_train = os.path.join(path, 'X_train.npy')
    path_X_test = os.path.join(path, 'X_test.npy')
    path_X_val = os.path.join(path, 'X_val.npy')
    path_y_train = os.path.join(path, 'y_train.npy')
    path_y_test = os.path.join(path, 'y_test.npy')
    path_y_val = os.path.join(path, 'y_val.npy')
    print(path_X_train)
    print(path_X_test)
    print(path_y_train)
    print(path_y_test)
    print(path_X_val)
    print(path_y_val)
    X_train = np.load(path_X_train, allow_pickle=True)
    X_val = np.load(path_X_val, allow_pickle=True)
    X_test = np.load(path_X_test, allow_pickle=True)
    y_train = np.load(path_y_train, allow_pickle=True)
    y_test = np.load(path_y_test, allow_pickle=True)
    y_val = np.load(path_y_val, allow_pickle=True)

    return X_train, X_val, X_test, y_train, y_val, y_test


def training(model, preprocessing_method="no_denoising"):

    X_train, X_val, X_test, y_train, y_val, y_test = load_partitions(preprocessing_method)

    print("X_test shape:", X_test.shape)
    print("y_test shape:", y_test.shape)
    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
    print("X_val shape:", X_val.shape)
    print("y_val shape:", y_val.shape)
        
    # Data augmentation
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # Create data generators for training and testing
    train_datagen = datagen.flow(X_train, y_train, batch_size=32)
    test_datagen = datagen.flow(X_test, y_test, batch_size=32, shuffle=False)
    val_datagen = datagen.flow(X_val, y_val, batch_size=32, shuffle=False)

    # Define an EarlyStopping callback
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',          # Monitor the validation loss
        patience=5,                  # Number of epochs with no improvement after which training will be stopped
        min_delta=1e-7,              # Minimum change in the monitored quantity to be considered an improvement
        restore_best_weights=True,   # Restore model weights from the epoch with the best value of monitored quantity
    )

    # Define a ReduceLROnPlateau callback
    plateau = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',   # Monitor the validation loss
        factor=0.2,           # Factor by which the learning rate will be reduced (new_lr = lr * factor)
        patience=2,           # Number of epochs with no improvement after which learning rate will be reduced
        min_delta=1e-7,       # Minimum change in the monitored quantity to trigger a learning rate reduction
        cooldown=0,           # Number of epochs to wait before resuming normal operation after learning rate reduction
        verbose=1             # Verbosity mode (1: update messages, 0: no messages)
    )

    history = model.fit(train_datagen,
                    validation_data = val_datagen,
                    epochs = 25,
                    batch_size = 75, 
                    callbacks=[early_stopping, plateau])  

    #history = model.fit(X_train,
    #                y_train,
    #                validation_data = (X_val, y_val),
    #                epochs = 25,
    #                batch_size = 75, 
    #                callbacks=[early_stopping, plateau]) 
    
    with open('training_history.pkl', 'wb') as f:
        pickle.dump(history.history, f)
    
    return history


# create model
def create_model():
    
    # Set a random seed for reproducibility
    tf.random.set_seed(42)

    input_shape=(50, 50, 3)

    # Create a Sequential model
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        # Convolutional layer with 32 filters, a 3x3 kernel, 'same' padding, and ReLU activation
        tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        # MaxPooling layer with a 2x2 pool size and default stride (2)
        tf.keras.layers.MaxPooling2D(strides=2),
        
        # Convolutional layer with 64 filters, a 3x3 kernel, 'same' padding, and ReLU activation
        tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        # MaxPooling layer with a 3x3 pool size and stride of 2
        tf.keras.layers.MaxPooling2D((3, 3), strides=2),
        
        # Convolutional layer with 128 filters, a 3x3 kernel, 'same' padding, and ReLU activation
        tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        # MaxPooling layer with a 3x3 pool size and stride of 2
        tf.keras.layers.MaxPooling2D((3, 3), strides=2),
        
        # Convolutional layer with 128 filters, a 3x3 kernel, 'same' padding, and ReLU activation
        tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        # MaxPooling layer with a 3x3 pool size and stride of 2
        tf.keras.layers.MaxPooling2D((3, 3), strides=2),
        
        # Flatten the output to prepare for fully connected layers
        tf.keras.layers.Flatten(),
        
        # Fully connected layer with 128 units and ReLU activation
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        # Output layer with 2 units (binary classification) and softmax activation
        tf.keras.layers.Dense(2, activation='softmax')
    ])

    # Display a summary of the model architecture
    model.summary()

    # Compile the model with Adam optimizer, binary cross-entropy loss, and accuracy metric
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss='binary_crossentropy',
                metrics=['accuracy'])
        
    return model



# evaluate model
def valuate_model(model, X_test, y_test):

    # evaluate test set
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"Test Loss: {test_loss}")
    print(f"Test Accuracy: {test_accuracy}")

    # prediction
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)     
    y_true = np.argmax(y_test, axis=1)              

    # calculate confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred_classes)

    # show confusion matrix
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.savefig('../models/confusion_matrix.jpg')  
    plt.close() 
    plt.show()

    # report
    print("Classification Report:")
    print(classification_report(y_true, y_pred_classes, zero_division=1))

    # accuracy
    accuracy = accuracy_score(y_true, y_pred_classes)
    print(f"Accuracy: {accuracy}")



# plot accuracy and loss function
def plot_acc_loss(history):
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    train_accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']

    epochs_range = range(1, len(train_loss) + 1)

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
    plt.savefig('../models/acc_loss_plot.jpg')   
    plt.show()

    plt.close() 


if __name__ == "__main__":
    training()

















