import sys
import numpy as np
import os
import io
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, LeakyReLU
from keras.optimizers import Adam, SGD, RMSprop, Nadam
from tensorflow.keras.regularizers import l2
from keras.applications import InceptionResNetV2
from keras.utils import plot_model
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow.keras import layers, models
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import pickle
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.applications import VGG16
from contextlib import redirect_stdout
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from tensorflow.keras.layers import Flatten, Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Dropout, Flatten


def load_partitions(path):
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


# create model
def create_model():

    image_size=224
    input_shape=(image_size, image_size, 3)

    # create model
    model = models.Sequential()
    
    # first convolutional layer
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    
    # secondo convolutional layer
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    # third convolutional layer
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    # fourth convolutional layer
    model.add(layers.Conv2D(256, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    # flatten layer
    model.add(layers.Flatten())
    
    # dense layers
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.5))

    model.add(layers.Dense(128, activation='relu'))
    
    # output layer, softmax function
    model.add(layers.Dense(2, activation='softmax'))   # 2 classes

    # compile model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    # Crea un oggetto StringIO per catturare l'output
    string_buffer = io.StringIO()

    # Usa redirect_stdout per catturare l'output di model.summary()
    with redirect_stdout(string_buffer):
        model.summary()

    # Estrai il contenuto sotto forma di stringa
    summary_string = string_buffer.getvalue()

    # Chiudi il buffer StringIO
    string_buffer.close()

    # save the model
    model_path = os.path.join('../models/', 'model.pkl')
    with open(model_path, 'wb') as file:  
        pickle.dump(model, file)

    # save the summary
    summary_path = os.path.join('../models/', 'model_summary.txt')
    # Scrivi il contenuto del riepilogo in un file di testo
    with open(summary_path, 'w') as file:
        file.write(summary_string)
        
    return model


# create model 2
def create_model_2():
    
    image_size=224
    input_shape=(image_size, image_size, 3)

    # Carica il modello VGG16 preaddestrato (escludendo i top layers)
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(image_size, image_size, 3))

    # Congela i livelli del modello base (non si addestrano)
    for layer in base_model.layers:
        layer.trainable = False

    # Creazione del modello
    model = Sequential()

    # Aggiunge il modello preaddestrato VGG16 come base
    model.add(base_model)

    # Aggiungi appiattimento
    model.add(Flatten())

    # Aggiungi livello denso con 256 neuroni e ReLU
    model.add(Dense(256, activation='relu'))

    # Aggiungi dropout per evitare overfitting
    model.add(Dropout(0.5))

    # Livello finale per la classificazione binaria
    model.add(Dense(2, activation='sigmoid'))

    # Compilazione del modello
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

    # Riassunto del modello
    model.summary()
    
    # Crea un oggetto StringIO per catturare l'output
    string_buffer = io.StringIO()

    # Usa redirect_stdout per catturare l'output di model.summary()
    with redirect_stdout(string_buffer):
        model.summary()

    # Estrai il contenuto sotto forma di stringa
    summary_string = string_buffer.getvalue()

    # Chiudi il buffer StringIO
    string_buffer.close()

    # save the model
    model_path = os.path.join('../models/', 'model.pkl')
    with open(model_path, 'wb') as file:  
        pickle.dump(model, file)

    # save the summary
    summary_path = os.path.join('../models/', 'model_summary.txt')
    # Scrivi il contenuto del riepilogo in un file di testo
    with open(summary_path, 'w') as file:
        file.write(summary_string)
        
    return model


# create model 3
def create_model_3():
    
    image_size=224
    input_shape=(image_size, image_size, 3)

    model = Sequential()

    # Primo blocco convoluzionale
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))  # Aggiungi Dropout anche dopo il MaxPooling

    # Secondo blocco convoluzionale
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # Terzo blocco convoluzionale
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))  # Aggiungi Dropout anche nei blocchi più profondi

    # Flatten per passare ai livelli completamente connessi
    model.add(Flatten())

    # Livelli completamente connessi
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))  # Manteniamo il Dropout alto per i livelli completamente connessi
    model.add(Dense(2, activation='sigmoid'))

    # Compilazione del modello
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

    model.summary()
    
    # Crea un oggetto StringIO per catturare l'output
    string_buffer = io.StringIO()

    # Usa redirect_stdout per catturare l'output di model.summary()
    with redirect_stdout(string_buffer):
        model.summary()

    # Estrai il contenuto sotto forma di stringa
    summary_string = string_buffer.getvalue()

    # Chiudi il buffer StringIO
    string_buffer.close()

    # save the model
    model_path = os.path.join('../models/', 'model.pkl')
    with open(model_path, 'wb') as file:  
        pickle.dump(model, file)

    # save the summary
    summary_path = os.path.join('../models/', 'model_summary.txt')
    # Scrivi il contenuto del riepilogo in un file di testo
    with open(summary_path, 'w') as file:
        file.write(summary_string)
        
    return model



# create model 4
def create_model_4():
    
    image_size = 224
    input_shape = (image_size, image_size, 3)

    # Load InceptionResNetV2 with pre-trained ImageNet weights
    base_model = InceptionResNetV2(input_shape=input_shape, include_top=False, weights='imagenet')

    # Freeze all layers except the last 5
    for layer in base_model.layers[:-5]:
        layer.trainable = False

    # Create the model
    model = tf.keras.Sequential([
        base_model,
        GlobalAveragePooling2D(),  # Better than Flatten for CNNs
        Dense(128, activation='relu', kernel_regularizer=l2(0.01)),
        Dense(2, activation='softmax')  # For two classes; for binary, you could use 'sigmoid'
    ])

    # Compile the model
    model.compile(loss='categorical_crossentropy',  # If binary, you might use 'binary_crossentropy'
                optimizer=Nadam(learning_rate=0.0001),
                metrics=['accuracy'])

    # Print the model summary
    model.summary()
    
    # Crea un oggetto StringIO per catturare l'output
    string_buffer = io.StringIO()

    # Usa redirect_stdout per catturare l'output di model.summary()
    with redirect_stdout(string_buffer):
        model.summary()

    # Estrai il contenuto sotto forma di stringa
    summary_string = string_buffer.getvalue()

    # Chiudi il buffer StringIO
    string_buffer.close()

    # save the model
    model_path = os.path.join('../models/', 'model.pkl')
    with open(model_path, 'wb') as file:  
        pickle.dump(model, file)

    # save the summary
    summary_path = os.path.join('../models/', 'model_summary.txt')
    # Scrivi il contenuto del riepilogo in un file di testo
    with open(summary_path, 'w') as file:
        file.write(summary_string)
        
    return model



# training
def training(model, X_train, X_val, y_train, y_val, epochs, batch_size, patience, image_size=224):

    # data augumentation
    data_train = train_generator(X_train, y_train, batch_size)

    early_stopping = EarlyStopping(
        monitor='val_loss',     
        patience=5,             
        verbose=1,                     
        restore_best_weights=True 
    )

    # train model
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val))

    # save history
    with open('../models/history.pkl', 'wb') as f:
        pickle.dump(history.history, f)

    return history


# generator for data augumentation
def train_generator(X_train, y_train, batch_size):

     # define data generator
    datagen = ImageDataGenerator(
        rotation_range=5,             
        width_shift_range=0.1,         
        height_shift_range=0.1,                      
    )

    datagen.fit(X_train)

    return datagen.flow(X_train, y_train, batch_size)



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
    plt.close()  
    plt.show()


if __name__ == "__main__":
    training()

