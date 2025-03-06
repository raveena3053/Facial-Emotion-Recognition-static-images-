import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import matplotlib.image as mpimg
import cv2
import matplotlib.pyplot as plt
import gc
import keras.backend as K

K.clear_session()
gc.collect()

# Directory paths (assuming FER-2013 images are in a directory)
train_dir = "images/train"
test_dir = "images/test"

# Emotion classes
emotion_labels = {
    0: 'Angry',
    1: 'Disgust',
    2: 'Fear',
    3: 'Happy',
    4: 'Sad',
    5: 'Surprise',
    6: 'Neutral'
}

# Image processing function
def preprocess_images(image_dir):
    images = []
    labels = []
    
    for emotion in os.listdir(image_dir):
        emotion_path = os.path.join(image_dir, emotion)
        if os.path.isdir(emotion_path):
            for img_name in os.listdir(emotion_path):
                img_path = os.path.join(emotion_path, img_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, (48, 48))  # Resize to 48x48
                images.append(img)
                labels.append(int(emotion))
    
    images = np.array(images) / 255.0  # Normalize
    images = np.expand_dims(images, axis=-1)  # Add channel dimension (grayscale)
    
    labels = np.array(labels)
    labels = to_categorical(labels, num_classes=7)  # One-hot encoding
    return images, labels

# Load train and test images
train_images, train_labels = preprocess_images(train_dir)
test_images, test_labels = preprocess_images(test_dir)

# Split into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(train_images, train_labels, test_size=0.2, random_state=42)

# Build the CNN model
model = Sequential([
    Conv2D(64, (3, 3), activation="relu", input_shape=(48, 48, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    Conv2D(128, (3, 3), activation="relu"),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    Conv2D(256, (3, 3), activation="relu"),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(7, activation='softmax')  # 7 classes of emotions
])

# Compile the model
from keras.optimizers import Adam
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# Train the model
early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)

# history = model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_val, y_val), callbacks=[early_stopping])

history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"Test accuracy: {test_acc * 100:.2f}%")

# Save the model
model.save('model\emotion_detection_model.h5')

# model = load_model('model\emotion_detection_model.h5')

# Function to predict emotion from a new image
def predict_emotion(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (48, 48))  # Resize to 48x48
    img = np.expand_dims(img, axis=-1) / 255.0  # Normalize and add channel dimension
    img = np.expand_dims(img, axis=0)  # Batch size of 1
    
    prediction = model.predict(img)
    emotion_index = np.argmax(prediction)
    emotion = emotion_labels[emotion_index]
    
    return emotion

print(f"Number of epochs: {len(history.history['accuracy'])}")  # Should be equal to the number of epochs

# Plot training and validation accuracy
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)  # Subplot for accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

# Plot training and validation loss
plt.subplot(1, 2, 2)  # Subplot for loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')

plt.tight_layout()  # Adjusts subplots to prevent overlap
plt.show()
