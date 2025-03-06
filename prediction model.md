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

model = load_model('model\emotion_detection_model.h5')

def predict_emotion(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (48, 48))  # Resize to 48x48
    img = np.expand_dims(img, axis=-1) / 255.0  # Normalize and add channel dimension
    img = np.expand_dims(img, axis=0)  # Batch size of 1
    
    prediction = model.predict(img)
    emotion_index = np.argmax(prediction)
    emotion = emotion_labels[emotion_index]
    
    return emotion

# Example usage:
image_path = 'images/train/4/fer0000049.png'
print("The actual emotion is sad")
predicted_emotion = predict_emotion(image_path)
print(f"The predicted emotion is: {predicted_emotion}")

trainedImg = mpimg.imread(image_path)
imgplot = plt.imshow(trainedImg, cmap= 'gray')

image_path = 'images/train/0/fer0000119.png'
print("The actual emotion is angry")
predicted_emotion = predict_emotion(image_path)
print(f"The predicted emotion is: {predicted_emotion}")

image_path = 'images/train/3/fer0000025.png'
print("The actual emotion is happy")
predicted_emotion = predict_emotion(image_path)
print(f"The predicted emotion is: {predicted_emotion}")

image_path = 'images/train/2/fer0000299.png'
print("The actual emotion is fear")
predicted_emotion = predict_emotion(image_path)
print(f"The predicted emotion is: {predicted_emotion}")

image_path = 'images/train/6/fer0000011.png'
print("The actual emotion is neutral")
predicted_emotion = predict_emotion(image_path)
print(f"The predicted emotion is: {predicted_emotion}")

image_path = 'images/train/5/fer0000174.png'
print("The actual emotion is surprise")
predicted_emotion = predict_emotion(image_path)
print(f"The predicted emotion is: {predicted_emotion}")

image_path = 'images/train/1/fer0027695.png'
print("The actual emotion is disgust")
predicted_emotion = predict_emotion(image_path)
print(f"The predicted emotion is: {predicted_emotion}")
