# -*- coding: utf-8 -*-
from pathlib import Path

import cv2
import numpy as np
import IPython
from skimage import io
import os
import glob
import math
import random
import pandas as pd
import matplotlib.pyplot as plt

class FaceDetectorYunet():
    def __init__(self,
                  model_path='face_detection_yunet_2023mar.onnx',
                  img_size=(300, 300),
                  threshold=0.5):
        self.model_path = model_path
        self.img_size = img_size
        self.fd = cv2.FaceDetectorYN_create(str(model_path),
                                            "",
                                            img_size,
                                            score_threshold=threshold)

    def draw_faces(self,
                   image,
                   faces,
                   draw_landmarks=False,
                   show_confidence=False):
        for face in faces:
            color = (0, 0, 255)
            thickness = 2
            cv2.rectangle(image, (face['x1'], face['y1']), (face['x2'], face['y2']), color, thickness, cv2.LINE_AA)

            if draw_landmarks:
                landmarks = face['landmarks']
                for landmark in landmarks:
                    radius = 5
                    thickness = -1
                    cv2.circle(image, landmark, radius, color, thickness, cv2.LINE_AA)

            if show_confidence:
                confidence = face['confidence']
                confidence = "{:.2f}".format(confidence)
                position = (face['x1'], face['y1'] - 10)
                font = cv2.FONT_HERSHEY_SIMPLEX
                scale = 0.5
                thickness = 2
                cv2.putText(image, confidence, position, font, scale, color, thickness, cv2.LINE_AA)
        return image

    def scale_coords(self, image, prediction):
        ih, iw = image.shape[:2]
        rw, rh = self.img_size
        a = np.array([
                (prediction['x1'], prediction['y1']),
                (prediction['x1'] + prediction['x2'], prediction['y1'] + prediction['y2'])
                    ])
        b = np.array([iw/rw, ih/rh])
        c = a * b
        prediction['img_width'] = iw
        prediction['img_height'] = ih
        prediction['x1'] = int(c[0,0].round())
        prediction['x2'] = int(c[1,0].round())
        prediction['y1'] = int(c[0,1].round())
        prediction['y2'] = int(c[1,1].round())
        prediction['face_width'] = (c[1,0] - c[0,0])
        prediction['face_height'] = (c[1,1] - c[0,1])
        prediction['area'] = prediction['face_width'] * prediction['face_height']
        prediction['pct_of_frame'] = prediction['area']/(prediction['img_width'] * prediction['img_height'])
        return prediction

    def detect(self, image):
        if isinstance(image, str):
            image = cv2.imread(str(image))
        img = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        img = cv2.resize(img, self.img_size)
        self.fd.setInputSize(self.img_size)
        _, faces = self.fd.detect(img)
        if faces is None:
            return None
        else:
            predictions = self.parse_predictions(image, faces)
            return predictions

    def parse_predictions(self,
                          image,
                          faces):
        data = []
        for num, face in enumerate(list(faces)):
            x1, y1, x2, y2 = list(map(int, face[:4]))
            landmarks = list(map(int, face[4:len(face)-1]))
            landmarks = np.array_split(landmarks, len(landmarks) / 2)
            positions = ['left_eye', 'right_eye', 'nose', 'right_mouth', 'left_mouth']
            landmarks = {positions[num]: x.tolist() for num, x in enumerate(landmarks)}
            confidence = face[-1]
            datum = {'x1': x1,
                     'y1': y1,
                     'x2': x2,
                     'y2': y2,
                     'face_num': num,
                     'landmarks': landmarks,
                     'confidence': confidence,
                     'model': 'yunet'}
            d = self.scale_coords(image, datum)
            data.append(d)
        return data

    def extract_face(self, image, face, target_size=(256, 256)):
      if isinstance(image, str):
          image = cv2.imread(str(image))

      x1, y1, x2, y2 = face['x1'], face['y1'], face['x2'], face['y2']

      h, w = image.shape[:2]
      margin = 10
      x1 = max(0, x1 - margin)
      y1 = max(0, y1 - margin)
      x2 = min(w, x2 + margin)
      y2 = min(h, y2 + margin)

      face_region = image[y1:y2, x1:x2]
      face_region = cv2.resize(face_region, target_size)

      return face_region

def show_image(image):
    _, ret = cv2.imencode('.jpg', image)
    i = IPython.display.Image(data=ret)
    IPython.display.display(i)

def show_by_name(name):
    fp = Path('../test_images').joinpath(name)
    img = cv2.imread(str(fp))
    show_image(img)

file_path = "avengersPhotos/images/train"

all_train_data = []
for path in os.listdir(file_path):
    for image in os.listdir(os.path.join(file_path, path)):
        all_train_data.append(os.path.join(file_path, path, image))

import IPython
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def show_image(image):
    if len(image.shape) == 3 and image.shape[2] == 3:
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)
        _, ret = cv2.imencode('.jpg', cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        i = IPython.display.Image(data=ret)
        IPython.display.display(i)
    else:
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(image)
        plt.axis('off')
        plt.show()

def prepare_classification_data(base_dir, face_detector, target_size=(256, 256)):
    X = []
    y = []

    train_dir = os.path.join(base_dir, 'images/train')

    for character in os.listdir(train_dir):
        character_dir = os.path.join(train_dir, character)
        if not os.path.isdir(character_dir):
            continue

        print(f"Processing {character} images...")

        for img_file in os.listdir(character_dir):
            img_path = os.path.join(character_dir, img_file)

            try:
                img = io.imread(img_path)
                faces = face_detector.detect(img)

                if faces:
                    best_face = max(faces, key=lambda x: x['confidence'])
                    face_img = face_detector.extract_face(img, best_face, target_size)
                    X.append(face_img)
                    y.append(character)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")

    return np.array(X), np.array(y)

def visualize_samples(X, y, num_samples=5):
    plt.figure(figsize=(15, 5))
    for i in range(min(num_samples, len(X))):
        plt.subplot(1, num_samples, i+1)
        plt.imshow(X[i])
        plt.title(y[i])
        plt.axis('off')
    plt.tight_layout()
    plt.show()

def create_classification_model(input_shape, num_classes):
    data_augmentation = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.3),
        layers.RandomZoom(0.3),
        layers.RandomContrast(0.3)
    ])

    base_model = keras.applications.InceptionV3(include_top=False, input_shape=(256, 256, 3))

    for layer in base_model.layers:
        layer.trainable = False

    model = keras.Sequential()
    model.add(data_augmentation)
    model.add(base_model)
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(5, activation='softmax'))

    model.compile(
        loss=keras.losses.CategoricalCrossentropy(),
        optimizer=keras.optimizers.Adam(),
        metrics=['accuracy']
    )

    return model

train_data= keras.utils.image_dataset_from_directory(
    directory= 'avengersPhotos/images/train',
    labels="inferred",
    label_mode="int",
    color_mode="rgb",
    batch_size=32,
    image_size=(256, 256)
)

test_data= keras.utils.image_dataset_from_directory(
    directory= 'avengersPhotos/images/test',
    labels="inferred",
    label_mode="int",
    color_mode="rgb",
    batch_size=32,
    image_size=(256, 256)
)

val_data= keras.utils.image_dataset_from_directory(
    directory= 'avengersPhotos/images/val',
    labels="inferred",
    label_mode="int",
    color_mode="rgb",
    batch_size=32,
    image_size=(256, 256)
)

fd = FaceDetectorYunet()

print("Preparing classification data...")
X, y = prepare_classification_data('avengersPhotos/', fd)

if len(X) == 0:
    print("No face data was extracted. Check your dataset structure and face detection threshold.")
else:
    print(f"Extracted {len(X)} faces from {len(set(y))} different characters.")
    visualize_samples(X, y)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    y_categorical = to_categorical(y_encoded)
    X_train, X_val, y_train, y_val = train_test_split(X, y_categorical, test_size=0.2, random_state=42)
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    X_train = X_train.astype('float32') / 255.0
    X_val = X_val.astype('float32') / 255.0
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2
    )
    input_shape = X_train.shape[1:]
    num_classes = len(label_encoder.classes_)
    print(f"Input shape: {input_shape}")
    print(f"Number of classes: {num_classes}")
    model = create_classification_model(input_shape, num_classes)
    print(model.summary())
    print("Training model...")
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=32),
        epochs=50,
        validation_data=(X_val, y_val),
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                min_delta= 0.001,
                patience= 25,
                verbose= True,
                restore_best_weights= True
            ),
        ]
    )
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Validation'])
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Validation'])
    plt.tight_layout()
    plt.show()
    np.save('class_names.npy', label_encoder.classes_)
    val_loss, val_acc = model.evaluate(X_val, y_val)
    print(f"Validation accuracy: {val_acc*100:.2f}%")

model.save('inception_model.keras')

def test_classifier(image_path, face_detector, model, class_names):
    img = io.imread(image_path)
    display_img = img.copy()
    faces = face_detector.detect(img)
    if faces:
        for face in faces:
            face_img = face_detector.extract_face(img, face)
            face_img = face_img.astype('float32') / 255.0
            face_img = np.expand_dims(face_img, axis=0)
            prediction = model.predict(face_img)[0]
            predicted_class = class_names[np.argmax(prediction)]
            confidence = np.max(prediction) * 100
            cv2.rectangle(display_img,
                          (face['x1'], face['y1']),
                          (face['x2'], face['y2']),
                          (0, 255, 0), 2)
            label = f"{predicted_class}: {confidence:.1f}%"
            cv2.putText(display_img,
                        label,
                        (face['x1'], face['y1'] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    else:
        print("No faces detected in the image.")
    return display_img

loaded_model = keras.models.load_model('inception_model.keras')

result = test_classifier('avengersGroup/group1.png', fd, loaded_model, label_encoder.classes_)
print(f"Results for {os.path.basename('avengersGroup/group1.png')}:")
show_image(result)

val_loss, val_acc = model.evaluate(X_val, y_val)
print(f"Validation accuracy: {val_acc*100:.3f}%")
print(f"Validation loss: {val_loss*100:.3f}%")

from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

y_pred = model.predict(X_val)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_val, axis=1)

cm = confusion_matrix(y_true_classes, y_pred_classes)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

print(classification_report(y_true_classes, y_pred_classes, target_names=label_encoder.classes_))

num_samples = 5
random_indices = np.random.choice(X_val.shape[0], num_samples, replace=False)

plt.figure(figsize=(30, 20))
for i, idx in enumerate(random_indices):
    plt.subplot(1, num_samples, i+1)
    plt.imshow(X_val[idx])
    predicted_label = label_encoder.classes_[y_pred_classes[idx]]
    true_label = label_encoder.classes_[y_true_classes[idx]]
    title = f"Predicted: {predicted_label}, True: {true_label}"
    plt.title(title)
    plt.axis('off')
plt.subplots_adjust(wspace=0.4)
plt.show()

num_samples = 5
random_indices = np.random.choice(X_val.shape[0], num_samples, replace=False)

plt.figure(figsize=(30, 20))
for i, idx in enumerate(random_indices):
    plt.subplot(1, num_samples, i+1)
    plt.imshow(X_val[idx])
    predicted_label = label_encoder.classes_[y_pred_classes[idx]]
    true_label = label_encoder.classes_[y_true_classes[idx]]
    title = f"Predicted: {predicted_label}, True: {true_label}"
    plt.title(title)
    plt.axis('off')
plt.subplots_adjust(wspace=0.4)
plt.show()

num_samples = 5
random_indices = np.random.choice(X_val.shape[0], num_samples, replace=False)

plt.figure(figsize=(30, 20))
for i, idx in enumerate(random_indices):
    plt.subplot(1, num_samples, i+1)
    plt.imshow(X_val[idx])
    predicted_label = label_encoder.classes_[y_pred_classes[idx]]
    true_label = label_encoder.classes_[y_true_classes[idx]]
    title = f"Predicted: {predicted_label}, True: {true_label}"
    plt.title(title)
    plt.axis('off')
plt.subplots_adjust(wspace=0.4)
plt.show()
