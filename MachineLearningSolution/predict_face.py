# predict_face.py
# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
from skimage import io
import tensorflow as tf
from tensorflow import keras
import IPython.display
import matplotlib.pyplot as plt

# =========================
# Face Detector Class
# =========================
class FaceDetectorYunet:
    def __init__(self,
                 model_path='face_detection_yunet_2023mar.onnx',
                 img_size=(300, 300),
                 threshold=0.5):
        self.model_path = model_path
        self.img_size = img_size
        self.fd = cv2.FaceDetectorYN_create(
            str(model_path),
            "",
            img_size,
            score_threshold=threshold
        )

    def scale_coords(self, image, prediction):
        ih, iw = image.shape[:2]
        rw, rh = self.img_size
        a = np.array([
            (prediction['x1'], prediction['y1']),
            (prediction['x1'] + prediction['x2'], prediction['y1'] + prediction['y2'])
        ])
        b = np.array([iw / rw, ih / rh])
        c = a * b
        prediction['x1'] = int(c[0, 0].round())
        prediction['x2'] = int(c[1, 0].round())
        prediction['y1'] = int(c[0, 1].round())
        prediction['y2'] = int(c[1, 1].round())
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

    def parse_predictions(self, image, faces):
        data = []
        for num, face in enumerate(list(faces)):
            x1, y1, x2, y2 = list(map(int, face[:4]))
            confidence = face[-1]
            datum = {
                'x1': x1,
                'y1': y1,
                'x2': x2,
                'y2': y2,
                'face_num': num,
                'confidence': confidence
            }
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


# =========================
# Helper to Display Image
# =========================
def show_image(image):
    _, ret = cv2.imencode('.jpg', cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    i = IPython.display.Image(data=ret)
    IPython.display.display(i)


# =========================
# Prediction Function
# =========================
def test_classifier(image_path, face_detector, model, class_names):
    img = io.imread(image_path)
    display_img = img.copy()

    faces = face_detector.detect(img)
    if not faces:
        print("No faces detected.")
        return display_img

    for face in faces:
        # Extract and preprocess face
        face_img = face_detector.extract_face(img, face)
        face_img = face_img.astype('float32') / 255.0

        # Ensure the image has 3 channels (RGB)
        if face_img.shape[-1] == 4:
            face_img = cv2.cvtColor(face_img, cv2.COLOR_RGBA2RGB)

        face_img = np.expand_dims(face_img, axis=0)

        # Predict
        prediction = model.predict(face_img, verbose=0)[0]
        predicted_class = class_names[np.argmax(prediction)]
        confidence = np.max(prediction) * 100

        # Draw
        cv2.rectangle(display_img,
                      (face['x1'], face['y1']),
                      (face['x2'], face['y2']),
                      (0, 255, 0), 2)
        label = f"{predicted_class}: {confidence:.1f}%"
        cv2.putText(display_img, label,
                    (face['x1'], face['y1'] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return display_img


# =========================
# Main Script
# =========================
if __name__ == "__main__":
    # Load model and classes
    model = keras.models.load_model('inception_model.keras')
    class_names = np.load('class_names.npy')

    # Init face detector
    fd = FaceDetectorYunet()

    # Image to test
    image_path = "avengersGroup/test-images-avengers-4.jpg"

    # Run prediction
    result_img = test_classifier(image_path, fd, model, class_names)

    # Show results
    show_image(result_img)

    # Optionally save the output
    cv2.imwrite("prediction_result.jpg", cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR))
    print("Prediction saved as prediction_result.jpg")
