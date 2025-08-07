# inception_yunetdetector_timespace.py
# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
from skimage import io

# ← Add these so the demo can call keras.models.load_model()
import tensorflow as tf
from tensorflow import keras


class FaceDetectorYunet:
    def __init__(self,
                 model_path='face_detection_yunet_2023mar.onnx',
                 img_size=(300, 300),
                 threshold=0.5):
        self.img_size = img_size
        self.fd = cv2.FaceDetectorYN_create(
            str(model_path), "", img_size, score_threshold=threshold
        )

    def detect(self, image):
        if isinstance(image, str):
            image = cv2.imread(image, cv2.IMREAD_UNCHANGED)

        # Drop alpha if present
        if image.ndim == 3 and image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

        small = cv2.resize(image, self.img_size)
        self.fd.setInputSize(self.img_size)
        _, faces = self.fd.detect(small)
        if faces is None:
            return []
        return self._parse(image, faces)

    def _parse(self, orig, faces):
        ih, iw = orig.shape[:2]
        rw, rh = self.img_size
        results = []
        for f in faces:
            x1, y1, x2, y2 = map(int, f[:4])
            sx, sy = iw/rw, ih/rh
            x1, x2 = int(round(x1*sx)), int(round(x2*sx))
            y1, y2 = int(round(y1*sy)), int(round(y2*sy))
            results.append({
                'x1': x1, 'y1': y1,
                'x2': x2, 'y2': y2,
                'confidence': float(f[-1])
            })
        return results

    def extract_face(self, image, face, target_size=(256,256)):
        if isinstance(image, str):
            image = cv2.imread(image, cv2.IMREAD_UNCHANGED)
        if image.ndim == 3 and image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

        h, w = image.shape[:2]
        m = 10
        x1 = max(0, face['x1'] - m)
        y1 = max(0, face['y1'] - m)
        x2 = min(w, face['x2'] + m)
        y2 = min(h, face['y2'] + m)

        # guard against zero‐sized crops
        if x2 <= x1:
            x2 = min(w, x1 + 1)
        if y2 <= y1:
            y2 = min(h, y1 + 1)

        crop = image[y1:y2, x1:x2]
        return cv2.resize(crop, target_size)


def test_classifier(image_path, face_detector, model, class_names):
    """Detect + classify faces; drops any alpha channel before prediction."""
    img = io.imread(image_path)
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]

    out = img.copy()
    faces = face_detector.detect(img)
    if not faces:
        print(f"No faces detected in {image_path!r}")
        return out

    for f in faces:
        crop = face_detector.extract_face(img, f)
        if crop.ndim == 3 and crop.shape[2] == 4:
            crop = crop[:, :, :3]
        inp = crop.astype('float32')/255.0
        inp = np.expand_dims(inp, 0)

        preds = model.predict(inp)[0]
        idx = np.argmax(preds)
        cls = class_names[idx]
        conf = preds[idx]*100

        cv2.rectangle(out, (f['x1'],f['y1']), (f['x2'],f['y2']), (0,255,0), 2)
        cv2.putText(out,
                    f"{cls}: {conf:.1f}%",
                    (f['x1'], f['y1']-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
    return out


def show_image(image):
    """Inline display for Jupyter; falls back to plt otherwise."""
    from IPython.display import Image, display
    import matplotlib.pyplot as plt

    if image.ndim == 3 and image.shape[2] == 3:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        _, buf = cv2.imencode('.jpg', rgb)
        display(Image(data=buf.tobytes()))
    else:
        plt.imshow(image)
        plt.axis('off')
        plt.show()


if __name__ == "__main__":
    # Quick standalone demo
    fd = FaceDetectorYunet()
    model = keras.models.load_model("inception_model.keras")
    names = np.load("class_names.npy")
    demo = test_classifier("avengersGroup/group1.png", fd, model, names)
    show_image(demo)
