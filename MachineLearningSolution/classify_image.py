#!/usr/bin/env python3
import cv2
import numpy as np
from tensorflow import keras

from inception_yunetdetector_timespace import FaceDetectorYunet

def list_people_in_image(image_path, fd, model, class_names, conf_thresh=0.0):
    """
    Detects all faces in `image_path`, classifies each,
    and returns a list of predicted class names (one per face)
    whose confidence ≥ conf_thresh.
    """
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    faces = fd.detect(img)
    if not faces:
        return []

    names = []
    for f in faces:
        crop = fd.extract_face(img, f, target_size=(256, 256))
        crop = crop.astype("float32") / 255.0
        inp  = np.expand_dims(crop, 0)

        preds = model.predict(inp)[0]
        idx   = np.argmax(preds)
        conf  = preds[idx]

        if conf >= conf_thresh:
            names.append(class_names[idx])

    return names

if __name__ == "__main__":
    fd = FaceDetectorYunet(
        model_path="face_detection_yunet_2023mar.onnx",
        img_size=(300, 300),
        threshold=0.5
    )
    model       = keras.models.load_model("inception_model.keras")
    class_names = np.load("class_names.npy")

    group_photo = "avengersGroup/group1.png"
    predictions = list_people_in_image(
        group_photo, fd, model, class_names,
        conf_thresh=0.0
    )

    if not predictions:
        print("No faces detected.")
    else:
        # Remove duplicates while preserving order
        unique = []
        for name in predictions:
            if name not in unique:
                unique.append(name)

        print("People detected in group photo:")
        for name in unique:
            print(f"  {name}")
