#!/usr/bin/env python3
"""
FaceTrack.py

Loads two images (gp-1.jpg and gp-2.jpg) from the Event/ folder
and passes them into a compare_images() function.
"""

from pathlib import Path
from PIL import Image
from pathlib import Path

import cv2
import numpy as np
import IPython
from skimage import io
import os
import glob
import math
import cv2
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from deepface import DeepFace
from retinaface import RetinaFace


def compare_images(img1: Image.Image, img2: Image.Image) -> None:
    """
    Example function that takes two PIL Image objects.
    You can swap this out for your actual face-tracking logic.
    """
    # for demo, just display them
    img1.show(title="Image 1")
    img2.show(title="Image 2")

    group_photos = img1
    start_time = time.time()  # Start timer

    faces = RetinaFace.extract_faces(img_path=group_photos, align=True, expand_face_area=5)

    end_time = time.time()  # End timer
    elapsed = end_time - start_time

    for idx, face in enumerate(faces):

        # Convert to display format (ensure it's uint8 and in RGB)
        if face.dtype != np.uint8:
            face = (face * 255).astype(np.uint8)

    counter = 0

    for face in faces:
        plt.imshow(face)
        cv2.imwrite(f"MachineLearningSolution/extracted_faces/face_{counter}.jpg", face)
        counter += 1

    file_path = "MachineLearningSolution/extracted_faces"
    extracted_faces_data = []
    for path in os.listdir(file_path):
        extracted_faces_data.append(os.path.join(file_path, path))

    true_counter = 0
    false_counter = 0
    testing_path = img2
    MIN_SIZE = 112  # ArcFace expected input size
    counter_face = 0

    for train_path in extracted_faces_data:
        if os.path.isfile(train_path):
            try:
                print(f"\nVerifying: {train_path} vs {testing_path}")
                img = cv2.imread(train_path)
                if img is None:
                    print(f"Could not load {train_path}. Skipping.")
                    continue
                h, w = img.shape[:2]
                print(f"Original size: {w}x{h}")

                # Upscale if too small
                if h < MIN_SIZE or w < MIN_SIZE:
                    print(f"Upscaling {train_path} from ({w}x{h}) to ({MIN_SIZE}x{MIN_SIZE})...")
                    img = cv2.resize(img, (MIN_SIZE, MIN_SIZE), interpolation=cv2.INTER_CUBIC)
                    tmp_path = f"/MachineLearningSolution/upscaled_faces/upscaled_face_{counter}.jpg"
                    cv2.imwrite(tmp_path, img)
                    img_to_verify = tmp_path
                    counter += 1
                else:
                    img_to_verify = train_path

                start_time = time.time()  # Start timer

                obj = DeepFace.verify(
                    img1_path=img_to_verify,
                    img2_path=testing_path,
                    model_name='ArcFace',
                    detector_backend='dlib',
                    enforce_detection=False
                )

                end_time = time.time()
                elapsed = end_time - start_time

                if obj['verified']:
                    true_counter += 1
                else:
                    false_counter += 1

                print(f"Result: {obj['verified']}")
                print(f"Time taken: {elapsed:.2f} seconds")

                plt.imshow(img)
                plt.show()

            except Exception as e:
                print(f"Error processing {train_path}: {e}")
        else:
            print(f"Skipping {train_path} as it is not a file.")

if __name__ == "__main__":
    # 1) Locate the directory this script lives in:
    base_dir = Path(__file__).resolve().parent
    #    e.g. .../MachineLearningSolution

    # 2) Point at the Event/ subfolder:
    event_dir = base_dir / "Event"
    person_dir = base_dir / "Person"

    # 3) Build the paths to your two images:
    img1_path = event_dir / "gp-3.jpg"
    img2_path = person_dir / "ar4152.jpg"

    # 4) (Optional) sanity-check the paths before loading:
    print(f"Loading image 1 from: {img1_path}")
    print(f"Loading image 2 from: {img2_path}")
    if not img1_path.is_file():
        raise FileNotFoundError(f"Cannot find {img1_path}")
    if not img2_path.is_file():
        raise FileNotFoundError(f"Cannot find {img2_path}")

    # 5) Load the images with Pillow:
    img1 = Image.open(img1_path)
    img2 = Image.open(img2_path)

    # 6) Call your processing function:
    compare_images(img1, img2)
