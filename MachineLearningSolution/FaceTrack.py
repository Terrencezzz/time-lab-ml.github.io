#!/usr/bin/env python3
"""
FaceTrack.py

1) Extract faces from a group image using MTCNN.
2) Upscale any tiny faces to 112×112.
3) For each person image in Person/, verify all extracted faces via your custom Inception model.
"""

import time
import os
from pathlib import Path

import cv2
import numpy as np
from mtcnn import MTCNN
from tensorflow.keras.models import load_model

# --- Configuration ---
MIN_SIZE    = 112                    # smallest face side we'll accept
THRESHOLD   = 0.4                    # max cosine distance for a “match”
MODEL_PATH  = "inception_model.keras"  # path to your saved Keras model

# Load your embedding model once
embedder = load_model(MODEL_PATH)

# Build the face detector once
detector = MTCNN()


def extract_faces(group_path: Path,
                  out_dir:    Path,
                  min_size:   int = MIN_SIZE) -> list[Path]:
    """
    - Runs MTCNN on the group image
    - Saves each crop to out_dir/face_{i}.jpg
    - Upscales any face below min_size
    - Returns a list of the saved face file Paths
    """
    out_dir.mkdir(exist_ok=True)
    img_bgr = cv2.imread(str(group_path))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    detections = detector.detect_faces(img_rgb)
    saved = []

    for i, det in enumerate(detections):
        x, y, w, h = det["box"]
        x, y = max(0, x), max(0, y)
        face = img_rgb[y : y + h, x : x + w]

        # Upscale if too small
        if face.shape[0] < min_size or face.shape[1] < min_size:
            face = cv2.resize(face, (min_size, min_size),
                              interpolation=cv2.INTER_CUBIC)

        out_path = out_dir / f"face_{i}.jpg"
        # save as BGR for cv2.imwrite
        cv2.imwrite(str(out_path),
                    cv2.cvtColor(face, cv2.COLOR_RGB2BGR))
        saved.append(out_path)

    print(f"Extracted {len(saved)} faces → {out_dir}")
    return saved


def get_embedding(img_path: Path) -> np.ndarray:
    """
    - Loads an image, resizes to the model's input size,
      normalizes pixels, and returns the embedding vector.
    """
    img_bgr = cv2.imread(str(img_path))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # model expects square inputs
    _, H, W, _ = embedder.input_shape
    face = cv2.resize(img_rgb, (W, H), interpolation=cv2.INTER_CUBIC)

    face = face.astype("float32") / 255.0
    face = np.expand_dims(face, axis=0)
    emb  = embedder.predict(face)
    return emb[0]  # return 1D vector


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Compute 1 - cosine_similarity(a, b)."""
    return 1.0 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def verify_faces(face_paths: list[Path],
                 person_path: Path,
                 threshold: float = THRESHOLD) -> tuple[int, int]:
    """
    For each extracted face, compute its embedding vs. the person's embedding.
    Returns (match_count, non_match_count).
    """
    person_emb = get_embedding(person_path)
    matches = 0
    misses  = 0

    for p in face_paths:
        start     = time.time()
        face_emb  = get_embedding(p)
        dist      = cosine_distance(face_emb, person_emb)
        is_match  = (dist <= threshold)
        elapsed   = time.time() - start

        print(f"  {p.name}: dist={dist:.3f}, match={is_match} ({elapsed:.2f}s)")

        if is_match:
            matches += 1
        else:
            misses  += 1

    return matches, misses


if __name__ == "__main__":
    base_dir         = Path(__file__).resolve().parent
    group_img        = base_dir / "avengersGroup" / "group1.png"
    extract_dir      = base_dir / "extracted_faces"
    person_directory = base_dir / "avengersTest"

    # sanity-check inputs
    if not group_img.is_file():
        raise FileNotFoundError(f"Missing group image: {group_img}")
    if not person_directory.is_dir():
        raise FileNotFoundError(f"Missing Person directory: {person_directory}")

    # --- Clear extracted_faces directory first ---
    if extract_dir.exists():
        for f in extract_dir.iterdir():
            if f.is_file():
                f.unlink()
    else:
        extract_dir.mkdir(exist_ok=True)

    # 1) Extract faces from the group image
    faces = extract_faces(group_img, extract_dir, MIN_SIZE)

    # 2) For each person image, verify against all extracted faces
    found_people = []
    print("\n=== Checking each person image ===")
    for person_file in os.listdir(person_directory):
        person_path = person_directory / person_file
        if not person_path.is_file():
            continue

        print(f"\n-- {person_file} --")
        t, f = verify_faces(faces, person_path, threshold=THRESHOLD)
        if t > 0:
            print(f"=> **{person_file} FOUND** ({t} of {len(faces)} faces matched)")
            found_people.append(person_file)
        else:
            print(f"=> {person_file} NOT found")

    # 3) Summary of detected people
    print("\n=== Summary: People detected in group photo ===")
    if found_people:
        for name in found_people:
            print(f"- {name}")
    else:
        print("No known persons detected.")