#!/usr/bin/env python3
"""
FaceTrack.py

1) Extract faces from a group image using MTCNN.
2) Upscale any tiny faces to 112×112.
3) For each person image in Person/, verify all extracted faces via DeepFace.
"""

import time
import os
from pathlib import Path

import cv2
import numpy as np
from mtcnn import MTCNN
from deepface import DeepFace


def extract_faces(group_path: Path,
                  out_dir:    Path,
                  min_size:   int = 112) -> list[Path]:
    """
    - Runs MTCNN on the group image
    - Saves each crop to out_dir/face_{i}.jpg
    - Upscales any face below min_size
    - Returns a list of the saved face file Paths
    """
    out_dir.mkdir(exist_ok=True)
    img_bgr = cv2.imread(str(group_path))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    detector = MTCNN()
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
        # save as BGR
        cv2.imwrite(str(out_path), cv2.cvtColor(face, cv2.COLOR_RGB2BGR))
        saved.append(out_path)

    print(f"Extracted {len(saved)} faces → {out_dir}")
    return saved


def verify_faces(face_paths: list[Path],
                 person_path: Path,
                 model_name: str = "Facenet",
                 threshold: float = 0.4) -> tuple[int, int]:
    """
    Runs DeepFace.verify on each face vs. person_path using the given threshold.
    Returns (true_count, false_count).
    """
    trues = falses = 0
    for p in face_paths:
        start = time.time()
        res = DeepFace.verify(
            img1_path=str(p),
            img2_path=str(person_path),
            model_name=model_name,
            detector_backend="mtcnn",
            enforce_detection=False,
            threshold=threshold            # ← here
        )
        dt = time.time() - start
        ok = res.get("verified", False)
        print(f"  {p.name}: verified={ok} ({dt:.2f}s)")

        if ok:
            trues += 1
        else:
            falses += 1

    return trues, falses


if __name__ == "__main__":
    base_dir        = Path(__file__).resolve().parent
    group_img       = base_dir / "Event"  / "gp-18.jpg"
    extract_dir     = base_dir / "extracted_faces"
    person_directory = base_dir / "Person"

    # sanity-check
    if not group_img.is_file():
        raise FileNotFoundError(f"Missing group image: {group_img}")
    if not person_directory.is_dir():
        raise FileNotFoundError(f"Missing Person directory: {person_directory}")

    # 1) extract & save all faces
    faces = extract_faces(group_img, extract_dir, min_size=112)

    # 2) for each person image, see if any extracted face matches
    found_people = []
    print("\n=== Checking each person image ===")
    for person_file in os.listdir(person_directory):
        person_path = person_directory / person_file
        if not person_path.is_file():
            continue

        print(f"\n-- {person_file} --")
        t, f = verify_faces(faces, person_path, model_name="Facenet")
        if t > 0:
            print(f"=> **{person_file} FOUND** ({t} of {len(faces)} faces matched)")
            found_people.append(person_file)
        else:
            print(f"=> {person_file} NOT found")

    # 3) summary
    print("\n=== Summary: People detected in group photo ===")
    if found_people:
        for name in found_people:
            print(f"- {name}")
    else:
        print("No known persons detected.")
