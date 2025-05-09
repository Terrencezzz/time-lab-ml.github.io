#!/usr/bin/env python3
"""
FaceTrack.py

1) Extract faces from a group image using MTCNN.
2) Upscale any tiny faces to 112×112.
3) Verify each face against one Person image via DeepFace (Facenet).
"""

import time
from pathlib import Path

import cv2
import numpy as np
from mtcnn import MTCNN
from deepface import DeepFace
import os
import matplotlib.pyplot as plt


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
            face = cv2.resize(face, (min_size, min_size), interpolation=cv2.INTER_CUBIC)

        out_path = out_dir / f"face_{i}.jpg"
        # save as BGR
        cv2.imwrite(str(out_path), cv2.cvtColor(face, cv2.COLOR_RGB2BGR))
        saved.append(out_path)

    print(f"Extracted {len(saved)} faces → {out_dir}")
    return saved


def verify_faces(face_paths: list[Path],
                 person_path: Path,
                 model_name:  str = "Facenet") -> tuple[int, int]:
    """
    Runs DeepFace.verify on each face vs. person_path.
    Returns (true_count, false_count).
    """
    trues = falses = 0
    for p in face_paths:
        start = time.time()
        res = DeepFace.verify(
            img1_path=str(p),
            img2_path=str(person_path),
            model_name=model_name,
            detector_backend="mtcnn",      # no extra TF deps
            enforce_detection=False
        )
        dt = time.time() - start
        ok = res.get("verified", False)
        print(f"{p.name}: verified={ok} ({dt:.2f}s)")

        if ok:   trues  += 1
        else:    falses += 1

    return trues, falses


if __name__ == "__main__":
    base_dir   = Path(__file__).resolve().parent
    group_img  = base_dir / "Event"  / "gp-17.jpg"
    extract_dir = base_dir / "extracted_faces"
    person_directory = base_dir / "Person"

    # sanity-check: make sure files exist
    if not group_img.is_file():
        raise FileNotFoundError(f"Missing: {group_img}")

    if not person_directory.is_dir():
        raise FileNotFoundError(f"Missing directory: {person_directory}")

    faces = extract_faces(group_img, extract_dir, min_size=112)

    report = []
    total_comparisons = 0
    total_matches = 0

    for face_name in os.listdir(extract_dir):
        face_path = extract_dir / face_name
        if not face_path.is_file():
            continue  # skip if not a file

        for person_name in os.listdir(person_directory):
            person_path = person_directory / person_name
            if not person_path.is_file():
                continue  # skip if not a file

            print(f"\n=== Comparing {face_name} vs {person_name} ===")
            t, f = verify_faces([face_path], person_path, model_name="Facenet")
            
            total_comparisons += 1
            if t > 0:
                total_matches += 1
                report.append({
                    "extracted_face": face_name,
                    "matched_person": person_name
                })
                
                print(f"Matched: {t}")
                
                break

            
            print(f"Not matched: {f}")

    # Final report
    print("\n=== Final Matching Report ===")
    if report:
        for entry in report:
            print(f"- {entry['extracted_face']} matched with {entry['matched_person']}")
    else:
        print("No matches found.")