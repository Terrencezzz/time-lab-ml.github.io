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
            detector_backend="ssd",      # no extra TF deps
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
    group_img  = base_dir / "Event"  / "gp-3.jpg"
    person_img = base_dir / "Person" / "ar4152.jpg"
    extract_dir = base_dir / "extracted_faces"

    # sanity-check
    for path in (group_img, person_img):
        if not path.is_file():
            raise FileNotFoundError(f"Missing: {path}")

    faces = extract_faces(group_img, extract_dir, min_size=112)
    t, f = verify_faces(faces, person_img, model_name="Facenet")

    print("\n=== Summary ===")
    print(f"Matched:     {t}")
    print(f"Not matched: {f}")
