#!/usr/bin/env python3
"""
FaceTrack.py

1) Extract faces from a group image using MTCNN.
2) Upscale any tiny faces to 112×112.
3) For each person image in Person/, verify all extracted faces via your custom model.
"""
import json
import time
import os
from pathlib import Path
from typing import List

import cv2
import numpy as np
from mtcnn import MTCNN
from tensorflow.keras.models import load_model

# --- Configuration ---
MIN_SIZE    = 112
THRESHOLD   = 0.4
DEFAULT_MODEL_PATH = "MachineLearningSolution/inception_model.keras"

# Cache models when first requested
_model_cache = {}

# Build the face detector once
detector = MTCNN()


def get_model(model_name: str = "Inception"):
    """
    Lazy-load and return the requested embedding model.
    Currently supports:
      - "Inception" (default)
      - "Facenet"  (placeholder path)
    """
    if model_name in _model_cache:
        return _model_cache[model_name]

    if model_name.lower() == "facenet":
        model_path = "MachineLearningSolution/inception_model.keras"
    else:
        model_path = DEFAULT_MODEL_PATH

    if not Path(model_path).is_file():
        raise FileNotFoundError(f"Embedding model not found at {model_path}")

    model = load_model(model_path)
    _model_cache[model_name] = model
    return model


def extract_faces(group_path: Path,
                  out_dir:    Path,
                  min_size:   int = MIN_SIZE) -> list[Path]:
    out_dir.mkdir(exist_ok=True)
    img_bgr = cv2.imread(str(group_path))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    detections = detector.detect_faces(img_rgb)
    saved = []

    for i, det in enumerate(detections):
        x, y, w, h = det["box"]
        x, y = max(0, x), max(0, y)
        face = img_rgb[y : y + h, x : x + w]

        if face.shape[0] < min_size or face.shape[1] < min_size:
            face = cv2.resize(face, (min_size, min_size),
                              interpolation=cv2.INTER_CUBIC)

        out_path = out_dir / f"face_{i}.jpg"
        cv2.imwrite(str(out_path),
                    cv2.cvtColor(face, cv2.COLOR_RGB2BGR))
        saved.append(out_path)

    print(f"Extracted {len(saved)} faces → {out_dir}")
    return saved


def get_embedding(img_path: Path, model_name: str = "Inception") -> np.ndarray:
    model = get_model(model_name)

    img_bgr = cv2.imread(str(img_path))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # model expects square inputs
    _, H, W, _ = model.input_shape
    face = cv2.resize(img_rgb, (W, H), interpolation=cv2.INTER_CUBIC)

    face = face.astype("float32") / 255.0
    face = np.expand_dims(face, axis=0)
    emb  = model.predict(face, verbose=0)
    return emb[0]


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    return 1.0 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def verify_faces(face_paths: list[Path],
                 person_path: Path,
                 threshold: float = THRESHOLD,
                 model_name: str = "Inception") -> tuple[int, int]:
    """
    For each extracted face, compute its embedding vs. the person's embedding.
    Returns (match_count, non_match_count).
    """
    person_emb = get_embedding(person_path, model_name)
    matches = 0
    misses  = 0

    for p in face_paths:
        start     = time.time()
        face_emb  = get_embedding(p, model_name)
        dist      = cosine_distance(face_emb, person_emb)
        is_match  = (dist <= threshold)
        elapsed   = time.time() - start

        print(f"  {p.name}: dist={dist:.3f}, match={is_match} ({elapsed:.2f}s)")

        if is_match:
            matches += 1
        else:
            misses  += 1

    return matches, misses


def find_people_in_group_simple(
    group_img: Path,
    person_directory: Path,
    *,
    verbose: bool = True,
    model_name: str = "Inception"
) -> str:
    extract_dir = Path(__file__).resolve().parent / "extracted_faces"

    if not group_img.is_file():
        return json.dumps({
            "status": "error",
            "message": f"Missing group image: {group_img}"
        })
    if not person_directory.is_dir():
        return json.dumps({
            "status": "error",
            "message": f"Missing person directory: {person_directory}"
        })

    if extract_dir.exists():
        for f in extract_dir.iterdir():
            if f.is_file():
                f.unlink()
    else:
        extract_dir.mkdir(parents=True, exist_ok=True)

    faces = extract_faces(group_img, extract_dir, MIN_SIZE)

    if verbose:
        print("\n=== Checking each person image ===")

    found_people: List[str] = []
    for person_file in sorted(os.listdir(person_directory)):
        person_path = person_directory / person_file
        if not person_path.is_file():
            continue

        name_out = person_path.stem

        if verbose:
            print(f"\n-- {person_file} --")
        t, f = verify_faces(faces, person_path,
                            threshold=THRESHOLD,
                            model_name=model_name)

        if t > 0:
            if verbose:
                print(f"=> **{person_file} FOUND** ({t} of {len(faces)} faces matched)")
            found_people.append(name_out)
        else:
            if verbose:
                print(f"=> {person_file} NOT found")

    response = {
        "status": True,
        "found": found_people,
        "total_faces": len(faces)
    }
    return json.dumps(response, indent=2)


if __name__ == "__main__":
    base_dir       = Path(__file__).resolve().parent
    group_img      = base_dir / "avengersGroup" / "group1.png"
    person_dir     = base_dir / "avengersTest"

    result_json = find_people_in_group_simple(group_img, person_dir)
    result = json.loads(result_json)

    print("\n=== Summary: People detected in group photo ===")
    if result["status"] and result["found"]:
        for name in result["found"]:
            print(f"- {name}")
    else:
        print("No known persons detected.")
