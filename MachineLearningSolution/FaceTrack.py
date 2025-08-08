#!/usr/bin/env python3
"""
FaceTrack.py

1) Extract faces from a group image using MTCNN.
2) Upscale any tiny faces to 112×112.
3) Verify each face against one Person image via DeepFace (Facenet).
"""
import json
import time
import os
import json
from pathlib import Path

import cv2
import numpy as np
from mtcnn import MTCNN
from tensorflow.keras.models import load_model

# --- Configuration ---
MIN_SIZE   = 112      # smallest face side we'll accept
THRESHOLD  = 0.4      # max cosine distance for a “match”
MODEL_PATH = "inception_model.keras"

# load your model once
embedder = load_model(MODEL_PATH)

# build MTCNN detector once
detector = MTCNN()


def extract_faces(group_path: Path,
                  out_dir:    Path,
                  min_size:   int = MIN_SIZE) -> list[Path]:
    """
    - Runs MTCNN on the group image
    - Upscales any tiny faces to `min_size`×`min_size`
    - Saves each crop to out_dir/face_{i}.jpg
    - Returns list of saved face Paths
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

        # upscale if needed
        if face.shape[0] < min_size or face.shape[1] < min_size:
            face = cv2.resize(face, (min_size, min_size),
                              interpolation=cv2.INTER_CUBIC)

        out_path = out_dir / f"face_{i}.jpg"
        cv2.imwrite(str(out_path),
                    cv2.cvtColor(face, cv2.COLOR_RGB2BGR))
        saved.append(out_path)

    print(f"Extracted {len(saved)} faces → {out_dir}")
    return saved


def get_embedding(img_path: Path) -> np.ndarray:
    """
    - Loads image, normalizes it, runs `embedder.predict`, and returns 1D embedding.
    """
    img_bgr = cv2.imread(str(img_path))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    # your model expects square input of size MODEL_INPUT × MODEL_INPUT
    MODEL_INPUT = embedder.input_shape[1]
    face = cv2.resize(img_rgb, (MODEL_INPUT, MODEL_INPUT),
                      interpolation=cv2.INTER_CUBIC)
    face = face.astype("float32") / 255.0
    face = np.expand_dims(face, axis=0)
    emb = embedder.predict(face)
    return emb[0]


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """1 − cosine_similarity"""
    return 1.0 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def verify_faces(face_paths: list[Path],
                 person_path: Path,
                 threshold: float = THRESHOLD) -> tuple[int, int]:
    """
    For each face in `face_paths`, compute its embedding vs. `person_path` embedding.
    Returns (count_matches, count_non_matches).
    """
    person_emb = get_embedding(person_path)
    matches = 0
    misses  = 0

    for p in face_paths:
        start = time.time()
        face_emb = get_embedding(p)
        dist     = cosine_distance(face_emb, person_emb)
        ok       = (dist <= threshold)
        print(f"{p.name}: dist={dist:.3f}, verified={ok} "
              f"({(time.time()-start):.2f}s)")
        if ok:    matches += 1
        else:     misses  += 1

    return matches, misses


if __name__ == "__main__":
    base_dir         = Path(__file__).resolve().parent
    group_img        = base_dir / "Event" / "gp-17.jpg"
    person_directory = base_dir / "Person"
    extract_dir      = base_dir / "extracted_faces"

    # sanity-check
    if not group_img.is_file():
        raise FileNotFoundError(f"Missing group image: {group_img}")
    if not person_directory.is_dir():
        raise FileNotFoundError(f"Missing Person dir: {person_directory}")

    # 1) extract & save faces
    faces = extract_faces(group_img, extract_dir, MIN_SIZE)

    # 2) loop through extracted faces vs each person image
    report = []
    for face_file in os.listdir(extract_dir):
        face_path = extract_dir / face_file
        if not face_path.is_file():
            continue

        for person_file in os.listdir(person_directory):
            person_path = person_directory / person_file
            if not person_path.is_file():
                continue

            print(f"\n=== Comparing {face_file} vs {person_file} ===")
            t, f = verify_faces([face_path], person_path, THRESHOLD)
            if t > 0:
                report.append({
                    "extracted_face": face_file,
                    "matched_person": person_file
                })
                print(f"Matched: {t}")
                break
            else:
                print(f"Not matched: {f}")

    # 3) final JSON report
    print("\n=== Final Matching Report ===")
    if report:
        print(json.dumps(report, indent=2))
    else:
        print("No matches found.")
    
    # base_dir      = Path(__file__).resolve().parent
    # group_dir     = base_dir / "avengersGroup"
    # test_dir      = base_dir / "avengersTest"
    # extract_root  = base_dir / "extracted_faces"
    #
    # results = {}
    #
    # # make sure extract_root is empty before we start
    # if extract_root.exists():
    #     for f in extract_root.iterdir():
    #         f.unlink()
    # else:
    #     extract_root.mkdir()
    #
    # for grp in group_dir.iterdir():
    #     if not grp.is_file(): continue
    #
    #     # 1) extract faces for this group image
    #     faces = extract_faces(grp, extract_root, min_size=112)
    #
    #     # 2) verify each face against every test image
    #     matches = []
    #     for face_path in faces:
    #         for person_img in test_dir.iterdir():
    #             if not person_img.is_file(): continue
    #
    #             t, _ = verify_faces([face_path], person_img, model_name="Facenet")
    #             if t > 0:
    #                 matches.append({
    #                     "extracted_face": face_path.name,
    #                     "matched_person": person_img.name
    #                 })
    #                 break
    #
    #     results[grp.name] = matches
    #
    # # 3) print out the whole thing as pretty JSON
    # print(json.dumps(results, indent=2))