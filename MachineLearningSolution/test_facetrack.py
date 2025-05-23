# test_facetrack.py
import cv2
import pytest
from pathlib import Path
import FaceTrack


def get_group_image():
    grp_dir = Path(__file__).parent / "avengersGroup"
    imgs = list(grp_dir.glob("*.png")) + list(grp_dir.glob("*.jpg"))
    if not imgs:
        pytest.skip("No group image found in avengersGroup/")
    return imgs[0]


def get_person_images():
    test_dir = Path(__file__).parent / "avengersTest"
    persons = sorted(test_dir.glob("*.png")) + sorted(test_dir.glob("*.jpg"))
    if len(persons) < 4:
        pytest.skip("Need at least 4 individual images in avengersTest/")
    return persons


def test_extract_avengers_faces(tmp_path):
    group_img = get_group_image()
    out_dir   = tmp_path / "faces"
    faces     = FaceTrack.extract_faces(group_img, out_dir, min_size=112)

    # now expect 4 people in the shot
    assert len(faces) == 4, f"expected 4 faces, got {len(faces)}"

    # each face file must exist and be at least 112×112
    for fpath in faces:
        assert fpath.exists(), f"Missing crop {fpath.name}"
        img = cv2.imread(str(fpath))
        h, w = img.shape[:2]
        assert h >= 112 and w >= 112, f"{fpath.name} too small: {h}×{w}"


def test_verify_extracted_vs_individual(tmp_path):
    # 1) extract into a new temp dir
    group_img   = get_group_image()
    extract_dir = tmp_path / "faces2"
    faces       = FaceTrack.extract_faces(group_img, extract_dir, min_size=112)
    
    print(">>> extracted faces are in:", extract_dir)

    # 2) load all five portraits
    persons = get_person_images()

    matches = {}
    for face_path in faces:
        for person_path in persons:
            t, f = FaceTrack.verify_faces([face_path], person_path, model_name="Facenet")
            if t == 1:
                matches[face_path.name] = person_path.name
                break
    
    print("Matches:")
    for face_name, person_name in matches.items():
        print(f"  {face_name} matched with {person_name}")
    
    # 3) we should have exactly 4 matches
    assert len(matches) == 4, f"expected 4 total matches, got {len(matches)}"
    

    # 4) and none of those matched names should be Scarlett's file
    unmatched = set(p.name for p in persons) - set(matches.values())
    assert "scarlett_johansson37.png" in unmatched, (
        "Scarlett should not have been matched, but she was"
    )
