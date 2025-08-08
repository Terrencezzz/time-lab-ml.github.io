# test_predict_face.py
# Unit tests for predict_face.py

import os
import numpy as np
import cv2
import builtins
import importlib
import pytest
import predict_face


class FakeDetectorNoFaces:
    def detect(self, image):
        # Simulate no faces found
        return None

    def extract_face(self, image, face, target_size=(256, 256)):
        raise RuntimeError("should not be called when no faces are detected")


class FakeDetectorOneFace:
    def __init__(self, face_bbox, extracted_face):
        # face_bbox is a dict with x1,y1,x2,y2
        self._face = face_bbox
        self._extracted = extracted_face

    def detect(self, image):
        # Return a single-face list matching what predict_face.parse_predictions produces
        return [self._face]

    def extract_face(self, image, face, target_size=(256, 256)):
        # Return a pre-made face crop (already sized to target_size)
        return self._extracted


class FakeModel:
    def __init__(self, probs):
        # probs should be a 1D array-like representing softmax output
        self._probs = np.array(probs).reshape((1, -1))

    def predict(self, x, verbose=0):
        # Ignore input and return the preconfigured probs
        return self._probs


def make_blank_image(width=200, height=150, channels=3, value=255):
    """Create a plain image (RGB) useful for tests."""
    return np.full((height, width, channels), value, dtype=np.uint8)


def test_no_faces_detected(monkeypatch):
    # Arrange
    img = make_blank_image(100, 80)

    # Ensure io.imread returns our test image
    monkeypatch.setattr(predict_face.io, 'imread', lambda path: img)

    fd = FakeDetectorNoFaces()

    # A dummy model and class_names (not used since no faces)
    model = FakeModel([1.0])
    class_names = np.array(["person"])

    # Act
    out = predict_face.test_classifier('dummy_path.jpg', fd, model, class_names)

    # Assert: if no faces detected, function returns the original image unchanged
    assert isinstance(out, np.ndarray)
    assert out.shape == img.shape
    # Pixel-by-pixel equality
    assert np.array_equal(out, img)


def test_single_face_prediction_draws_box_and_label(monkeypatch):
    # Arrange
    img = make_blank_image(300, 200, value=200)  # gray background

    # Create a face bbox within image bounds
    face_bbox = {
        'x1': 50,
        'y1': 40,
        'x2': 150,
        'y2': 140,
        'face_num': 0,
        'confidence': 0.99
    }

    # Create a fake extracted face (256x256) expected by the model
    extracted_face = make_blank_image(256, 256, value=120)

    fd = FakeDetectorOneFace(face_bbox, extracted_face)

    # Fake model predicts class 0 with high confidence
    model = FakeModel([0.9, 0.1])
    class_names = np.array(["ironman", "thor"])

    # Ensure io.imread returns our test image
    monkeypatch.setattr(predict_face.io, 'imread', lambda path: img.copy())

    # Act
    out = predict_face.test_classifier('dummy_path.jpg', fd, model, class_names)

    # Assert
    assert out.shape == img.shape
    # The returned image should NOT be exactly equal to the input (because rectangles/labels drawn)
    assert not np.array_equal(out, img)

    # Check that a pixel on the top-left corner of the rectangle was changed to the drawn color.
    # predict_face uses cv2.rectangle(display_img, (x1,y1), (x2,y2), (0,255,0), 2)
    # The rectangle stroke thickness is 2, so check pixel at (y1, x1)
    y, x = face_bbox['y1'], face_bbox['x1']
    # Note: io.imread returns an RGB image; the drawn color is (0,255,0) which should be green in RGB
    pixel = out[y, x]
    assert (pixel == np.array([0, 255, 0])).all(), f"Expected green pixel at {(x,y)}, got {pixel}"


if __name__ == '__main__':
    pytest.main([os.path.abspath(__file__)])
