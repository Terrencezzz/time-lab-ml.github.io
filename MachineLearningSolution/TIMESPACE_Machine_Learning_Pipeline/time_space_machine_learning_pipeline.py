#!/usr/bin/env python3
"""
TIME SPACE Face Pipeline — Command-line Training & Inference

This script refactors the original Colab notebook into a CLI tool you can run
from the terminal. It:
  • downloads YuNet weights if missing
  • prepares face crops using YuNet from images/<split>/<class> directories
  • trains an InceptionV3-based classifier
  • evaluates on a validation set
  • runs prediction on images with drawn boxes + labels

"""

from __future__ import annotations
import argparse
import os
import sys
import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np
import cv2
from skimage import io

# Silence TF logs before import
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report

YU_WEIGHTS_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/"
    "face_detection_yunet_2023mar.onnx"
)


# ------------------------------ Utils ---------------------------------------

def str2bool(v: str) -> bool:
    return str(v).lower() in {"1", "true", "t", "yes", "y"}


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def _to_rgb(img: np.ndarray) -> np.ndarray:
    """Ensure image is 3-channel RGB (handles GRAY and RGBA)."""
    if img is None:
        return img
    if img.ndim == 2:  # GRAY -> RGB
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if img.ndim == 3 and img.shape[2] == 4:  # RGBA -> RGB
        return cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    return img  # already RGB (3 channels)



# --------------------------- YuNet Face Detector ----------------------------

class FaceDetectorYunet:
    def __init__(self, model_path: Path, img_size=(300, 300), threshold=0.5):
        self.model_path = str(model_path)
        self.img_size = tuple(img_size)
        if not Path(self.model_path).exists():
            raise FileNotFoundError(
                f"YuNet weights not found at {self.model_path}. Call download_yunet_weights() first."
            )
        self.fd = cv2.FaceDetectorYN_create(self.model_path, "", self.img_size, score_threshold=float(threshold))

    @staticmethod
    def download_yunet_weights(dest: Path) -> Path:
        import urllib.request
        ensure_dir(dest.parent)
        logging.info("Downloading YuNet weights → %s", dest)
        urllib.request.urlretrieve(YU_WEIGHTS_URL, dest.as_posix())
        return dest

    def _scale_coords(self, image: np.ndarray, prediction: dict) -> dict:
        ih, iw = image.shape[:2]
        rw, rh = self.img_size
        a = np.array(
            [
                (prediction['x1'], prediction['y1']),
                (prediction['x1'] + prediction['x2'], prediction['y1'] + prediction['y2'])
            ]
        )
        b = np.array([iw / rw, ih / rh])
        c = a * b
        prediction['img_width'] = iw
        prediction['img_height'] = ih
        prediction['x1'] = int(round(c[0, 0]))
        prediction['x2'] = int(round(c[1, 0]))
        prediction['y1'] = int(round(c[0, 1]))
        prediction['y2'] = int(round(c[1, 1]))
        prediction['face_width'] = (c[1, 0] - c[0, 0])
        prediction['face_height'] = (c[1, 1] - c[0, 1])
        prediction['area'] = prediction['face_width'] * prediction['face_height']
        prediction['pct_of_frame'] = prediction['area'] / (prediction['img_width'] * prediction['img_height'])
        return prediction

    def detect(self, image: np.ndarray | str):
        if isinstance(image, (str, Path)):
            image = cv2.imread(str(image))
        if image is None:
            return None
        img = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        img = cv2.resize(img, self.img_size)
        self.fd.setInputSize(self.img_size)
        _, faces = self.fd.detect(img)
        if faces is None:
            return None
        data = []
        for num, face in enumerate(list(faces)):
            x1, y1, x2, y2 = list(map(int, face[:4]))
            landmarks = list(map(int, face[4:len(face)-1]))
            landmarks = np.array_split(landmarks, len(landmarks) / 2)
            positions = ['left_eye', 'right_eye', 'nose', 'right_mouth', 'left_mouth']
            landmarks = {positions[num]: x.tolist() for num, x in enumerate(landmarks)}
            confidence = float(face[-1])
            datum = {
                'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                'face_num': num, 'landmarks': landmarks,
                'confidence': confidence, 'model': 'yunet'
            }
            data.append(self._scale_coords(image, datum))
        return data

    def extract_face(self, image: np.ndarray | str, face: dict, target_size=(256, 256)) -> np.ndarray:
        if isinstance(image, (str, Path)):
            image = cv2.imread(str(image))
        if image is None:
            raise ValueError("Invalid image for face extraction")
        x1, y1, x2, y2 = face['x1'], face['y1'], face['x2'], face['y2']
        h, w = image.shape[:2]
        margin = 10
        x1 = max(0, x1 - margin)
        y1 = max(0, y1 - margin)
        x2 = min(w, x2 + margin)
        y2 = min(h, y2 + margin)
        face_region = image[y1:y2, x1:x2]
        face_region = cv2.resize(face_region, target_size)
        return face_region


# ---------------------- Data Preparation with YuNet -------------------------

def prepare_classification_data(base_dir: Path, face_detector: FaceDetectorYunet, split: str = 'train',
                                target_size: Tuple[int, int] = (256, 256)) -> Tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    split_dir = base_dir / 'images' / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Expected directory not found: {split_dir}")

    classes = [d for d in sorted(os.listdir(split_dir)) if (split_dir / d).is_dir()]
    logging.info("Preparing %s split from %s (classes=%d)", split, split_dir, len(classes))

    for cls in classes:
        cls_dir = split_dir / cls
        img_files = [p for p in cls_dir.iterdir() if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}]
        for img_path in img_files:
            try:
                img = io.imread(img_path.as_posix())
                faces = face_detector.detect(img)
                if faces:
                    best_face = max(faces, key=lambda x: x['confidence'])
                    face_img = face_detector.extract_face(img, best_face, target_size)
                    X.append(face_img)
                    y.append(cls)
            except Exception as e:
                logging.warning("Error processing %s: %s", img_path, e)
    X = np.array(X)
    y = np.array(y)
    logging.info("Prepared %d face crops across %d classes for split '%s'", len(X), len(set(y.tolist() or [''])), split)
    return X, y


# --------------------------- Model Definition -------------------------------

def create_classification_model(input_shape: Tuple[int, int, int], num_classes: int) -> keras.Model:
    data_augmentation = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.3),
        layers.RandomZoom(0.3),
        layers.RandomContrast(0.3),
    ])

    base_model = keras.applications.InceptionV3(include_top=False, input_shape=input_shape)
    for layer in base_model.layers:
        layer.trainable = False

    model = keras.Sequential([
        layers.Input(shape=input_shape),
        data_augmentation,
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dense(32, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax'),
    ])

    model.compile(
        loss=keras.losses.CategoricalCrossentropy(),
        optimizer=keras.optimizers.Adam(),
        metrics=['accuracy']
    )
    return model


# ------------------------------ Commands ------------------------------------

def cmd_train(args: argparse.Namespace) -> None:
    logging.info("Starting training with args: %s", vars(args))

    weights_path = Path(args.yunet_weights)
    if not weights_path.exists():
        logging.info("YuNet weights not found; downloading...")
        FaceDetectorYunet.download_yunet_weights(weights_path)

    detector = FaceDetectorYunet(weights_path, img_size=(args.detector_w, args.detector_h), threshold=args.threshold)
    base_dir = Path(args.data_root)

    # Build dataset via detector (train split always needed)
    X, y = prepare_classification_data(base_dir, detector, split='train', target_size=(args.img_size, args.img_size))
    if len(X) == 0:
        logging.error("No face data extracted from training set. Check dataset and YuNet threshold.")
        sys.exit(2)

    # --- Ensure all training crops are 3-channel RGB ---
    X = np.array([_to_rgb(x) for x in X])

    # Encode labels
    label_encoder = LabelEncoder()
    y_enc = label_encoder.fit_transform(y)
    y_cat = keras.utils.to_categorical(y_enc)

    # Optional separate validation split directory; if missing, split from train
    val_dir = base_dir / 'images' / 'val'
    if val_dir.exists():
        X_val, y_val = prepare_classification_data(base_dir, detector, split='val', target_size=(args.img_size, args.img_size))
        # --- Ensure all val crops are 3-channel RGB ---
        X_val = np.array([_to_rgb(x) for x in X_val])

        y_val_enc = label_encoder.transform(y_val)
        y_val_cat = keras.utils.to_categorical(y_val_enc)
        X_train, y_train = X, y_cat
    else:
        X_train, X_val, y_train, y_val_cat = train_test_split(X, y_cat, test_size=0.2, random_state=args.seed)

    # Normalize
    X_train = X_train.astype('float32') / 255.0
    X_val = X_val.astype('float32') / 255.0

    input_shape = (args.img_size, args.img_size, 3)
    num_classes = len(label_encoder.classes_)
    model = create_classification_model(input_shape, num_classes)

    callbacks = [
        keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=args.patience, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(filepath=(Path(args.output_dir)/'checkpoint.keras').as_posix(), monitor='val_loss', save_best_only=True),
    ]

    datagen = keras.preprocessing.image.ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2,
    )

    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=args.batch_size),
        epochs=args.epochs,
        validation_data=(X_val, y_val_cat),
        callbacks=callbacks,
        verbose=1,
    )

    # Save artifacts
    out_dir = ensure_dir(Path(args.output_dir))
    model_path = out_dir / 'inception_model.keras'
    labels_path = out_dir / 'class_names.npy'
    model.save(model_path.as_posix())
    np.save(labels_path.as_posix(), label_encoder.classes_)

    # Metrics
    val_loss, val_acc = model.evaluate(X_val, y_val_cat, verbose=0)
    logging.info("Validation accuracy: %.2f%% | loss: %.4f", val_acc * 100.0, val_loss)

    # Optional confusion matrix
    y_pred = model.predict(X_val, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_val_cat, axis=1)
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    logging.info("Confusion matrix:\n%s", cm)
    logging.info("Classification report:\n%s", classification_report(y_true_classes, y_pred_classes, target_names=label_encoder.classes_))

    logging.info("Saved model → %s", model_path)
    logging.info("Saved label classes → %s", labels_path)


def _load_model_and_labels(model_path: Path, labels_path: Path) -> tuple[keras.Model, np.ndarray]:
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    if not labels_path.exists():
        raise FileNotFoundError(labels_path)
    model = keras.models.load_model(model_path.as_posix())
    class_names = np.load(labels_path.as_posix())
    return model, class_names


def cmd_eval(args: argparse.Namespace) -> None:
    logging.info("Evaluating model: %s", args.model)

    # Prepare detector for val set cropping if needed
    weights_path = Path(args.yunet_weights)
    if not weights_path.exists():
        logging.info("YuNet weights not found; downloading...")
        FaceDetectorYunet.download_yunet_weights(weights_path)
    detector = FaceDetectorYunet(weights_path, img_size=(args.detector_w, args.detector_h), threshold=args.threshold)

    base_dir = Path(args.data_root)
    X_val, y_val = prepare_classification_data(base_dir, detector, split='test', target_size=(args.img_size, args.img_size))
    if len(X_val) == 0:
        logging.error("No face data extracted from validation set.")
        sys.exit(2)

    X_val = X_val.astype('float32') / 255.0

    model, class_names = _load_model_and_labels(Path(args.model), Path(args.labels))

    # Map y_val strings to indices of class_names
    le = LabelEncoder().fit(class_names)
    y_val_enc = le.transform(y_val)
    y_val_cat = keras.utils.to_categorical(y_val_enc)

    val_loss, val_acc = model.evaluate(X_val, y_val_cat, verbose=0)
    logging.info("Validation accuracy: %.2f%% | loss: %.4f", val_acc * 100.0, val_loss)

    y_pred = model.predict(X_val, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_val_cat, axis=1)
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    logging.info("Confusion matrix:\n%s", cm)
    logging.info("Classification report:\n%s", classification_report(y_true_classes, y_pred_classes, target_names=class_names))


def draw_prediction(image: np.ndarray, face: dict, label: str) -> np.ndarray:
    x1, y1, x2, y2 = face['x1'], face['y1'], face['x2'], face['y2']
    color = (0, 255, 0)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    cv2.putText(image, label, (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return image


def cmd_predict(args: argparse.Namespace) -> None:
    logging.info("Predicting on %d image(s)", len(args.images))

    weights_path = Path(args.yunet_weights)
    if not weights_path.exists():
        logging.info("YuNet weights not found; downloading...")
        FaceDetectorYunet.download_yunet_weights(weights_path)
    detector = FaceDetectorYunet(weights_path, img_size=(args.detector_w, args.detector_h), threshold=args.threshold)

    model, class_names = _load_model_and_labels(Path(args.model), Path(args.labels))

    save_dir = ensure_dir(Path(args.save_dir))

    for img_path_str in args.images:
        img_path = Path(img_path_str)

        # skimage.io.imread can return RGBA; keep as loaded then normalize channels
        img = io.imread(img_path.as_posix())
        img = _to_rgb(img)  # ensure 3 channels
        disp = img.copy()

        faces = detector.detect(img)
        if not faces:
            logging.warning("No faces detected: %s", img_path)
            continue

        for face in faces:
            face_img = detector.extract_face(img, face, target_size=(args.img_size, args.img_size))
            face_img = _to_rgb(face_img)  # force RGB before normalization

            face_norm = face_img.astype('float32') / 255.0
            pred = model.predict(np.expand_dims(face_norm, axis=0), verbose=0)[0]
            cls_idx = int(np.argmax(pred))
            conf = float(np.max(pred)) * 100.0
            label = f"{class_names[cls_idx]}: {conf:.1f}%"
            disp = draw_prediction(disp, face, label)

        # disp is RGB now; convert to BGR for cv2.imwrite
        out_path = save_dir / f"{img_path.stem}_pred.jpg"
        cv2.imwrite(out_path.as_posix(), cv2.cvtColor(disp, cv2.COLOR_RGB2BGR))
        logging.info("Saved: %s", out_path)


# ----------------------------- CLI Parsing ----------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="YuNet + InceptionV3 face pipeline (train/eval/predict)")
    p.add_argument('--yunet_weights', type=str, default='./face_detection_yunet_2023mar.onnx', help='Path to YuNet .onnx file (auto-download if missing)')
    p.add_argument('--detector_w', type=int, default=300, help='YuNet input width')
    p.add_argument('--detector_h', type=int, default=300, help='YuNet input height')
    p.add_argument('--threshold', type=float, default=0.5, help='YuNet score threshold')
    p.add_argument('--img_size', type=int, default=256, help='Face crop size for classifier (square)')
    p.add_argument('--seed', type=int, default=42, help='Random seed for splits')
    p.add_argument('--log_level', type=str, default='INFO', help='Logging level (DEBUG, INFO, WARNING, ERROR)')

    sub = p.add_subparsers(dest='cmd', required=True)

    # train
    p_train = sub.add_parser('train', help='Train a classifier')
    p_train.add_argument('--data_root', type=str, required=True, help='Root folder containing images/{train,val,test}')
    p_train.add_argument('--epochs', type=int, default=50)
    p_train.add_argument('--batch_size', type=int, default=32)
    p_train.add_argument('--patience', type=int, default=25)
    p_train.add_argument('--output_dir', type=str, default='./artifacts')
    p_train.set_defaults(func=cmd_train)

    # eval
    p_eval = sub.add_parser('eval', help='Evaluate a saved model on validation set')
    p_eval.add_argument('--data_root', type=str, required=True)
    p_eval.add_argument('--model', type=str, required=True)
    p_eval.add_argument('--labels', type=str, required=True, help='Path to class_names.npy')
    p_eval.set_defaults(func=cmd_eval)

    # predict
    p_pred = sub.add_parser('predict', help='Predict on one or more images')
    p_pred.add_argument('--model', type=str, required=True)
    p_pred.add_argument('--labels', type=str, required=True)
    p_pred.add_argument('--images', type=str, nargs='+', required=True)
    p_pred.add_argument('--save_dir', type=str, default='./predictions')
    p_pred.set_defaults(func=cmd_predict)

    return p


def main(argv: List[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    # Logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format='[%(levelname)s] %(message)s'
    )

    # Set seeds
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    # Dispatch
    args.func(args)


if __name__ == '__main__':
    main()
