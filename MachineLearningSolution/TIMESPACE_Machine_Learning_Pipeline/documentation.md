# TIME SPACE Machine Learning Pipeline Documentation

This pipeline performs face detection and face classification in two stages 
1. Face Detection - using the YuNet model to extract face regions from input images
2. Face Classification - using an InceptionV3-based neural network to classify and verify the extracted faces

This script refactors the original Colab notebook into a CLI tool you can run
from the terminal. It:
1. downloads YuNet weights if missing
2. prepares face crops using YuNet from images/<split>/<class> directories
3. trains an InceptionV3-based classifier
4. evaluates on a validation set
5. runs prediction on images with drawn boxes + labels

# Set up and Environment
## Dependencies
Make sure you have the following installed: 

````
pip install requirements.txt
````

## Folder Structure
Your dataset should be organized as: 
````
images/
├── train/
│   ├── person1/
│   │   ├── img1.jpg
│   │   ├── img2.jpg
│   └── person2/
│       ├── img1.jpg
│       ├── img2.jpg
├── val/
│   ├── person1/
│   └── person2/
└── test/
    ├── person1/
    └── person2/
`````

# Usage
## 1) Train
````
  python time_space_machine_learning_pipeline.py train \
      --data_root /path/to/dataset \
      --epochs 50 --batch_size 32 --img_size 256 --patience 25 \
      --output_dir ./artifacts
````

* **`--data_root /path/to/dataset`** – Path to your dataset root folder containing the image directories.
* **`--epochs 50`** – Number of full training cycles through the dataset. More epochs allow better learning but take longer.
* **`--batch_size 32`** – Number of images processed in one training step; affects training speed and memory use.
* **`--img_size 256`** – Target size (width and height) for face images fed into the model (e.g., 256×256 pixels).
* **`--patience 25`** – Number of epochs with no improvement in validation loss before early stopping occurs.
* **`--output_dir ./artifacts`** – Directory where trained model, class labels, checkpoints, and logs will be saved.


## 2) Evaluate (uses saved model + label classes)
````
  python time_space_machine_learning_pipeline.py eval \
      --data_root /path/to/dataset \
      --model ./artifacts/inception_model.keras \
      --labels ./artifacts/class_names.npy
````
* `--data_root /path/to/dataset` – Path to your dataset root folder containing the evaluation images.

* `--model ./artifacts/inception_model.keras` – Path to the trained model file to be evaluated.

* `--labels ./artifacts/class_names.npy` – Path to the saved class labels file to interpret prediction results.

## 3) Predict on one or more images
````
  python time_space_machine_learning_pipeline.py predict \
      --model ./artifacts/inception_model.keras \
      --labels ./artifacts/class_names.npy \
      --images /path/to/img1.jpg /path/to/img2.png \
      --save_dir ./predictions

````

* **`--model ./artifacts/inception_model.keras`** – Path to the trained model file you generated during training.
* **`--labels ./artifacts/class_names.npy`** – Path to the saved class labels file that maps predictions to human-readable class names.
* **`--images /path/to/img1.jpg /path/to/img2.png`** – One or more image files you want to run predictions on.
* **`--save_dir ./predictions`** – Directory where output images with bounding boxes and predicted labels will be saved.


# Notes
1. Plots are optional and suppressed in CLI; metrics print to stdout.
2. For reproducibility you can set --seed.