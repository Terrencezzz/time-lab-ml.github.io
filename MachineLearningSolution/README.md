# FaceTrack - Face Detection and Recognition System

A Python module for detecting and recognizing faces in group photos using deep learning. FaceTrack combines MTCNN face detection with pre-trained face recognition models to identify known individuals in group photographs.

## 🌟 Features

- **Face Detection**: Uses MTCNN (Multi-task CNN) for robust face detection in images
- **Face Recognition**: Employs deep learning models (Inception-based) for face verification
- **Automatic Preprocessing**: Upscales small faces and handles various image formats
- **Batch Processing**: Can identify multiple people in a single group photo
- **JSON API**: Returns structured results for easy integration
- **Model Caching**: Optimized performance with lazy loading and model caching

## 📁 Project Structure

```
MachineLearningSolution/
├── FaceTrack.py              # Main face recognition module
├── test_facetrack.py         # Comprehensive test suite
├── inception_model.keras     # Pre-trained face recognition model
├── requirements.txt          # Python dependencies
├── avengersGroup/           # Sample group photos for testing
├── avengersTest/            # Sample individual photos for reference
├── extracted_faces/         # Temporary directory for extracted faces
└── README.md               # This file
```

## 🚀 Quick Start

### Installation

1. **Clone the repository** (if not already done):
   ```bash
   git clone <repository-url>
   cd MachineLearningSolution
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation** by running the test:
   ```bash
   python test_facetrack.py
   ```

### Basic Usage

```python
from pathlib import Path
import json
import FaceTrack

# Setup paths
group_image = Path("path/to/group_photo.jpg")
known_people_dir = Path("path/to/known_people/")

# Run face recognition
result_json = FaceTrack.find_people_in_group_simple(group_image, known_people_dir)
result = json.loads(result_json)

# Display results
print(f"Found {len(result['found'])} people:")
for person in result['found']:
    print(f"- {person}")
```

### Command Line Usage

Run the module directly with the provided test data:

```bash
python FaceTrack.py
```

This will analyze the sample Avengers group photo and identify known characters.

## 📋 API Reference

### Main Function

#### `find_people_in_group_simple(group_img, person_directory, *, verbose=True, model_name="Inception")`

Identifies known people in a group photograph.

**Parameters:**
- `group_img` (Path): Path to the group photograph
- `person_directory` (Path): Directory containing reference images of known people
- `verbose` (bool): Print detailed progress information (default: True)
- `model_name` (str): Embedding model to use (default: "Inception")

**Returns:**
- `str`: JSON string with results:
  ```json
  {
    "status": true,
    "found": ["person1", "person2"],
    "total_faces": 5
  }
  ```

### Core Functions

#### `extract_faces(group_path, out_dir, min_size=112)`

Extract all faces from an image using MTCNN.

**Parameters:**
- `group_path` (Path): Input image path
- `out_dir` (Path): Output directory for extracted faces
- `min_size` (int): Minimum face size (smaller faces are upscaled)

**Returns:**
- `list[Path]`: Paths to extracted face images

#### `get_embedding(img_path, model_name="Inception")`

Generate face embedding vector from an image.

**Parameters:**
- `img_path` (Path): Path to face image
- `model_name` (str): Model to use for embedding

**Returns:**
- `np.ndarray`: Face embedding vector

#### `verify_faces(face_paths, person_path, threshold=0.4, model_name="Inception")`

Verify extracted faces against a known person.

**Parameters:**
- `face_paths` (list[Path]): Extracted face image paths
- `person_path` (Path): Reference image of known person
- `threshold` (float): Cosine distance threshold for matching
- `model_name` (str): Embedding model to use

**Returns:**
- `tuple[int, int]`: (matches, misses)

## ⚙️ Configuration

The module uses several configurable constants:

```python
MIN_SIZE = 112      # Minimum face size in pixels
THRESHOLD = 0.4     # Cosine distance threshold for matching
DEFAULT_MODEL_PATH  # Path to the default model file
```

### Threshold Guidelines

- **0.2-0.3**: Very strict matching (high precision, low recall)
- **0.4-0.5**: Balanced matching (recommended for most use cases)
- **0.6-0.7**: Loose matching (high recall, lower precision)

## 🧪 Testing

The project includes a comprehensive test suite covering:

- Face extraction functionality
- Embedding generation
- Face verification accuracy
- Distance calculations
- End-to-end workflows
- Error handling

Run tests with:
```bash
python test_facetrack.py
# or
pytest test_facetrack.py -v
```

### Test Coverage

- **Unit Tests**: Individual function testing
- **Integration Tests**: Full pipeline testing with real images
- **Edge Cases**: Error handling and boundary conditions
- **Mock Testing**: Isolated component testing

## 🔧 Dependencies

Core dependencies (see `requirements.txt` for versions):

- **opencv-contrib-python**: Image processing and face detection
- **numpy**: Numerical computing
- **tensorflow**: Deep learning models
- **mtcnn**: Face detection
- **scikit-image**: Additional image utilities

Development dependencies:
- **pytest**: Testing framework
- **unittest.mock**: Test mocking

## 📊 Performance

### Typical Processing Times

- **Face Detection**: ~1-3 seconds per image (depends on image size and number of faces)
- **Embedding Generation**: ~0.1-0.5 seconds per face
- **Model Loading**: ~2-5 seconds (cached after first load)

### Memory Usage

- **Model Memory**: ~200-400 MB (cached in memory)
- **Image Processing**: Depends on image size and number of faces

## 🔍 How It Works

### 1. Face Detection
- Uses MTCNN (Multi-task CNN) for detecting faces in images
- Handles various face orientations and lighting conditions
- Automatically crops detected faces with bounding boxes

### 2. Face Preprocessing
- Resizes faces to consistent dimensions (model input size)
- Normalizes pixel values to [0, 1] range
- Upscales small faces to minimum resolution for better accuracy

### 3. Feature Extraction
- Uses pre-trained Inception-based model to generate face embeddings
- Embeddings are high-dimensional vectors representing facial features
- Model is trained to produce similar embeddings for the same person

### 4. Face Verification
- Compares embeddings using cosine distance
- Distance below threshold indicates a match
- Cosine distance is robust to embedding magnitude variations

## 🎯 Use Cases

- **Event Photography**: Identify attendees in group photos
- **Security Systems**: Verify known individuals in surveillance footage
- **Photo Organization**: Automatically tag people in photo collections
- **Social Media**: Auto-tagging features
- **Attendance Systems**: Automated attendance tracking

## ⚠️ Limitations and Considerations

### Technical Limitations

- **Face Quality**: Works best with clear, front-facing faces
- **Lighting**: Performance degrades in poor lighting conditions
- **Occlusion**: Partially covered faces may not be detected
- **Face Size**: Very small faces (< 30x30 pixels) may be missed

### Ethical Considerations

- **Privacy**: Ensure proper consent before using face recognition
- **Bias**: Model performance may vary across different demographics
- **Data Protection**: Handle biometric data according to local regulations
- **Transparency**: Inform users when face recognition is being used

## 🐛 Troubleshooting

### Common Issues

1. **"Could not read image"**
   - Check file path and format
   - Ensure image file is not corrupted
   - Verify file permissions

2. **"Model not found"**
   - Ensure `inception_model.keras` is in the correct directory
   - Check file permissions
   - Re-download model if necessary

3. **Poor recognition accuracy**
   - Adjust threshold parameter
   - Ensure reference images are clear and high quality
   - Check lighting conditions in images

4. **Memory errors**
   - Reduce image size before processing
   - Process images in batches
   - Close other memory-intensive applications

### Debug Mode

Enable verbose logging for detailed information:
```python
result = FaceTrack.find_people_in_group_simple(
    group_img, person_dir, verbose=True
)
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Add tests for new functionality
4. Ensure all tests pass (`python test_facetrack.py`)
5. Commit changes (`git commit -m 'Add amazing feature'`)
6. Push to branch (`git push origin feature/amazing-feature`)
7. Create a Pull Request

### Development Guidelines

- Follow PEP 8 style guidelines
- Add comprehensive docstrings to new functions
- Include unit tests for new features
- Update this README for significant changes

## 📄 License

This project is part of the TIME-Space Machine Learning initiative. See the repository's main LICENSE file for details.

## 🙏 Acknowledgments

- **MTCNN**: Face detection implementation
- **TensorFlow**: Deep learning framework
- **OpenCV**: Computer vision utilities
- **TIME-Space Lab**: Research and development support

## 📞 Support

For questions, issues, or contributions:

1. Check existing issues in the repository
2. Create a new issue with detailed description
3. Contact the TIME-Space Machine Learning team

---

**Note**: This module is designed for research and educational purposes. For production use, consider additional testing, validation, and compliance with relevant regulations.
