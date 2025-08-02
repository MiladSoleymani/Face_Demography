# Face Demography

A real-time facial emotion detection system using YOLOv8 for face detection and FER (Facial Expression Recognition) for emotion analysis. Supports images, video files, and live webcam streams.

## Features

- **Real-time face detection** using YOLOv8-face model
- **Emotion recognition** with 7 emotion categories (angry, disgust, fear, happy, sad, surprise, neutral)
- **Multiple input sources**: Single images, image folders, video files, and live webcam
- **High-performance processing** with ONNX model optimization
- **Easy-to-use command-line interface**

## Requirements

- Python 3.7+
- OpenCV
- PyTorch/Ultralytics YOLO
- FER (Facial Expression Recognition)
- NumPy

## Installation

1. Clone the repository:
```bash
git clone https://github.com/MiladSoleymani/Face_Demography.git
cd Face_Demography
```

2. Install dependencies:
```bash
pip install opencv-python ultralytics fer numpy
```

## Usage

### Basic Usage

Run the script with different input sources:

```bash
# Process a single image
python script/run.py --source 0 --input_path dataset/img1.jpg

# Process all images in a folder
python script/run.py --source 1 --input_path dataset/

# Process a video file
python script/run.py --source 2 --input_path path/to/video.mp4

# Use live webcam (default)
python script/run.py --source 3
```

### Command Line Arguments

- `--source`: Input source type (required)
  - `0`: Single image
  - `1`: Folder of images
  - `2`: Video file
  - `3`: Live webcam (default)
- `--input_path`: Path to input file/folder (default: `dataset/`)
- `--yolo_face`: Path to YOLO face detection model (default: `models/yolov8n-face.onnx`)
- `--save_path`: Directory to save results (default: `results`)

### Examples

```bash
# Analyze emotions in a single image
python script/run.py --source 0 --input_path dataset/couple.jpg --save_path output/

# Process all images in the dataset folder
python script/run.py --source 1 --input_path dataset/ --save_path batch_results/

# Real-time emotion detection from webcam
python script/run.py --source 3
```

## Project Structure

```
Face_Demography/
├── dataset/              # Sample images for testing
│   ├── couple.jpg
│   └── img[1-11].jpg
├── models/               # Model files and face detection module
│   ├── __init__.py
│   ├── face_detection.py # YOLOv8 face detection wrapper
│   └── weights/          # Model weights
│       ├── yolov8n-face.onnx
│       └── yolov8n.pt
├── script/               # Main execution scripts
│   └── run.py           # Main entry point
├── utils/                # Utility functions
│   ├── images.py        # Image processing utilities
│   ├── videos.py        # Video/camera processing utilities
│   └── utils.py         # General utilities
└── README.md            # This file
```

## How It Works

1. **Face Detection**: Uses YOLOv8-face model to detect faces in the input
2. **Emotion Recognition**: Applies FER (Facial Expression Recognition) to analyze emotions
3. **Visualization**: Draws bounding boxes around faces with emotion labels and confidence scores
4. **Output**: Saves processed images/videos with emotion annotations to the specified directory

## Output

The system provides:
- Bounding boxes around detected faces
- Emotion labels with confidence percentages
- Saved results in the specified output directory
- Real-time display for video/webcam sources

## Performance

- YOLOv8n-face ONNX model ensures fast face detection
- Suitable for real-time applications
- Processes multiple faces simultaneously

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License.
