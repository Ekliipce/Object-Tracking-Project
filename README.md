# Object Tracking with YOLO and Matcher

## Project Description
This project aims to implement an advanced object tracking system using the YOLO (You Only Look Once) deep learning model for object detection, combined with a custom tracking algorithm. The tracking algorithm uses a combination of Intersection over Union (IoU), visual similarity (using embeddings from EfficientNet), and Kalman filters for effective and robust object tracking. The system handles track initialization, updating, and termination based on various factors like detection confidence, longevity, and appearance consistency.

## Key Concepts
- **YOLO Object Detection**: Utilizes YOLOv5 for real-time object detection.
- **Matcher Algorithm**: Custom algorithm for tracking objects across frames. It uses IoU, visual similarity, and Kalman Filters.
- **Kalman Filter**: Predicts the state of a moving object based on its previous state.
- **Embedding Similarity**: Uses EfficientNet to extract feature embeddings for visual similarity calculation.
- **Longevity Management**: Tracks the persistence of objects across frames and manages the termination of old tracks.

## Dependencies
The project requires the following libraries:
- OpenCV (`cv2`): For image and video processing.
- PyTorch (`torch`): As the backbone for YOLOv5 and tensor operations.
- Torchvision: For image transformations and EfficientNet.
- NumPy: For numerical computations.
- SciPy: Specifically for `linear_sum_assignment` used in the Matcher.
- Memory Profiler: For memory profiling (optional).

## Installation
Ensure you have Python 3.x installed. You can install the required libraries using:
```bash
pip install opencv-python-headless torch torchvision numpy scipy memory-profiler
```

## Usage
### Running YOLO Predictions
To run the YOLO model for object detection on a set of video frames and output the detections:
```bash
python yolo_predictions.py --video_input path/to/input/frames --output_file path/to/output/detections.txt
```

### Running the Tracker
After generating the detection file with YOLO, you can run the tracker as follows:
```bash
python tracker.py --det_file path/to/detections.txt --video_input path/to/input/frames --output_video path/to/output/video.avi --output_file path/to/output/solution.txt
```

## Additional Notes
- Make sure the input video frames are sequentially named and placed in the specified directory.
- The output from the tracker (`sol.txt`) will contain the tracked object information for each frame.
- Adjust the `iou_weight` and `similarity_weight` arguments in `tracker.py` to fine-tune the tracking based on your specific requirements.