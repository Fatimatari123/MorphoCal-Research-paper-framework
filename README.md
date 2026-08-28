# MorphoCal-Research-paper-framework
Here is a structured, comprehensive README section detailing your workflow, the use of the pre-trained model on the **depondfi** dataset, the pipeline steps, validation methods, and testing results derived from your notebook.

# MorphoCal: Underwater Fish Length Estimation & Pose Analysis Pipeline

This repository contains the complete inference, landmark parsing, and 3D metric validation pipeline for **MorphoCal**, a multi-stage deep learning framework designed for accurate fish length estimation in challenging underwater pond environments.

# The already Pre-tarined model:https://drive.google.com/drive/folders/1n3hnxBX71_ztsoAHyB1rjHJZFEN67uFk

##  1. Model Overview & Dataset Context

* **Pre-trained Weights**: The pipeline leverages an **already trained YOLO-Pose model** fine-tuned on the **depondfi** dataset (`best_updated.pt`), downloaded securely via Google Drive integration.
* **Core Task**: Detects anatomical keypoints (such as fish mouths, centers, and tail outline landmarks) to measure individual fish dimensions accurately despite occlusion and complex underwater lighting.


##  2. Step-by-Step Implementation Pipeline

The pipeline executes the following sequential stages in Google Colab (leveraging GPU acceleration):

1. **Environment & Dependency Setup**:
* Mounts Google Drive to access the workspace.
* Installs and verifies core packages including **Ultralytics YOLO**, `opencv-python-headless`, `numpy`, `matplotlib`, and `pandas`.


2. **Model Weight Retrieval**:
* Automatically fetches the fine-tuned model weights (`best_updated.pt`) from Google Drive using `gdown`.


3. **Batch Inference Execution**:
* Locates the test image directory (`Testing/`) and runs batched inference using YOLO-Pose (`conf=0.4`, `batch=16`) to maximize GPU parallelism and extract structural bounding boxes and keypoint labels.



##  3. Landmark Parsing & Multi-Fish Handling

To convert raw keypoint detections into meaningful measurements, the script implements robust processing logic:

* **Filtering & Parsing**: Reads YOLO text label outputs from the prediction directory, mapping class IDs for mouths and tails to pixel coordinates ($640 \times 640$ frame scale).
* **Biological Constraints**: Enforces strict pixel-length boundaries ($40.0\text{ px} \le \text{Length} \le 450.0\n\text{px}$) to filter out false positives.
* **Multi-Fish Association**: Utilizes nearest-neighbor pairing logic between detected mouths and tails, ensuring that individual fish in crowded multi-fish frames (where multiple specimens appear simultaneously) are correctly matched without double-assigning tails.

##  4. Validation, Testing, and 3D Ray-Plane Projection

To bridge the gap between 2D pixel estimates and real-world metric scales, the pipeline implements **Ray-Plane Projection**:

* **Camera Calibration**: Loads native intrinsic camera parameters ($K_{native}$), image dimensions ($W_{native} = 4498.78$, $H_{native} = 3641.20$), non-linear lens distortion coefficients, and a reference depth plane ($z_{plane} = 50.0\text{ cm}$).
* **Undistortion & Ray Casting**: Applies `cv2.undistortPoints` to correct optical distortion, casting rays from 20 pixel coordinates into 3D metric space.
* **Qualitative & Quantitative Verification**: Evaluates individual test frames (e.g., multi-fish scenes) to confirm that the projected 3D Euclidean distances accurately reflect true physical lengths in centimeters.

##  5. Results & Output Summary

* **Processing Scale**: Successfully evaluated 120 test frames, identifying multi-fish scenarios across 101 frames.
* **Extracted Measurements**: Successfully parsed and verified **199 individual fish length measurements**.
* **Exported Artifacts**: All validated individual measurements are compiled and saved locally to:

