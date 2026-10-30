\# Driver Safety System



A Python-based desktop prototype that monitors driver drowsiness and detects surrounding objects using a laptop webcam.



\## Project Overview



The Driver Safety System combines facial landmark monitoring and YOLO-based object detection to demonstrate driver-safety monitoring features.



\*\*This is an academic prototype, not a certified vehicle safety system. It must not be relied upon for real-world driving decisions.\*\*



\## Features



\- Real-time webcam monitoring.

\- Face and eye landmark detection using MediaPipe.

\- Drowsiness warning based on prolonged eye closure.

\- General object detection using YOLO11n.

\- Custom object detection using a trained YOLO11n model.

\- Color-coded object bounding boxes and labels.

\- Duplicate phone detection suppression.

\- Audible warning alerts.

\- FPS display for performance monitoring.

\- CPU-based inference support.



\## Object Detection Classes



\### General Object Detection



Uses the COCO-trained YOLO model.



\### Custom Object Detection



The custom model detects these 10 classes:



1\. Card

2\. Charger

3\. Earphones

4\. Glasses

5\. Keys

6\. Pen

7\. Phone

8\. Sunglasses

9\. Wallet

10\. Watch



\## Technology Stack



\- Python

\- OpenCV

\- MediaPipe

\- PyTorch

\- Torchvision

\- Ultralytics YOLO

\- NumPy

\- OpenCV-based webcam interface



\## Project Structure



```text

driver-safety-system/

├── models/

├── scripts/

├── src/

│   ├── alerts.py

│   ├── app.py

│   ├── custom\_object\_detection.py

│   ├── drowsiness.py

│   └── object\_detection.py

├── tests/

├── requirements.txt

├── .gitignore

└── README.md

```



Dataset files, training runs, temporary outputs, and environment-specific files are excluded from the repository.



\## Requirements



\- Windows 10 or Windows 11

\- Python 3.11

\- Webcam

\- Speaker or headphones for audible alerts

\- Recommended: 16 GB RAM

\- CPU execution supported; a dedicated GPU is not required.



\## Installation



Clone the repository:



```powershell

git clone YOUR\_GITHUB\_REPOSITORY\_URL

cd driver-safety-system

```



Create a virtual environment:



```powershell

py -3.11 -m venv .venv

```



Activate it:



```powershell

.\\.venv\\Scripts\\Activate.ps1

```



Install dependencies:



```powershell

python -m pip install --upgrade pip

pip install -r requirements.txt

```



Ensure the required model files are placed inside the `models/` directory.



\## Running the Application



From the project root, run:



```powershell

python -m src.app

```



Press \*\*Q\*\* to quit the application.



\## Model Evaluation



The custom object detection model was evaluated on an independent test split.



| Metric | Result |

|---|---:|

| Test images | 849 |

| Test instances | 1,053 |

| Precision | 0.7311 |

| Recall | 0.6925 |

| mAP@50 | 0.6943 |

| mAP@50-95 | 0.5376 |

| Inference time | 16.9 ms/image |



The custom model was trained for 30 epochs at an image size of 416 using a CPU-based training environment.



\### Known Limitations



\- Detection performance varies across object classes.

\- Keys and pens have comparatively lower detection performance.

\- Webcam lighting, camera quality, object position, and occlusion can affect results.

\- CPU inference performance depends on the computer's hardware.

\- Drowsiness detection is a prototype and is not a substitute for a certified driver-monitoring system.



\## Testing



The project was manually tested for:



\- Application startup and webcam access.

\- Face and eye landmark detection.

\- Eye closure and drowsiness warning behavior.

\- Audible warning and reset behavior.

\- General object detection.

\- Custom object detection.

\- Duplicate phone detection suppression.

\- Application stability during a five-minute integration test.



Webcam and speaker tests require compatible local hardware.



\## Dataset and Attribution



The custom object detection model was trained using the \*\*lost-and-found Dataset\*\*, obtained from Roboflow Universe.



\- \*\*Dataset:\*\* lost-and-found

\- \*\*Creator:\*\* New Workspace

\- \*\*Platform:\*\* Roboflow Universe

\- \*\*Year:\*\* 2026

\- \*\*License:\*\* CC BY 4.0

\- \*\*Source:\*\* https://universe.roboflow.com/dj-ct2wf/lost-and-found-9oqhn-ddqnl



\### Modifications



The original dataset was prepared for object detection by:

\- Converting polygon annotations into enclosing bounding boxes.

\- Repairing detection annotations where required.

\- Organizing the dataset into training, validation, and test splits.



The original dataset files are not included in this repository.



License information: https://creativecommons.org/licenses/by/4.0/



\## Disclaimer



This project is developed for educational and demonstration purposes only. It is not a certified automotive safety product and must not be used as the sole basis for driver-safety decisions.



\## License



License and third-party model attribution details will be added after reviewing the applicable licenses and redistribution permissions.
