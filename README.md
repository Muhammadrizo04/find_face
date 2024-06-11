# Employee Face Recognition System

This repository contains code for a real-time employee face recognition system using Python and OpenCV. The system utilizes the dlib library for face detection, facial landmark prediction, and face recognition. It identifies employees by comparing their facial features with a pre-existing database of employees.

## Installation

Before running the code, make sure you have the following libraries installed:

- OpenCV (`cv2`)
- dlib
- NumPy
- pandas

You can install these libraries using pip:

```bash
pip install opencv-python dlib numpy pandas
```

Additionally, you need to download the pre-trained models:

- `shape_predictor_68_face_landmarks.dat`: Facial landmark predictor model
- `dlib_face_recognition_resnet_model_v1.dat`: Face recognition model

## Usage

To use the system:

1. Clone the repository:

```bash
git clone https://github.com/Muhammadrizo04/find_face.git
```

2. Navigate to the repository directory:

```bash
cd find_face
```

3. Run the Python script:

```bash
python face_recognition.py
```

The script will access the camera feed and recognize employees in real-time.

## Data

Employee data is stored in a CSV file named `employees.csv`. This file contains employee names and paths to their photos. Make sure to update this file with the relevant information before running the system.

## Contributing

Feel free to contribute to this project by submitting bug reports, feature requests, or pull requests.

