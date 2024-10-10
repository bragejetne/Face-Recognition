# Facial Recognition Program

This project implements a facial recognition system using Python. It utilizes machine learning models for detecting and recognizing faces in images or video streams. The core functionality is built using OpenCV, TensorFlow, and other libraries for image processing and deep learning.

## Features
- **Face Detection:** Uses pre-trained Haar Cascade models to detect faces in images or video streams.
- **Facial Recognition:** Employs deep learning techniques to identify and verify faces.
- **Flask API:** Provides a RESTful API for face detection and recognition via HTTP requests.

## Requirements

To run this program, you need to install the required dependencies listed in the `requirements.txt` file. It's recommended to use a virtual environment to manage these dependencies.

### Dependencies
The following libraries are required to run the program:

- `requests>=2.27.1`
- `numpy>=1.14.0`
- `pandas>=0.23.4`
- `gdown>=3.10.1`
- `tqdm>=4.30.0`
- `Pillow>=5.2.0`
- `opencv-python>=4.5.5.64`
- `tensorflow>=1.9.0`
- `keras>=2.2.0`
- `Flask>=1.1.2`
- `flask_cors>=4.0.1`
- `mtcnn>=0.1.0`
- `retina-face>=0.0.1`
- `fire>=0.4.0`
- `gunicorn>=20.1.0`

You can find all these dependencies listed in the `requirements.txt` file.

## Setup Instructions

1. **Clone the Repository:**
   Clone this repository to your local machine.

   ```bash
   git clone <repository_url>
   cd <repository_directory>

