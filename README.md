# Real-Time Age and Gender Detection Using Deep Learning

A real-time computer vision application that detects human faces and predicts **age group and gender** using deep learning. The system supports both **image-based analysis and live webcam detection** through an interactive Streamlit interface.

## 📌 Project Overview

The **Real-Time Age and Gender Detection System** uses a Convolutional Neural Network (CNN) trained on the **UTKFace dataset** to perform age-group and gender prediction from facial images.

The system processes an input image or webcam frame, detects the face, preprocesses the detected facial region, and uses the trained deep learning model to predict:

- **Gender:** Male / Female
- **Age:** One of 8 predefined age groups

The application is implemented using **Python, PyTorch, OpenCV, and Streamlit**.

## ✨ Features

- Real-time age and gender prediction
- Supports **live webcam input**
- Supports **image upload**
- Face detection using OpenCV
- CNN-based age and gender classification
- Eight predefined age groups
- Interactive Streamlit web interface
- GPU support when CUDA is available
- Easy local deployment

## 🧠 Age Groups

The system categorizes age into the following eight groups:

| Class | Age Group |
|------:|-----------|
| 0 | 0–2 |
| 1 | 3–9 |
| 2 | 10–19 |
| 3 | 20–29 |
| 4 | 30–39 |
| 5 | 40–49 |
| 6 | 50–69 |
| 7 | 70+ |

## 🏗️ System Workflow

```text
Input Image / Webcam
        ↓
   Face Detection
        ↓
   Face Preprocessing
        ↓
      CNN Model
        ↓
Age & Gender Prediction
        ↓
   Results Display
```

## 🛠️ Technologies Used

- **Python**
- **PyTorch**
- **OpenCV**
- **Streamlit**
- **NumPy**
- **Matplotlib**
- **UTKFace Dataset**

```

## 📊 Dataset

The model is trained using the **UTKFace dataset**, which contains facial images annotated with age and gender information.

The age labels are converted into eight predefined age groups before training.


## 🎯 Applications

The system can be applied in areas such as:

- Security and surveillance
- Retail analytics
- Customer demographic analysis
- Smart monitoring
- Human-computer interaction

## 🔮 Future Enhancements

Possible future improvements include:

- Improved age estimation accuracy
- Emotion recognition
- Face recognition integration
- Support for additional demographic attributes
- Improved performance under challenging lighting and poses
- Cloud-based deployment

## ⚠️ Limitations

Age and gender prediction from facial images is inherently challenging because performance can be affected by:

- Lighting conditions
- Facial expressions
- Camera angle
- Image quality
- Occlusion
- Dataset bias

Therefore, predictions should be considered estimates rather than definitive personal attributes.
