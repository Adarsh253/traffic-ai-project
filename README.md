Traffic Signal Recognition AI
Overview

This project implements a deep learning-based system for recognizing and classifying traffic signs from images. The application is built using TensorFlow and deployed as an interactive web interface using Streamlit. It supports both image uploads and real-time camera-based detection.

The system leverages transfer learning with MobileNetV2 to achieve efficient and accurate classification across multiple traffic sign categories.

Objectives
Develop an AI-based traffic sign recognition system
Enable real-time prediction through a web interface
Utilize transfer learning for improved performance
Provide remote accessibility for presentation and demonstration
Technologies Used
Python
TensorFlow and Keras
Streamlit
NumPy
Pillow
Scikit-learn
Project Structure

traffic-ai-project/
│
├── app/
│ └── app.py
│
├── model/
│ ├── mobilenetv2_traffic.h5
│ └── class_indices.json
│
├── data/
│ ├── train/
│ └── test/
│
├── prepare_dataset.py
├── train.py
├── requirements.txt
└── README.md

Methodology
Data Preparation

The dataset is prepared using the German Traffic Sign Recognition Benchmark (GTSRB). Images are resized and organized into training and testing directories.

Model Development

The model is built using MobileNetV2 as a base architecture with frozen convolutional layers. A custom classification head is added for traffic sign prediction.

Training

The model is trained using augmented image data to improve generalization. Early stopping and learning rate reduction techniques are used for optimization.

Deployment

The trained model is integrated into a Streamlit-based web application, allowing users to interact with the system via a browser.

Installation and Setup
1. Clone the Repository

git clone https://github.com/your-username/traffic-ai-project.git

cd traffic-ai-project

2. Create Virtual Environment

python -m venv venv
venv\Scripts\activate

3. Install Dependencies

pip install -r requirements.txt

Running the Application

streamlit run app/app.py

Usage
Upload an image containing a traffic sign, or
Capture an image using the integrated camera

The application will display the predicted class along with a confidence score.

Model Details
Architecture: MobileNetV2 (Transfer Learning)
Input Size: 96 x 96
Output: Multi-class classification
Dataset: GTSRB (custom subset or full dataset)
Deployment

The application can be deployed using Streamlit Community Cloud:

Push the project to a GitHub repository
Connect the repository to Streamlit Cloud
Set the entry point to app/app.py
Deploy the application
Future Enhancements
Support for full traffic sign dataset (40+ classes)
Real-time video stream processing
Integration with embedded systems or IoT devices
Mobile application interface
Author

Adarsh
B.Tech Computer Science (AI/ML)