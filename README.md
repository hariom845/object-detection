# Gun Object Detection System (MLOps Pipeline)

This project implements a deep learning based gun detection system capable of identifying firearms in images using Faster R-CNN. The system includes a complete MLOps pipeline from automated dataset ingestion to model deployment as a real-time inference API.

---

# Project Architecture

Kaggle Dataset
      ↓
Data Ingestion
      ↓
Image & Annotation Processing
      ↓
PyTorch Dataset Loader
      ↓
Model Training (Faster R-CNN)
      ↓
Experiment Tracking (TensorBoard)
      ↓
Model Serialization
      ↓
FastAPI Inference Service
      ↓
Real-Time Object Detection API

---

# Dataset

Dataset Source:
https://www.kaggle.com/datasets/issaisasank/guns-object-detection

The dataset contains:

Images of guns
Bounding box annotations
Label files describing object coordinates

Each label file contains:

Number of objects
Bounding box coordinates

Example:

1
x_min y_min x_max y_max

---

# Data Ingestion

The system automatically downloads the dataset from Kaggle using KaggleHub.

Steps:

1 Download dataset
2 Extract images and labels
3 Store them in artifacts/raw directory

This enables automated dataset retrieval for reproducible pipelines.

---

# Data Processing

The custom PyTorch Dataset class performs:

Image loading using OpenCV
Image normalization
Conversion to PyTorch tensors
Bounding box extraction from label files

Each sample returns:

Image tensor
Bounding box coordinates
Object labels

---

# Model Architecture

The system uses Faster R-CNN with ResNet50 backbone.

Architecture Components:

Backbone Network (ResNet50)
Region Proposal Network
RoI Pooling
Bounding Box Regression
Object Classification

This architecture is widely used for object detection tasks.

---

# Training Pipeline

Training pipeline includes:

Dataset loading
Train validation split
Batch loading using DataLoader
Loss computation
Backpropagation
Model checkpoint saving

TensorBoard is used to track:

Training loss
Training progress
Experiment runs

---

# Model Deployment

The trained model is deployed using FastAPI.

API Endpoints:

GET /

Health check endpoint.

POST /predict/

Accepts image input and returns image with detected bounding boxes.

Example Request:

Upload image file.

Example Response:

Image with bounding boxes drawn around detected guns.

---

# DVC Pipeline

The project uses DVC to automate pipeline stages.

Stages:

data_ingestion
model_training

This ensures reproducible ML pipelines.

---

# Tech Stack

Programming

Python

Deep Learning

PyTorch
Torchvision

Computer Vision

OpenCV
Pillow

API

FastAPI
Uvicorn

Experiment Tracking

TensorBoard

MLOps

DVC
KaggleHub

---

# Key Features

Deep learning object detection
Automated dataset ingestion
Bounding box detection
Experiment tracking
API-based model inference
Reproducible ML pipelines

---

# Author

Hariom Birla  
M.Tech – Computer Science  
IIT Jodhpur
