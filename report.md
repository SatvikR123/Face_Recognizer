# Project Report

## Preface
This report provides a comprehensive overview of the Face Recognition Photo Gallery project. This document details the project's scope, architecture, and implementation, serving as a guide for understanding its features and technical components.

## Scope of project
This project is an advanced, local photo management application built with Python and Flask. It provides a modern web interface and a RESTful API to organize and search your photo collection in powerful ways:

1.  **Face Recognition Search**: Upload a query image to find all matching individuals within your photo gallery.
2.  **Manual Face Tagging**: Click on detected faces in your photos to assign names to them.
3.  **Search by Name**: Easily find all photos containing a specific person by typing their name.
4.  **Automatic Albums**: View photos grouped by date and location.
5.  **Object Removal**: Remove unwanted objects from your photos.

## Acknowledgement
This project utilizes several open-source libraries and technologies. We acknowledge the creators and maintainers of the following key components: Flask, DeepFace, MTCNN, OpenCV, Pillow, SQLite, PyTorch/TorchVision, and DeepFill.

## Keywords and definitions
*   **MTCNN**: Multi-task Cascaded Convolutional Networks, a framework for face detection and alignment.
*   **DeepFace**: A lightweight face recognition and facial attribute analysis library for Python.
*   **Facenet**: A face recognition system that learns a mapping from face images to a compact Euclidean space where distances directly correspond to a measure of face similarity.
*   **DBSCAN**: Density-Based Spatial Clustering of Applications with Noise, a data clustering algorithm.
*   **EXIF**: Exchangeable Image File Format, a standard that specifies the formats for images, sound, and ancillary tags used by digital cameras.
*   **Inpainting**: The process of reconstructing lost or deteriorated parts of images and videos.

## Introduction
The Face Recognition Photo Gallery is a sophisticated application designed to manage and organize large collections of digital photos. It leverages advanced machine learning models to provide features like face recognition, automatic tagging, and object removal, making it a powerful tool for personal photo management. The application is built with a Flask backend and a vanilla JavaScript frontend, providing a responsive and user-friendly interface.

## System flow
The system operates as follows:
1.  **Initialization**: The Flask application starts, initializing the `PhotoManager` and the object removal models.
2.  **Indexing**: The `PhotoManager` indexes the images in the `images/` directory. For each image, it extracts EXIF data, detects faces, and computes face embeddings.
3.  **Clustering**: After indexing, the `PhotoManager` clusters photos by time and location using the DBSCAN algorithm.
4.  **User Interaction**: The user interacts with the application through a web interface.
5.  **API Requests**: The frontend sends API requests to the Flask backend to perform actions like fetching photos, searching by face or name, tagging faces, and removing objects.
6.  **Backend Processing**: The backend processes these requests, interacting with the SQLite database and the machine learning models as needed.
7.  **Response**: The backend returns JSON data to the frontend, which then updates the user interface.

## Description of each class in the project
### `PhotoManager`
The `PhotoManager` class in `photo_manager.py` is the core of the application's backend logic. It handles:
*   **Database Initialization**: Creates and manages the SQLite database for storing photo metadata, face data, and person information.
*   **Image Indexing**: Scans the image gallery, extracts EXIF data, detects faces using MTCNN, and generates face embeddings using DeepFace.
*   **Clustering**: Groups photos by date and location using DBSCAN.
*   **Search**: Provides methods to search for photos by person name and by a query face image.
*   **Face Management**: Allows tagging faces with names and propagating those names to similar faces.

### `ObjectRemove`
The `ObjectRemove` class in `ObjectRemoval/src/objRemovalDrawing.py` handles the object removal functionality. It uses:
*   **Mask-RCNN**: For segmenting objects in an image.
*   **DeepFill**: An inpainting model to fill in the area where the object has been removed.

### `Generator`
The `Generator` class in `ObjectRemoval/src/models/deepFill.py` is the DeepFill inpainting model.

### `Flask App`
The `app.py` script contains the Flask application, which serves the frontend and exposes the RESTful API for all the application's functionalities.

## Screen shots of events
*(Please add relevant screenshots of the application in action here.)*

## List of software code
The main software components are:
*   `backend/app.py`: The main Flask application.
*   `photo_manager.py`: The core logic for photo processing, database management, and face recognition.
*   `ObjectRemoval/src/objRemovalDrawing.py`: The core logic for the object removal feature.
*   `ObjectRemoval/src/main.py`: A standalone script for testing the object removal feature.
*   `frontend/`: Contains the HTML, CSS, and JavaScript files for the web interface.

## Limitation of projects
*   The application runs locally and is not designed for multi-user access.
*   The accuracy of face recognition and object detection depends on the performance of the underlying models and the quality of the images.
*   The object removal feature requires manual selection of the object to be removed.
*   The application requires a Google Maps API key for location name resolution, which may have usage limits.

## Summary and conclusion
The Face Recognition Photo Gallery is a powerful and feature-rich application for managing personal photo collections. It successfully integrates advanced machine learning techniques to provide intelligent photo organization and editing capabilities. While there are some limitations, the project serves as an excellent demonstration of a modern, full-stack Python application.

## References
*   **Flask**: https://flask.palletsprojects.com/
*   **DeepFace**: https://github.com/serengil/deepface
*   **MTCNN**: https://github.com/ipazc/mtcnn
*   **OpenCV**: https://opencv.org/
*   **Pillow**: https://python-pillow.org/
*   **PyTorch**: https://pytorch.org/
*   **Mask R-CNN**: https://arxiv.org/abs/1703.06870
*   **DeepFill**: https://arxiv.org/abs/1806.03589
