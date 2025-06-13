# Face Recognition Project

## Description
This project is a Python application that performs face recognition using a Streamlit web interface. It allows users to upload an image and find similar faces from a local image dataset (a folder/gallery).

## Features
-   Detects faces in images using MTCNN.
-   Generates face embeddings using the DeepFace library with the Facenet model.
-   Calculates cosine similarity between face embeddings to find matches.
-   Searches a directory of images for faces similar to a user-uploaded query image.
-   Provides an interactive web interface built with Streamlit to upload images and view results.

## Technologies Used
-   Python 3.x
-   Streamlit
-   OpenCV (`opencv-python`)
-   MTCNN (`mtcnn`)
-   DeepFace (`deepface`)
-   NumPy (`numpy`)
-   Pillow (`pillow`)
-   TensorFlow (`tensorflow`)
-   Keras (`keras`)

## Setup and Installation
1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage
1.  **Place your dataset images** (the images you want to search within) into the `images/` directory.
2.  **Run the Streamlit application:**
    ```bash
    streamlit run main.py
    ```
3.  **Open the web interface:** Your browser should open a new tab with the application running. If not, navigate to the local URL provided in your terminal (usually `http://localhost:8501`).
4.  **Upload an image:** Use the file uploader to select a query image from your computer.
5.  **Adjust the threshold:** Use the slider to set the desired similarity threshold.
6.  **Search for faces:** Click the "Search for similar faces" button to start the search.
7.  **View results:** The application will display the matching images from your `images` directory along with their similarity scores.

## Directory Structure
```
.
├── .gitignore        # Specifies intentionally untracked files
├── images/           # Directory to store your dataset of images
│   └── ... (place your images here)
├── main.py           # The main script with the Streamlit application
├── requirements.txt  # A list of Python dependencies
└── README.md         # This file
```

## Contributing
Contributions are welcome! If you have suggestions for improvements or find any issues, please feel free to open an issue or submit a pull request.
