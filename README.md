# Advanced Face Recognition Project

## Description
This project is an advanced face recognition application built with Python and Streamlit. It provides a user-friendly web interface to upload a query image and find all matching individuals within a local photo gallery.

The application is optimized for performance using an embedding cache and is robust enough to handle images containing multiple people, drawing bounding boxes around each detected match.

## Key Features
-   **Multiple Face Detection**: Utilizes MTCNN to accurately detect all faces within an image, not just the first one.
-   **High-Performance Caching**: Implements an intelligent caching system (`embeddings.pkl`) that saves face embeddings. The app only processes new or modified images, making startup and subsequent searches significantly faster.
-   **Accurate Face Matching**: Generates face embeddings using the DeepFace library (with the Facenet model) and calculates cosine similarity to find matches.
-   **Interactive Web UI**: A clean and simple interface built with Streamlit allows for easy image uploads, threshold adjustments, and clear presentation of results.
-   **Rich Visual Feedback**: For each match found, the application draws a bounding box around the person's face and displays the similarity score directly on the image.

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
-   Pickle (for caching)

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
1.  **Populate your gallery**: Place your dataset of images (the photos you want to search within) into the `images/` directory.
2.  **Run the Streamlit application:**
    ```bash
    streamlit run main.py
    ```
3.  **First Run & Caching**: The first time you run the app, it will process all images in the `images` folder and create an `embeddings.pkl` cache file. This might take a moment. Subsequent runs will be much faster.
4.  **Open the web interface**: Your browser should open a new tab with the application. If not, navigate to the local URL provided in your terminal (usually `http://localhost:8501`).
5.  **Upload an image**: Use the file uploader to select a query image.
6.  **Search for faces**: Click the "Search for Similar Faces" button.
7.  **View results**: The application will display any images containing a match, with a green box drawn around the matched faces and their similarity scores.

## Screenshots

### Query Image
*This is the image uploaded by the user to find matches.*

![Query Image](path/to/your/query_image.png)

### Matching Results
*Here are the results showing the matched faces in the gallery, with bounding boxes and similarity scores.*

![Match 1](path/to/your/match_1.png)
![Match 2](path/to/your/match_2.png)

## Directory Structure
```
.
├── .gitignore        # Specifies intentionally untracked files
├── images/           # Directory to store your dataset of images
│   └── ... (place your images here)
├── main.py           # The main script with the Streamlit application
├── requirements.txt  # A list of Python dependencies
├── embeddings.pkl    # Cache file for face embeddings (auto-generated)
└── README.md         # This file
```

## Contributing
Contributions are welcome! If you have suggestions for improvements or find any issues, please feel free to open an issue or submit a pull request.
