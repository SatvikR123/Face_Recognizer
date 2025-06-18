# Local Photo Manager & Face Recognition

## Description

This project is an advanced, local photo management application built with Python and Streamlit. It provides a user-friendly web interface to organize and search your photo collection in two powerful ways:

1.  **Face Recognition**: Upload a query image to find all matching individuals within your photo gallery.
2.  **Location Clustering**: Automatically groups your photos into albums based on where they were taken, using GPS data from image EXIF tags and the Google Maps API.

The application is optimized for performance using an embedding cache and is robust enough to handle images containing multiple people.

## Key Features

- **Intelligent Face Search**:
  - Utilizes MTCNN to accurately detect all faces within an image.
  - Generates face embeddings using the DeepFace library (with the Facenet model).
  - Finds matches based on cosine similarity with a user-adjustable threshold.
- **Automatic Location Clustering**:
  - Extracts GPS coordinates from image EXIF data.
  - Uses the Google Maps API to get rich, human-readable location names (e.g., "Paschim Vihar, Delhi").
  - Groups photos into albums based on their location, making it easy to browse memories from specific places.
- **High-Performance Caching**: Implements an intelligent caching system for face embeddings. The app only processes new or modified images, making subsequent searches significantly faster.
- **Interactive Web UI**: A clean and simple interface built with Streamlit allows for easy image uploads, gallery management, and clear presentation of results.
- **Rich Visual Feedback**: For each face match found, the application draws a bounding box around the person's face and displays the similarity score.

## Technologies Used

- Python 3.x
- Streamlit
- Google Maps API (`googlemaps`)
- Dotenv (`python-dotenv`)
- OpenCV (`opencv-python`)
- MTCNN (`mtcnn`)
- DeepFace (`deepface`)
- NumPy (`numpy`)
- Pillow (`pillow`)

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
4.  **Set up your Google Maps API Key:**
    - Create a file named `.env` in the project root.
    - Add your Google Maps API key to this file:
      ```
      GOOGLE_MAPS_API_KEY="YOUR_API_KEY_HERE"
      ```
    - **Note**: The `.env` file is included in `.gitignore` to keep your API key private.

## Usage

1.  **Populate your gallery**: Place your dataset of images (the photos you want to search within) into the `images/` directory.
2.  **Run the Streamlit application:**
    ```bash
    streamlit run main.py
    ```
3.  **Index your gallery**:
    - The application will automatically index your photos on first run.
    - You can click **"Force Re-index"** in the sidebar at any time to re-process all images. This is necessary after adding new photos or to apply new features like location naming.
4.  **Search by Face**:
    - Upload a photo of a person in the sidebar.
    - Adjust the similarity threshold if needed.
    - The application will display all matching photos from your gallery.
5.  **Browse by Location**:
    - Click the **"Cluster Photos"** button in the sidebar.
    - The application will generate albums based on location.
    - Scroll down to see your photos neatly organized by where they were taken.

## Project Structure

```
.
├── images/             # Directory to store your photo gallery (ignored by Git)
├── .env                # Stores your Google Maps API key (ignored by Git)
├── .gitignore          # Specifies files and directories to be ignored by Git
├── main.py             # Main Streamlit application file
├── photo_manager.py    # Core logic for face recognition and photo management
├── requirements.txt    # A list of Python dependencies
└── README.md           # This file

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue if you have any suggestions or find any bugs.
```
