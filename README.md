# Face Recognition Photo Gallery

## Description

This project is an advanced, local photo management application built with Python and Flask. It provides a modern web interface to organize and search your photo collection in powerful ways:

1.  **Face Recognition Search**: Upload a query image to find all matching individuals within your photo gallery.
2.  **Manual Face Tagging**: Click on detected faces in your photos to assign names to them.
3.  **Search by Name**: Easily find all photos containing a specific person by typing their name.

The application uses a robust backend to handle image processing and a dynamic frontend for a seamless user experience.

## Key Features

- **Intelligent Face Search**:
    - Utilizes MTCNN to accurately detect all faces within an image.
    - Generates face embeddings using the DeepFace library (with the Facenet model).
    - Finds matches based on cosine similarity.
- **Interactive Face Tagging**:
    - Click on any photo to open a modal view.
    - Bounding boxes are drawn around all detected faces.
    - Click on an untagged face to enter and save a name, linking that face to a person.
- **Search by Name**:
    - A simple search bar allows you to find all photos associated with a tagged person.
    - Autocompletes names from the list of people in your database.
- **Modern Web UI**: A clean and responsive interface built with Flask, HTML, CSS, and vanilla JavaScript.
- **RESTful API**: A well-defined backend API serves photo, face, and person data to the frontend.

## Technologies Used

- **Backend**:
    - Python 3.x
    - Flask
    - DeepFace, MTCNN, OpenCV for face processing
    - Pillow for image handling
    - SQLite for the database
- **Frontend**:
    - HTML5
    - CSS3
    - Vanilla JavaScript (ES6+)
- **Other**:
    - Google Maps API (`googlemaps`) for location data extraction
    - Dotenv (`python-dotenv`) for environment variables

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/SatvikR123/Face_Recognizer.git
    cd Face_Recognizer
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
    - Add your Google Maps API key to this file. This is used to extract location data from photos.
      ```
      GOOGLE_MAPS_API_KEY="YOUR_API_KEY_HERE"
      ```
    - **Note**: The `.env` file is included in `.gitignore` to keep your API key private.

## Usage

1.  **Populate your gallery**: Place your images into the `frontend/static/images/` directory. The application will look for photos here.
2.  **Run the Flask application:**
    ```bash
    python backend/app.py
    ```
3.  **Open the application**: Navigate to `http://127.0.0.1:5001` in your web browser.
4.  **Index your gallery**:
    - The application automatically indexes your photos on the first run. If you add new photos later, you may need to restart the server or implement a re-indexing trigger.
5.  **Tag Faces**:
    - Click on any photo in the gallery to open it in a larger view.
    - If faces are detected, blue boxes will appear around them.
    - Click on a box to type a name and press "Save".
6.  **Search**:
    - **By Name**: Type a name in the top search bar and click "Search".
    - **By Face**: Click the "Search by Face" button and upload an image of the person you want to find.

## Project Structure

```
.
├── backend/
│   └── app.py              # Flask application, API endpoints
├── frontend/
│   ├── static/
│   │   ├── css/
│   │   │   └── style.css   # Stylesheets
│   │   ├── js/
│   │   │   └── script.js   # Frontend JavaScript logic
│   │   └── images/         # Default directory for user photos
│   └── templates/
│       └── index.html      # Main HTML page
├── .env                    # Stores your Google Maps API key (ignored by Git)
├── .gitignore              # Specifies files and directories to be ignored by Git
├── gallery.db              # SQLite database file (created on run)
├── photo_manager.py        # Core logic for face recognition and photo management
├── requirements.txt        # A list of Python dependencies
└── README.md               # This file
```

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue if you have any suggestions or find any bugs.

