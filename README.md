# Face Recognition Photo Gallery

## Description

This project is an advanced, local photo management application built with Python and Flask. It provides a modern web interface and a RESTful API to organize and search your photo collection in powerful ways:

1.  **Face Recognition Search**: Upload a query image to find all matching individuals within your photo gallery.
2.  **Manual Face Tagging**: Click on detected faces in your photos to assign names to them.
3.  **Search by Name**: Easily find all photos containing a specific person by typing their name.
4.  **Automatic Albums**: View photos grouped by date and location.

The application uses a robust backend to handle image processing and a dynamic frontend for a seamless user experience. An alternative Streamlit-based UI is also available for more advanced visualization.

## Key Features

- **Intelligent Face Search**:
    - Utilizes **MTCNN** to accurately detect all faces within an image.
    - Generates face embeddings using the **DeepFace** library (with the Facenet model).
    - Finds matches based on cosine similarity.
- **Automatic Face Tagging**: When new photos are indexed, the system automatically recognizes known individuals and applies the correct name tags.
- **Interactive Face Tagging**:
    - Click on any photo to open a modal view with detected faces highlighted.
    - Assign names to faces, which are stored in the database for future searches.
- **RESTful API**: A well-defined backend API serves photo, face, and person data, allowing for easy integration or alternative frontends.
- **Automatic Clustering**: The backend uses **DBSCAN** to automatically group photos by date and location based on EXIF data. These albums are exposed via the API.
- **Location Awareness**: Uses the **Google Maps API** to convert GPS coordinates into human-readable place names.

## Technologies Used

- **Backend**:
    - Python 3.x
    - **Flask** for the web server and REST API
    - **DeepFace, MTCNN, OpenCV** for face processing
    - **Pillow** for image handling and EXIF extraction
    - **SQLite** for the database
- **Frontend**:
    - HTML5, CSS3, Vanilla JavaScript (ES6+)
- **Alternative UI**:
    - **Streamlit** provides a second, data-centric interface.

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
4.  **Set up your Google Maps API Key (Optional):**
    - Create a file named `.env` in the project root.
    - Add your Google Maps API key to this file for location name resolution.
      ```
      GOOGLE_MAPS_API_KEY="YOUR_API_KEY_HERE"
      ```

## Usage

### Running the Flask Application (Primary)

This is the main web interface for the application.

1.  **Populate your gallery**: Place your images into the `images/` directory.
2.  **Run the Flask application:**
    ```bash
    python backend/app.py
    ```
3.  **Open the application**: Navigate to `http://127.0.0.1:5001` in your web browser.
4.  **Index your gallery**: The application automatically indexes photos on first run. To add new photos, upload them via the UI or restart the server.

### Running the Alternative Streamlit UI

For a more data-focused view and direct access to management tasks:

1.  **Run the Streamlit application:**
    ```bash
    streamlit run streamlit_app.py
    ```
2.  **Open the application**: Navigate to the URL provided by Streamlit (e.g., `http://localhost:8501`).

## Project Structure

```
.
├── backend/
│   └── app.py              # Main Flask application and REST API
├── frontend/               # HTML/CSS/JS for the Flask app
├── images/                 # Default directory for your photo gallery
├── .env                    # Stores your Google Maps API key (private)
├── gallery.db              # SQLite database file (created on run)
├── photo_manager.py        # Core logic for photo processing, DB, and face recognition
├── requirements.txt        # Python dependencies
├── streamlit_app.py        # Alternative Streamlit UI
└── README.md               # This file
```

## API Endpoints

The Flask application exposes the following RESTful endpoints:

- `GET /api/photos`: Get a paginated list of all photos.
- `GET /api/photos/by_location`: Get photos grouped into albums by location.
- `GET /api/photos/by_date`: Get photos grouped into albums by date.
- `POST /api/upload`: Upload new images to the gallery.
- `POST /api/search_by_face`: Search for photos matching an uploaded face image.
- `GET /api/persons`: Get a list of all unique person names.
- `GET /api/search_by_name?name=<name>`: Get all photos containing the specified person.
- `GET /api/photo/<id>/faces`: Get all detected faces for a specific photo.
- `POST /api/tag_face`: Assign a name to a specific face.
- `DELETE /api/photo/<id>`: Delete a photo from the gallery.

