# Face Recognition Project

## Description
This project is a Python application that performs face recognition. It can detect faces in images, generate embeddings for these faces, and then compare them to find similar faces in a given dataset of images(Folder/Gallery).

## Features
-   Detects faces in images using MTCNN.
-   Generates face embeddings using the DeepFace library with the Facenet model.
-   Calculates cosine similarity between face embeddings to find matches.
-   Searches a directory of images for faces similar to a query image.
-   Displays the query image and the matching images with their similarity scores.

## Technologies Used
-   Python 3.x
-   OpenCV (`opencv-python`)
-   MTCNN (`mtcnn`)
-   DeepFace (`deepface`)
-   NumPy (`numpy`)
-   Pillow (`pillow`)
-   TensorFlow (`tensorflow`)
-   Keras (`keras`)
-   Matplotlib (`matplotlib`) (Note: Matplotlib is in requirements.txt but not explicitly used in main.py, keeping it for now)

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
2.  **Set the query image:**
    *   Open the `main.py` file.
    *   Locate the line `query_image_path = 'sk_query.jpg'` within the `if __name__ == "__main__":` block.
    *   Change `'sk_query.jpg'` to the path of the image you want to use as the query. This can be an image inside or outside the `images/` directory. The default `sk_query.jpg` is provided as an example query image in the root directory.
3.  **Run the script:**
    ```bash
    python main.py
    ```
    The script will output the similarity scores for each image in the `images/` directory compared to the query image. It will then display the query image and any matching images that meet the similarity threshold.

## Directory Structure
```
.
├── .gitignore        # Specifies intentionally untracked files that Git should ignore
├── images/           # Directory to store your dataset of images
│   ├── ak.jpeg
│   ├── dp.jpeg
│   ├── ... (other images)
├── main.py           # The main script to run the face recognition
├── requirements.txt  # A list of Python dependencies for the project
├── sk_query.jpg      # An example query image
└── README.md         # This file
```

## Example Output
When you run `python main.py`, the console will first print a list of images from the `images/` directory and their similarity scores compared to the query image. For example:
```
Query image: sk_query.jpg
Scanning images and their similarity scores:
sj_2.jpeg: 0.285
hrx.jpg: 0.320
sk.jpg: 0.804
ak.jpeg: 0.168
dp.jpeg: 0.200
sj.jpg: 0.252
sk_2.jpeg: 0.750
sk_4.jpg: 0.701
```
Then, if matches are found above the similarity threshold (default 0.5), it will display the query image in a window titled "Query Image". Subsequently, it will open separate windows for each matching image, titled "Similar image - [filename] (Similarity: [score])".

For example, if `sk.jpg` was a match for `sk_query.jpg` with a similarity of 0.804, a window would appear showing the `sk.jpg` image, with the window title "Similar image - sk.jpg (Similarity: 0.804)".

Press any key to close each image window. After all matching images are shown, you'll need to press Enter in the console to exit the script.

## Contributing
Contributions are welcome! If you have suggestions for improvements or find any issues, please feel free to open an issue or submit a pull request.

## License
This project is currently unlicensed. You can add a license file (e.g., MIT, Apache 2.0) if you wish to share it under specific terms.
