"""
Face Recognition App

This is a face recognition application that uses MTCNN for face detection and DeepFace for face embedding. It allows users to upload an image and find similar faces from a local image dataset (a folder/gallery).

Features:
- Face detection using MTCNN
- Face embedding using DeepFace with the Facenet model
- Cosine similarity calculation between face embeddings
- Search for similar faces in a local image dataset
- Interactive web interface built with Streamlit

Technologies Used:
- Python 3.x
- Streamlit
- OpenCV (`opencv-python`)
- MTCNN (`mtcnn`)
- DeepFace (`deepface`)
- NumPy (`numpy`)
- TensorFlow (`tensorflow`)
- Keras (`keras`)

"""

import cv2
from mtcnn import MTCNN
from deepface import DeepFace
import numpy as np
import os
import streamlit as st

class FaceRecognition:
    def __init__(self, directory):
        self.detector = MTCNN()
        self.target_size = (160, 160)
        self.X = []
        self.y = []
        self.directory = directory
        self.model = DeepFace.build_model('Facenet')

    def extract_face(self, image):
        
        #Extracts a face from the given image using MTCNN.
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        faces = self.detector.detect_faces(image)
        if faces:
            x, y, w, h = faces[0]['box']
            face = image[y:y+h, x:x+w]
            face = cv2.resize(face, self.target_size)
            return face
        return None                

    def get_embeddings(self, face):
        
        #Generates face embeddings using the DeepFace library with the Facenet model.            
        
        face = face.astype('float32')
        embeddings = DeepFace.represent(face, model_name='Facenet', enforce_detection=False)
        if embeddings:
            return np.array(embeddings[0]['embedding']).reshape(1, -1)
        return None
        
    def load_dataset(self):
        
        #Loads the dataset of images from the specified directory.
        
        for filename in os.listdir(self.directory):
            if filename.endswith(('.jpg', '.jpeg', '.png')):  # Check for common image extensions
                path = os.path.join(self.directory, filename)
                image = cv2.imread(path)
                face = self.extract_face(image)
                if face is not None:
                    embedding = self.get_embeddings(face)
                    self.X.append(embedding)
                    # Use filename as label
                    self.y.append(filename)

    def check_similarity(self, face1, face2):
        
        #Calculates the cosine similarity between two face embeddings.
        
        face1 = face1.flatten()
        face2 = face2.flatten()
        
        cosine_similarity = np.dot(face1, face2) / (np.linalg.norm(face1) * np.linalg.norm(face2))
        return cosine_similarity

    def search_similar_faces(self, query_image_path, threshold=0.5):
        
        #Searches for similar faces in the dataset based on the query image.
        
        query_image = cv2.imread(query_image_path)
        query_face = self.extract_face(query_image)
        
        if query_face is None:
            print("No face detected in query image!")
            return []
            
        query_embedding = self.get_embeddings(query_face)
        matches = []
        
        print(f"\nQuery image: {query_image_path}")
        print("Scanning images and their similarity scores:")
        

        for filename in os.listdir(self.directory):
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                path = os.path.join(self.directory, filename)
                image = cv2.imread(path)
                face = self.extract_face(image)
                
                if face is None:
                    print(f"No face detected in {filename}")
                    continue
                    
                embedding = self.get_embeddings(face)
                if embedding is None:
                    print(f"Failed to get embedding for {filename}")
                    continue
                    
                similarity = self.check_similarity(query_embedding, embedding)
                print(f"{filename}: {similarity:.3f}")
                if similarity >= threshold:
                    matches.append((path, similarity))  
        
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches

def main():
            
    #Main function to run the face recognition application.
    
    st.title("Face Recognition App")

    face_recognizer = FaceRecognition("images")
    
    with st.spinner('Loading dataset...'):
        face_recognizer.load_dataset()
    st.success('Dataset loaded successfully!')

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        query_image_path = os.path.join("uploaded_image.jpg")
        with open(query_image_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.image(query_image_path, caption='Query Image', use_container_width=True)

        threshold = st.slider("Similarity Threshold", 0.0, 1.0, 0.5, 0.01)

        if st.button("Search for similar faces"):
            with st.spinner('Searching...'):
                matches = face_recognizer.search_similar_faces(query_image_path, threshold)
            
            if not matches:
                st.write("No similar faces found!")
            else:
                st.write(f"Found {len(matches)} matches:")
                
                for i, (path, score) in enumerate(matches):
                    st.write(f"{i+1}. {os.path.basename(path)}: Similarity = {score:.3f}")
                    st.image(path, caption=f'Match {i+1}', use_container_width=True)

if __name__ == "__main__":
    main()