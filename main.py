"""
An advanced face recognition application using Streamlit.

This script provides a web UI for uploading an image and finding matches
in a local gallery. It features multi-face detection, embedding caching for
performance, and visual feedback with bounding boxes on matched faces.
"""

import cv2
from mtcnn import MTCNN
from deepface import DeepFace
import numpy as np
import os
import streamlit as st
import pickle

class FaceRecognition:
    def __init__(self, directory, cache_path='embeddings.pkl'):
        self.detector = MTCNN()
        self.target_size = (160, 160)
        self.directory = directory
        self.model = DeepFace.build_model('Facenet')
        self.cache_path = cache_path
        self.data = self.load_dataset()

    def extract_faces(self, image):
        # Extracts all faces from an image using MTCNN.
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        detections = self.detector.detect_faces(image_rgb)
        faces = []
        for detection in detections:
            x, y, w, h = detection['box']
            face = image_rgb[y:y+h, x:x+w]
            face = cv2.resize(face, self.target_size)
            faces.append({'box': (x, y, w, h), 'face': face})
        return faces

    def get_embeddings(self, faces):
        # Generates face embeddings for a list of face images.
        embeddings = []
        for face_info in faces:
            face = face_info['face'].astype('float32')
            embedding_list = DeepFace.represent(face, model_name='Facenet', enforce_detection=False)
            if embedding_list:
                embeddings.append(np.array(embedding_list[0]['embedding']))
        return embeddings

    def load_from_cache(self):
        # Loads face embeddings from the cache file.
        if os.path.exists(self.cache_path):
            with open(self.cache_path, 'rb') as f:
                return pickle.load(f)
        return {}

    def save_to_cache(self, data):
        # Saves face embeddings to the cache file.
        with open(self.cache_path, 'wb') as f:
            pickle.dump(data, f)

    def load_dataset(self):
        # Loads image dataset, using cache to process only new or modified files.
        cached_data = self.load_from_cache()
        current_data = {}
        updated = False

        image_files = [f for f in os.listdir(self.directory) if f.endswith(('.jpg', '.jpeg', '.png'))]

        for filename in image_files:
            path = os.path.join(self.directory, filename)
            mod_time = os.path.getmtime(path)

            if path in cached_data and cached_data[path]['mod_time'] == mod_time:
                current_data[path] = cached_data[path]
            else:
                st.info(f"Processing new/modified file: {filename}")
                image = cv2.imread(path)
                if image is None:
                    continue
                
                faces = self.extract_faces(image)
                if not faces:
                    continue

                embeddings = self.get_embeddings(faces)
                
                face_data = []
                for i, face_info in enumerate(faces):
                    if i < len(embeddings):
                        face_data.append({
                            'box': face_info['box'],
                            'embedding': embeddings[i]
                        })

                if face_data:
                    current_data[path] = {'mod_time': mod_time, 'faces': face_data}
                    updated = True
        
        if updated:
            self.save_to_cache(current_data)
            st.success("Cache updated.")

        return current_data

    def check_similarity(self, emb1, emb2):
        # Calculates cosine similarity between two face embeddings.
        emb1 = emb1.flatten()
        emb2 = emb2.flatten()
        return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

    def search_similar_faces(self, query_image_path, threshold=0.5):
        # Searches the dataset for faces similar to the query image.
        query_image = cv2.imread(query_image_path)
        if query_image is None:
            st.error("Could not read query image.")
            return []
            
        query_faces = self.extract_faces(query_image)
        if not query_faces:
            st.warning("No face detected in the uploaded image.")
            return []
        
        query_embedding = self.get_embeddings([query_faces[0]])[0]
        
        all_matches = []

        for path, data in self.data.items():
            matched_faces_in_image = []
            for face_data in data['faces']:
                dataset_embedding = face_data['embedding']
                similarity = self.check_similarity(query_embedding, dataset_embedding)
                
                if similarity >= threshold:
                    matched_faces_in_image.append({
                        'box': face_data['box'],
                        'score': similarity
                    })
            
            if matched_faces_in_image:
                matched_faces_in_image.sort(key=lambda x: x['score'], reverse=True)
                all_matches.append((path, matched_faces_in_image))

        all_matches.sort(key=lambda x: x[1][0]['score'], reverse=True)
        return all_matches

def main():
    st.title("Advanced Face Recognition App")
    st.write("Upload an image to find matching faces in the gallery.")

    face_recognizer = FaceRecognition("images")
    
    st.success(f'Dataset loaded! Found {len(face_recognizer.data)} images in the gallery.')

    uploaded_file = st.file_uploader("Choose a query image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        query_image_path = os.path.join("uploaded_image.jpg")
        with open(query_image_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.image(query_image_path, caption='Query Image', use_container_width=True)

        threshold = st.slider("Similarity Threshold", 0.0, 1.0, 0.5, 0.01)

        if st.button("Search for Similar Faces"):
            with st.spinner('Searching for matches...'):
                matches = face_recognizer.search_similar_faces(query_image_path, threshold)
            
            if not matches:
                st.warning("No similar faces found in the gallery.")
            else:
                st.success(f"Found matches in {len(matches)} image(s).")
                
                for img_path, matched_faces in matches:
                    st.write(f"**Matches in: {os.path.basename(img_path)}**")
                    
                    display_image = cv2.imread(img_path)
                    display_image = cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB)

                    for face in matched_faces:
                        x, y, w, h = face['box']
                        score = face['score']
                        
                        cv2.rectangle(display_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        
                        label = f"{score:.2f}"
                        cv2.putText(display_image, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    
                    st.image(display_image, use_container_width=True)

if __name__ == "__main__":
    main()