import cv2
from mtcnn import MTCNN
from deepface import DeepFace
import numpy as np
from PIL import Image
import os

class FaceRecognition:
    def __init__(self, directory):
        self.detector = MTCNN()
        self.target_size = (160, 160)
        self.X = []
        self.y = []
        self.directory = directory
        self.model = DeepFace.build_model('Facenet')

    def extract_face(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        faces = self.detector.detect_faces(image)
        if faces:
            x, y, w, h = faces[0]['box']
            face = image[y:y+h, x:x+w]
            face = cv2.resize(face, self.target_size)
            return face
        return None                

    def get_embeddings(self, face):
        face = face.astype('float32')
        embeddings = DeepFace.represent(face, model_name='Facenet', enforce_detection=False)
        if embeddings:
            return np.array(embeddings[0]['embedding']).reshape(1, -1)
        return None
        
    def load_dataset(self):
        # Get all image files in the directory
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
        """
        Calculate cosine similarity between two face embeddings
        Args:
            face1: First embedding array (1,128)
            face2: Second embedding array (1,128)
        Returns:
            Cosine similarity score (float)
        """
        # Flatten the embeddings to 1D arrays
        face1 = face1.flatten()
        face2 = face2.flatten()
        
        # Calculate cosine similarity
        cosine_similarity = np.dot(face1, face2) / (np.linalg.norm(face1) * np.linalg.norm(face2))
        return cosine_similarity

    def search_similar_faces(self, query_image_path, threshold=0.5):  
        """
        Search for images in the dataset that contain similar faces to the query image
        Args:
            query_image_path: Path to the query image
            threshold: Minimum similarity score to consider a match
        Returns:
            List of tuples containing (image_path, similarity_score)
        """
        query_image = cv2.imread(query_image_path)
        query_face = self.extract_face(query_image)
        
        if query_face is None:
            print("No face detected in query image!")
            return []
            
        query_embedding = self.get_embeddings(query_face)
        matches = []
        
        print(f"\nQuery image: {query_image_path}")
        print("Scanning images and their similarity scores:")
        
        # Get all image files in the directory
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
        
        # Sort matches by similarity (highest first)
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches

    def display_results(self, query_image_path, matches, threshold):
        """
        Display the query image and its matches in a grid layout
        Args:
            query_image_path: Path to the query image
            matches: List of tuples containing (image_path, similarity_score)
            threshold: Minimum similarity score used for filtering
        """
        if not matches:
            print("No matches found!")
            return
            
        try:
            # Load query image
            query_img = cv2.imread(query_image_path)
            if query_img is None:
                raise ValueError(f"Could not load query image: {query_image_path}")
            
            # Get image dimensions
            height, width = query_img.shape[:2]
            
            # Create a grid layout
            num_matches = len(matches)
            grid_size = int(np.ceil(np.sqrt(num_matches)))
            
            # Calculate window positions
            window_width = width // grid_size
            window_height = height // grid_size
            
            # Display query image
            cv2.imshow('Query Image', query_img)
            
            # Display matching images in a grid
            for i, (match_path, similarity) in enumerate(matches):
                try:
                    image = cv2.imread(match_path)
                    if image is None:
                        continue
                        
                    # Resize image to fit grid
                    image = cv2.resize(image, (window_width, window_height))
                    
                    # Calculate window position
                    row = i // grid_size
                    col = i % grid_size
                    x = col * window_width
                    y = row * window_height
                    
                    # Create window name
                    filename = os.path.basename(match_path)
                    window_name = f"Similar image - {filename} (Similarity: {similarity:.3f})"
                    
                    # Create window and set position
                    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(window_name, window_width, window_height)
                    cv2.moveWindow(window_name, x, y)
                    cv2.imshow(window_name, image)
                    
                except Exception as e:
                    print(f"Error displaying match image {match_path}: {e}")
                    continue
            
            # Print summary
            print("\nSearch complete!")
            print(f"Found {len(matches)} similar faces with similarity >= {threshold}")
            print("\nPress any key to continue...")
            cv2.waitKey(0)  # Wait for key press
            
        except Exception as e:
            print(f"Error in display_results: {e}")
        finally:
            cv2.destroyAllWindows()

    def show_matches(self, query_image_path, matches):
        if not matches:
            print("No matches found!")
            return
            
        # Display query image
        try:
            query_img = cv2.imread(query_image_path)
            cv2.imshow('Query Image', query_img)
            cv2.waitKey(1000)  # Show for 1 second
            cv2.destroyAllWindows()
        except Exception as e:
            print(f"Error loading query image: {e}")
            return
            
        # Display matching images one by one
        for i, (match_path, similarity) in enumerate(matches):
            try:
                image = cv2.imread(match_path)
                filename = os.path.basename(match_path)
                print(f"Similarity score: {similarity:.4f}")
                cv2.imshow(f"Similar image - {filename}", image)
                cv2.waitKey(0)  # Wait for key press
                cv2.destroyAllWindows()
            except Exception as e:
                print(f"Error displaying match image {match_path}: {e}")
                continue
        
        print("\nSearch complete!")
        print(f"Found {len(matches)} similar faces with similarity >= {threshold}")

if __name__ == "__main__":
    # Initialize face recognition with your images directory
    face_recognizer = FaceRecognition("images")
    
    # Load the dataset
    face_recognizer.load_dataset()
    
    # Get query image path from user
    query_image_path = 'sk_query.jpg'
    threshold = 0.5
    
    # Search for similar faces
    matches = face_recognizer.search_similar_faces(query_image_path, threshold)
    
    if not matches:
        print("No similar faces found!")
    else:
        print(f"\nFound {len(matches)} matches:")
        for i, (path, score) in enumerate(matches):
            print(f"{i+1}. {os.path.basename(path)}: Similarity = {score:.3f}")
        
        # Show visualization with all matches
        face_recognizer.display_results(query_image_path, matches, threshold)
        input("Press Enter to exit...")  # Keep the script running until user presses Enter 