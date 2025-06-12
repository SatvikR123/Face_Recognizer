import cv2
from mtcnn import MTCNN
from deepface import DeepFace
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
os.environ['QT_QPA_PLATFORM'] = 'xcb'
os.environ['DISPLAY'] = ':0'

# Set matplotlib backend - try different ones if needed
import matplotlib
matplotlib.use('Qt5Agg')  # Alternatives: 'Qt5Agg', 'GTK3Agg', 'WXAgg' 


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

    def display_results(self, query_image_path, matches):
        """
        Display the query image and its matches in a beautiful grid layout
        """
        if not matches:
            print("No matches found!")
            return
            
        # Create a figure with subplots
        plt.close('all')  # Close any existing figures
        num_matches = len(matches)
        fig = plt.figure(figsize=(20, 10))
        
        # Add query image at the top
        try:
            ax = fig.add_subplot(2, 1, 1)
            query_img = Image.open(query_image_path)
            ax.imshow(query_img)
            ax.set_title('Query Image', fontsize=16, pad=15)
            ax.axis('off')
        except Exception as e:
            print(f"Error loading query image: {e}")
            return
            
        # Add matching images in a grid below
        grid_size = int(np.ceil(np.sqrt(num_matches)))
        
        for i, (match_path, similarity) in enumerate(matches):
            try:
                ax = fig.add_subplot(grid_size, grid_size, i + 1)
                match_img = Image.open(match_path)
                ax.imshow(match_img)
                
                # Create a more informative title
                filename = os.path.basename(match_path)
                title = f"{filename}\nSimilarity: {similarity:.3f}"
                ax.set_title(title, fontsize=12, pad=10)
                
                # Add a border around the image
                ax.patch.set_edgecolor('black')
                ax.patch.set_linewidth(2)
                
                # Add a background color for the title
                title = ax.set_title(title, pad=10)
                title.set_bbox(dict(facecolor='white', alpha=0.7, edgecolor='black'))
                
                ax.axis('off')
            except Exception as e:
                print(f"Error loading match image {match_path}: {e}")
                continue
        
        # Adjust layout for better spacing
        plt.tight_layout()
        plt.subplots_adjust(top=0.95)
        plt.show(block=True)  # This will keep the window open

if __name__ == "__main__":
    # Initialize face recognition with your images directory
    face_recognizer = FaceRecognition("images")
    
    # Load the dataset
    face_recognizer.load_dataset()
    
    # Get query image path from user
    query_image_path = 'image.jpeg'
    
    # Search for similar faces
    matches = face_recognizer.search_similar_faces(query_image_path)
    
    if not matches:
        print("No similar faces found!")
    else:
        print(f"\nFound {len(matches)} matches:")
        for i, (path, score) in enumerate(matches):
            print(f"{i+1}. {os.path.basename(path)}: Similarity = {score:.3f}")
        
        # Show visualization with all matches
        face_recognizer.display_results(query_image_path, matches)
        input("Press Enter to exit...")  # Keep the script running until user presses Enter