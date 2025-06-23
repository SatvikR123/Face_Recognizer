import sqlite3
import os
import json
from datetime import datetime
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
import googlemaps
from dotenv import load_dotenv
import numpy as np
import cv2
from mtcnn import MTCNN
from deepface import DeepFace

class PhotoManager:
    def __init__(self, db_path='gallery.db', gallery_path='images'):
        self.db_path = db_path
        self.gallery_path = gallery_path
        self.detector = MTCNN()
        self.face_model = DeepFace.build_model('Facenet')

        # Load environment variables and initialize Google Maps client
        load_dotenv()
        api_key = os.getenv("GOOGLE_MAPS_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_MAPS_API_KEY not found in .env file")
        self.gmaps = googlemaps.Client(key=api_key)

        self._initialize_db()

    def _initialize_db(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS photos (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL UNIQUE,
                filepath TEXT NOT NULL,
                capture_date DATETIME,
                latitude REAL,
                longitude REAL,
                location_name TEXT,
                mod_time REAL NOT NULL,
                time_cluster_id INTEGER,
                location_cluster_id INTEGER
            )
            ''')
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS persons (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE
            )
            ''')
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='faces'")
            if cursor.fetchone() is None:
                cursor.execute('''
                CREATE TABLE faces (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    photo_id INTEGER,
                    person_id INTEGER,
                    box_x INTEGER, box_y INTEGER, box_w INTEGER, box_h INTEGER,
                    embedding BLOB,
                    FOREIGN KEY(photo_id) REFERENCES photos(id) ON DELETE CASCADE,
                    FOREIGN KEY(person_id) REFERENCES persons(id) ON DELETE SET NULL
                )
                ''')
            else:
                cursor.execute("PRAGMA table_info(faces)")
                columns = [info[1] for info in cursor.fetchall()]
                if 'person_id' not in columns:
                    cursor.execute("ALTER TABLE faces ADD COLUMN person_id INTEGER REFERENCES persons(id) ON DELETE SET NULL")
            conn.commit()

    def _dms_to_decimal(self, dms, ref):
        try:
            degrees = float(dms[0])
            minutes = float(dms[1]) / 60.0
            seconds = float(dms[2]) / 3600.0
            dd = degrees + minutes + seconds
            if ref in ['S', 'W']:
                dd *= -1
            return dd
        except (TypeError, IndexError, ValueError):
            return None

    def _get_exif_data(self, image_path):
        try:
            image = Image.open(image_path)
            info = image._getexif()
            if not info:
                return {}
            
            exif_data = {}
            for tag, value in info.items():
                decoded = TAGS.get(tag, tag)
                if decoded == "GPSInfo":
                    if isinstance(value, dict):
                        gps_data = {}
                        for t, v in value.items():
                            sub_decoded = GPSTAGS.get(t, t)
                            gps_data[sub_decoded] = v
                        exif_data[decoded] = gps_data
                else:
                    exif_data[decoded] = value
            return exif_data
        except Exception:
            return {}

    def _get_location_from_gps(self, exif_data):
        if not exif_data or 'GPSInfo' not in exif_data:
            return None, None, None
        
        gps_info = exif_data['GPSInfo']
        if not isinstance(gps_info, dict):
            return None, None, None

        lat_dms = gps_info.get('GPSLatitude')
        lat_ref = gps_info.get('GPSLatitudeRef')
        lon_dms = gps_info.get('GPSLongitude')
        lon_ref = gps_info.get('GPSLongitudeRef')

        if not all([lat_dms, lat_ref, lon_dms, lon_ref]):
            return None, None, None

        latitude = self._dms_to_decimal(lat_dms, lat_ref)
        longitude = self._dms_to_decimal(lon_dms, lon_ref)

        if latitude is None or longitude is None:
            return None, None, None

        try:
            reverse_geocode_result = self.gmaps.reverse_geocode((latitude, longitude))
            if reverse_geocode_result:
                location_name = self._extract_friendly_location_name(reverse_geocode_result)
                return latitude, longitude, location_name
        except Exception as e:
            print(f"Google Maps reverse geocoding failed for ({latitude}, {longitude}): {e}")
            
        return latitude, longitude, None

    def _extract_friendly_location_name(self, geocode_results):
        if not geocode_results:
            return None

        best_components = {
            'neighborhood': None,
            'sublocality_level_1': None,
            'locality': None,
            'administrative_area_level_1': None,
            'country': None
        }

        for result in geocode_results:
            for component in result.get('address_components', []):
                for comp_type in best_components.keys():
                    if comp_type in component.get('types', []) and not best_components[comp_type]:
                        best_components[comp_type] = component['long_name']

        specific_loc = best_components['neighborhood'] or best_components['sublocality_level_1']
        city = best_components['locality']
        state = best_components['administrative_area_level_1']

        if specific_loc and city:
            if specific_loc != city:
                return f"{specific_loc}, {city}"
            else:
                return city
        if city:
            return city
        if state:
            return state
        if best_components['country']:
            return best_components['country']

        return geocode_results[0].get('formatted_address') if geocode_results else None

    def _extract_faces(self, image_path):
        try:
            image = cv2.imread(image_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            detections = self.detector.detect_faces(image_rgb)
            faces = []
            for detection in detections:
                x, y, w, h = detection['box']
                face = image_rgb[y:y+h, x:x+w]
                face = cv2.resize(face, (160, 160))
                faces.append({'box': (x, y, w, h), 'face': face})
            return faces
        except Exception:
            return []

    def _get_embeddings(self, faces):
        embeddings = []
        for face_info in faces:
            face = face_info['face'].astype('float32')
            embedding_list = DeepFace.represent(face, model_name='Facenet', enforce_detection=False)
            if embedding_list:
                embeddings.append(np.array(embedding_list[0]['embedding'], dtype=np.float32))
        return embeddings

    def index_gallery(self, force_reindex=False, recognition_threshold=0.7):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # 1. Get all known faces with names for recognition
            cursor.execute("SELECT person_id, embedding FROM faces WHERE person_id IS NOT NULL")
            known_faces_raw = cursor.fetchall()
            known_faces = [{
                "person_id": person_id,
                "embedding": np.frombuffer(embedding_blob, dtype=np.float32)
            } for person_id, embedding_blob in known_faces_raw]
            print(f"Loaded {len(known_faces)} known faces for recognition.")

            image_files = [f for f in os.listdir(self.gallery_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

            for filename in image_files:
                filepath = os.path.join(self.gallery_path, filename)
                mod_time = os.path.getmtime(filepath)

                cursor.execute("SELECT id, mod_time, filepath FROM photos WHERE filename=?", (filename,))
                result = cursor.fetchone()

                photo_id = None
                if result:
                    photo_id, db_mod_time, db_filepath = result
                    if not force_reindex and db_mod_time == mod_time and db_filepath == filepath:
                        continue
                    
                    print(f"Updating {filename}...")
                    exif_data = self._get_exif_data(filepath)
                    capture_date_str = exif_data.get('DateTimeOriginal')
                    capture_date = datetime.strptime(capture_date_str, '%Y:%m:%d %H:%M:%S') if capture_date_str else None
                    lat, lon, loc_name = self._get_location_from_gps(exif_data)

                    cursor.execute("""
                        UPDATE photos 
                        SET filepath=?, capture_date=?, latitude=?, longitude=?, location_name=?, mod_time=?
                        WHERE id=?
                    """, (filepath, capture_date, lat, lon, loc_name, mod_time, photo_id))
                    cursor.execute("DELETE FROM faces WHERE photo_id=?", (photo_id,))
                else:
                    print(f"Indexing {filename}...")
                    exif_data = self._get_exif_data(filepath)
                    capture_date_str = exif_data.get('DateTimeOriginal')
                    capture_date = datetime.strptime(capture_date_str, '%Y:%m:%d %H:%M:%S') if capture_date_str else None
                    lat, lon, loc_name = self._get_location_from_gps(exif_data)

                    cursor.execute("""
                        INSERT INTO photos (filename, filepath, capture_date, latitude, longitude, location_name, mod_time) 
                        VALUES (?,?,?,?,?,?,?)
                    """, (filename, filepath, capture_date, lat, lon, loc_name, mod_time))
                    photo_id = cursor.lastrowid

                # 2. Extract faces and try to recognize them
                if photo_id:
                    faces = self._extract_faces(filepath)
                    if faces:
                        embeddings = self._get_embeddings(faces)
                        for i, face_info in enumerate(faces):
                            if i < len(embeddings):
                                x, y, w, h = face_info['box']
                                new_embedding = embeddings[i]
                                embedding_blob = new_embedding.tobytes()
                                
                                # --- Auto-recognition logic ---
                                best_match_person_id = None
                                if known_faces:
                                    best_match_score = 0.0
                                    for known_face in known_faces:
                                        similarity = self._check_similarity(new_embedding, known_face['embedding'])
                                        if similarity > best_match_score:
                                            best_match_score = similarity
                                            if similarity >= recognition_threshold:
                                                best_match_person_id = known_face['person_id']
                                    
                                    if best_match_person_id:
                                        print(f"  -> Recognized person ID {best_match_person_id} for a face in {filename} (Score: {best_match_score:.2f})")

                                # 3. Insert face with or without a person_id
                                cursor.execute("""
                                    INSERT INTO faces (photo_id, box_x, box_y, box_w, box_h, embedding, person_id) 
                                    VALUES (?,?,?,?,?,?,?)
                                """, (photo_id, x, y, w, h, embedding_blob, best_match_person_id))
            conn.commit()
        print("Gallery indexing complete.")

    def cluster_photos(self, time_eps_hours=24, loc_eps_km=0.5, min_samples=2):
        from sklearn.cluster import DBSCAN

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, capture_date, latitude, longitude FROM photos WHERE capture_date IS NOT NULL")
            photos_with_date = cursor.fetchall()

            if photos_with_date:
                timestamps = np.array([datetime.strptime(p[1], '%Y-%m-%d %H:%M:%S').timestamp() for p in photos_with_date]).reshape(-1, 1)
                time_eps_seconds = time_eps_hours * 3600
                time_clusters = DBSCAN(eps=time_eps_seconds, min_samples=min_samples).fit(timestamps)
                
                for i, photo_id in enumerate([p[0] for p in photos_with_date]):
                    cluster_id = int(time_clusters.labels_[i])
                    cursor.execute("UPDATE photos SET time_cluster_id=? WHERE id=?", (cluster_id, photo_id))

            cursor.execute("SELECT id, latitude, longitude FROM photos WHERE latitude IS NOT NULL AND longitude IS NOT NULL")
            photos_with_loc = cursor.fetchall()
            
            if len(photos_with_loc) >= min_samples:
                coords = np.array([[p[1], p[2]] for p in photos_with_loc])
                coords_rad = np.radians(coords)
                earth_radius_km = 6371
                loc_eps_rad = loc_eps_km / earth_radius_km
                
                loc_clusters = DBSCAN(eps=loc_eps_rad, min_samples=min_samples, metric='haversine').fit(coords_rad)
                
                for i, photo_id in enumerate([p[0] for p in photos_with_loc]):
                    cluster_id = int(loc_clusters.labels_[i])
                    cursor.execute("UPDATE photos SET location_cluster_id=? WHERE id=?", (cluster_id, photo_id))
            
            conn.commit()
        print("Clustering complete.")

    def get_time_clusters(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("""
                SELECT time_cluster_id, filepath, capture_date
                FROM photos
                WHERE time_cluster_id IS NOT NULL AND time_cluster_id != -1
                ORDER BY time_cluster_id, capture_date
            """)
            clusters = {}
            for row in cursor.fetchall():
                cid = row['time_cluster_id']
                if cid not in clusters:
                    clusters[cid] = []
                clusters[cid].append(dict(row))
            return clusters

    def get_location_clusters(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("""
                SELECT location_cluster_id, filepath, location_name, capture_date, latitude, longitude
                FROM photos
                WHERE location_cluster_id IS NOT NULL AND location_cluster_id != -1
                ORDER BY location_cluster_id, capture_date
            """)
            clusters = {}
            for row in cursor.fetchall():
                cid = row['location_cluster_id']
                if cid not in clusters:
                    clusters[cid] = []
                clusters[cid].append(dict(row))
            return clusters

    def get_all_photos(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM photos ORDER BY capture_date DESC")
            return [dict(row) for row in cursor.fetchall()]

    def get_all_person_names(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM persons ORDER BY name")
            return [row[0] for row in cursor.fetchall()]

    def search_by_person_name(self, name):
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("""
                SELECT p.*
                FROM photos p
                JOIN faces f ON p.id = f.photo_id
                JOIN persons pers ON f.person_id = pers.id
                WHERE pers.name = ?
                GROUP BY p.id
                ORDER BY p.capture_date DESC
            """, (name,))
            return [dict(row) for row in cursor.fetchall()]

    def get_faces_for_photo(self, photo_id):
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("""
                SELECT f.id, f.box_x, f.box_y, f.box_w, f.box_h, p.name as person_name
                FROM faces f
                LEFT JOIN persons p ON f.person_id = p.id
                WHERE f.photo_id = ?
            """, (photo_id,))
            return [dict(row) for row in cursor.fetchall()]

    def delete_photo(self, photo_id):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            # First, get the filename to delete the file
            cursor.execute("SELECT filename FROM photos WHERE id = ?", (photo_id,))
            result = cursor.fetchone()
            if not result:
                return {'status': 'error', 'message': 'Photo not found'}

            filename = result[0]
            filepath = os.path.join(self.gallery_path, filename)

            # Delete the photo from the filesystem
            if os.path.exists(filepath):
                try:
                    os.remove(filepath)
                    print(f"Deleted file: {filepath}")
                except OSError as e:
                    print(f"Error deleting file {filepath}: {e}")
                    return {'status': 'error', 'message': f'Failed to delete file: {e}'}

            # Delete the photo record from the database.
            # The ON DELETE CASCADE for faces table will handle associated faces.
            cursor.execute("DELETE FROM photos WHERE id = ?", (photo_id,))
            conn.commit()

            print(f"Deleted photo ID {photo_id} from database.")
            return {'status': 'success', 'message': 'Photo deleted successfully'}



    def _check_similarity(self, emb1, emb2):
        emb1 = emb1.flatten()
        emb2 = emb2.flatten()
        norm_emb1 = np.linalg.norm(emb1)
        norm_emb2 = np.linalg.norm(emb2)
        if norm_emb1 == 0 or norm_emb2 == 0:
            return 0.0
        return np.dot(emb1, emb2) / (norm_emb1 * norm_emb2)

    def search_by_face(self, query_image_path, threshold=0.5):
        query_faces = self._extract_faces(query_image_path)
        if not query_faces:
            print("No face detected in the query image.")
            return []

        query_embeddings = self._get_embeddings(query_faces)
        if not query_embeddings:
            print("Could not generate embedding for the query image.")
            return []
        
        query_embedding = np.array(query_embeddings[0], dtype=np.float32)

        all_matches = {}
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("""
                SELECT
                    p.filepath, p.location_name, p.capture_date,
                    f.id as face_id, f.box_x, f.box_y, f.box_w, f.box_h, f.embedding,
                    pers.name as person_name
                FROM photos p
                JOIN faces f ON p.id = f.photo_id
                LEFT JOIN persons pers ON f.person_id = pers.id
            """)
            
            all_db_faces = cursor.fetchall()

            for row in all_db_faces:
                db_embedding = np.frombuffer(row['embedding'], dtype=np.float32)
                similarity = self._check_similarity(query_embedding, db_embedding)
                
                if similarity >= threshold:
                    filepath = row['filepath']
                    if filepath not in all_matches:
                        all_matches[filepath] = {
                            'location': row['location_name'],
                            'date': row['capture_date'],
                            'faces': []
                        }
                    
                    all_matches[filepath]['faces'].append({
                        'score': similarity,
                        'box': (row['box_x'], row['box_y'], row['box_w'], row['box_h']),
                        'face_id': row['face_id'],
                        'person_name': row['person_name']
                    })

        output_matches = []
        for filepath, data in all_matches.items():
            # A photo is only a match if it has at least one face meeting the threshold.
            if data['faces']:
                # Sort the faces within the photo by score, descending
                data['faces'].sort(key=lambda x: x['score'], reverse=True)
                output_matches.append((filepath, data))

        if not output_matches:
            return []

        # Sort the list of matched photos by the score of the best-matching face in each photo
        output_matches.sort(key=lambda x: x[1]['faces'][0]['score'], reverse=True)
        
        return output_matches

    def assign_name_to_face(self, face_id, name, propagation_threshold=0.7):
        """
        Assigns a name to a specific face and propagates that name to all other
        similar, unnamed faces in the database.
        Returns a dictionary with the status and person_id for the API.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            name = name.strip()
            if not name:
                return {"status": "error", "error": "Name cannot be empty."}

            # 1. Get or create the person
            cursor.execute("SELECT id FROM persons WHERE name = ?", (name,))
            person_result = cursor.fetchone()
            if person_result:
                person_id = person_result[0]
            else:
                cursor.execute("INSERT INTO persons (name) VALUES (?)", (name,))
                person_id = cursor.lastrowid
            
            # 2. Update the primary face and get its embedding
            cursor.execute("UPDATE faces SET person_id = ? WHERE id = ?", (person_id, face_id))
            cursor.execute("SELECT embedding FROM faces WHERE id = ?", (face_id,))
            target_embedding_blob = cursor.fetchone()
            if not target_embedding_blob:
                return {"status": "error", "error": "Could not find the target face to start propagation."}
            
            target_embedding = np.frombuffer(target_embedding_blob[0], dtype=np.float32)

            # 3. Find all other unnamed faces and compare to propagate the name
            cursor.execute("SELECT id, embedding FROM faces WHERE person_id IS NULL AND id != ?", (face_id,))
            unnamed_faces = cursor.fetchall()
            
            updated_count = 1  # Starts at 1 for the face we just named

            for other_face_id, other_embedding_blob in unnamed_faces:
                other_embedding = np.frombuffer(other_embedding_blob, dtype=np.float32)
                similarity = self._check_similarity(target_embedding, other_embedding)
                
                if similarity >= propagation_threshold:
                    cursor.execute("UPDATE faces SET person_id = ? WHERE id = ?", (person_id, other_face_id))
                    updated_count += 1
            
            conn.commit()
            print(f"Assigned name '{name}' to face {face_id} and propagated to {updated_count - 1} other faces.")
            return {"status": "success", "person_id": person_id}

    def unassign_name_from_face(self, face_id):
        """Removes the assigned person from a specific face."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            # Set person_id to NULL to unassign the name
            cursor.execute("UPDATE faces SET person_id = NULL WHERE id = ?", (face_id,))
            conn.commit()
            # Check if the update was successful
            if cursor.rowcount > 0:
                return {"status": "success", "message": f"Name unassigned from face {face_id}."}
            else:
                # This case might happen if the face_id is invalid
                return {"status": "error", "error": f"Face with id {face_id} not found."}

    def get_all_persons(self):
        """Returns a list of all named persons in the database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT id, name FROM persons ORDER BY name")
            return [dict(row) for row in cursor.fetchall()]

    def get_photos_by_person(self, person_id):
        """Returns all photos containing a specific person."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("""
                SELECT DISTINCT p.*
                FROM photos p
                JOIN faces f ON p.id = f.photo_id
                WHERE f.person_id = ?
                ORDER BY p.capture_date DESC
            """, (person_id,))
            return [dict(row) for row in cursor.fetchall()]
