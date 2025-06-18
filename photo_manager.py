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
            CREATE TABLE IF NOT EXISTS faces (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                photo_id INTEGER,
                box_x INTEGER, box_y INTEGER, box_w INTEGER, box_h INTEGER,
                embedding BLOB,
                FOREIGN KEY(photo_id) REFERENCES photos(id)
            )
            ''')
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
                # Use the new method to get a friendlier name
                location_name = self._extract_friendly_location_name(reverse_geocode_result)
                return latitude, longitude, location_name
        except Exception as e:
            print(f"Google Maps reverse geocoding failed for ({latitude}, {longitude}): {e}")
            
        # Return coordinates even if name lookup fails
        return latitude, longitude, None

    def _extract_friendly_location_name(self, geocode_results):
        """
        Extracts a descriptive and unambiguous location name from geocode results.
        For example: "Sector 38, Noida" or "Paschim Vihar, Delhi".
        """
        if not geocode_results:
            return None

        # We will try to find the best available components from all results.
        best_components = {
            'neighborhood': None,
            'sublocality_level_1': None,
            'locality': None,
            'administrative_area_level_1': None,
            'country': None
        }

        # Search through all results to populate our best components
        for result in geocode_results:
            for component in result.get('address_components', []):
                for comp_type in best_components.keys():
                    if comp_type in component.get('types', []) and not best_components[comp_type]:
                        best_components[comp_type] = component['long_name']

        # Now, construct the name based on what we found.
        specific_loc = best_components['neighborhood'] or best_components['sublocality_level_1']
        city = best_components['locality']
        state = best_components['administrative_area_level_1']

        if specific_loc and city:
            # Avoid redundancy e.g. "Delhi, Delhi"
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

        # Fallback to the formatted address of the first result
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

    def index_gallery(self, force_reindex=False):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            image_files = [f for f in os.listdir(self.gallery_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

            for filename in image_files:
                filepath = os.path.join(self.gallery_path, filename)
                mod_time = os.path.getmtime(filepath)

                cursor.execute("SELECT mod_time FROM photos WHERE filepath=?", (filepath,))
                result = cursor.fetchone()

                if not force_reindex and result and result[0] == mod_time:
                    continue

                print(f"Indexing {filename}...")
                exif_data = self._get_exif_data(filepath)
                capture_date_str = exif_data.get('DateTimeOriginal')
                capture_date = datetime.strptime(capture_date_str, '%Y:%m:%d %H:%M:%S') if capture_date_str else None
                lat, lon, loc_name = self._get_location_from_gps(exif_data)

                cursor.execute("SELECT id FROM photos WHERE filepath=?", (filepath,))
                photo_result = cursor.fetchone()
                if photo_result:
                    photo_id = photo_result[0]
                    cursor.execute("""
                        UPDATE photos 
                        SET capture_date=?, latitude=?, longitude=?, location_name=?, mod_time=?
                        WHERE id=?
                    """, (capture_date, lat, lon, loc_name, mod_time, photo_id))
                    cursor.execute("DELETE FROM faces WHERE photo_id=?", (photo_id,))
                else:
                    cursor.execute("""
                        INSERT INTO photos (filename, filepath, capture_date, latitude, longitude, location_name, mod_time) 
                        VALUES (?,?,?,?,?,?,?)
                    """, (filename, filepath, capture_date, lat, lon, loc_name, mod_time))
                    photo_id = cursor.lastrowid

                faces = self._extract_faces(filepath)
                if faces:
                    embeddings = self._get_embeddings(faces)
                    for i, face_info in enumerate(faces):
                        if i < len(embeddings):
                            x, y, w, h = face_info['box']
                            embedding_blob = embeddings[i].tobytes()
                            cursor.execute("INSERT INTO faces (photo_id, box_x, box_y, box_w, box_h, embedding) VALUES (?,?,?,?,?,?)",
                                           (photo_id, x, y, w, h, embedding_blob))
            conn.commit()
        print("Gallery indexing complete.")

    def cluster_photos(self, time_eps_hours=24, loc_eps_km=0.5, min_samples=2):
        from sklearn.cluster import DBSCAN

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, capture_date, latitude, longitude FROM photos WHERE capture_date IS NOT NULL")
            photos_with_date = cursor.fetchall()

            # Time Clustering
            if photos_with_date:
                timestamps = np.array([datetime.strptime(p[1], '%Y-%m-%d %H:%M:%S').timestamp() for p in photos_with_date]).reshape(-1, 1)
                time_eps_seconds = time_eps_hours * 3600
                time_clusters = DBSCAN(eps=time_eps_seconds, min_samples=min_samples).fit(timestamps)
                
                for i, photo_id in enumerate([p[0] for p in photos_with_date]):
                    cluster_id = int(time_clusters.labels_[i])
                    cursor.execute("UPDATE photos SET time_cluster_id=? WHERE id=?", (cluster_id, photo_id))

            # Location Clustering
            cursor.execute("SELECT id, latitude, longitude FROM photos WHERE latitude IS NOT NULL AND longitude IS NOT NULL")
            photos_with_loc = cursor.fetchall()
            
            if len(photos_with_loc) >= min_samples:
                coords = np.array([[p[1], p[2]] for p in photos_with_loc])
                # Convert coords to radians for haversine metric
                coords_rad = np.radians(coords)
                # Convert km to radians for eps
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

    def get_faces_for_photo(self, photo_id):
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM faces WHERE photo_id=?", (photo_id,))
            return [dict(row) for row in cursor.fetchall()]

    def search_similar_faces(self, query_embedding, threshold=0.5):
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute('''
            SELECT p.filepath, p.capture_date, p.location_name, f.box_x, f.box_y, f.box_w, f.box_h, f.embedding
            FROM photos p
            JOIN faces f ON p.id = f.photo_id
            ''')
            
            all_faces = cursor.fetchall()
            all_matches = {}

            for row in all_faces:
                db_embedding = np.frombuffer(row['embedding'], dtype=np.float32)
                similarity = self._check_similarity(query_embedding, db_embedding)
                
                if similarity >= threshold:
                    filepath = row['filepath']
                    match_data = {
                        'box': (row['box_x'], row['box_y'], row['box_w'], row['box_h']),
                        'score': similarity,
                        'capture_date': row['capture_date'],
                        'location_name': row['location_name']
                    }
                    
                    if filepath not in all_matches:
                        all_matches[filepath] = []
                    all_matches[filepath].append(match_data)
            
            output_matches = []
            for filepath, faces in all_matches.items():
                faces.sort(key=lambda x: x['score'], reverse=True)
                output_matches.append((filepath, faces))

            output_matches.sort(key=lambda x: x[1][0]['score'], reverse=True)
            
            return output_matches

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

        all_matches = []
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("""
                SELECT p.filepath, p.location_name, p.capture_date, f.box_x, f.box_y, f.box_w, f.box_h, f.embedding
                FROM photos p JOIN faces f ON p.id = f.photo_id
            """)
            
            for row in cursor.fetchall():
                db_embedding = np.frombuffer(row['embedding'], dtype=np.float32)
                similarity = self._check_similarity(query_embedding, db_embedding)
                
                if similarity >= threshold:
                    match_data = {
                        'filepath': row['filepath'],
                        'location': row['location_name'],
                        'date': row['capture_date'],
                        'box': (row['box_x'], row['box_y'], row['box_w'], row['box_h']),
                        'score': similarity
                    }
                    all_matches.append(match_data)
        
        grouped_matches = {}
        for match in all_matches:
            path = match['filepath']
            if path not in grouped_matches:
                grouped_matches[path] = {'faces': [], 'location': match['location'], 'date': match['date']}
            grouped_matches[path]['faces'].append({'box': match['box'], 'score': match['score']})

        final_results = []
        for path, data in grouped_matches.items():
            final_results.append((path, data))

        final_results.sort(key=lambda x: max(f['score'] for f in x[1]['faces']), reverse=True)
        
        return final_results
