import sys
import os
import tempfile
from collections import Counter
from datetime import datetime
from flask import Flask, jsonify, request, send_from_directory, render_template

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from photo_manager import PhotoManager

app = Flask(__name__, static_folder='../frontend/static', template_folder='../frontend/templates')

# Define paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
GALLERY_FOLDER = os.path.join(PROJECT_ROOT, 'images')
TEMP_FOLDER = os.path.join(PROJECT_ROOT, 'temp')
os.makedirs(TEMP_FOLDER, exist_ok=True)

# Initialize PhotoManager
photo_manager = PhotoManager(gallery_path=GALLERY_FOLDER)

@app.route('/')
def index():
    return send_from_directory(app.template_folder, 'index.html')

@app.route('/images/<path:filename>')
def serve_image(filename):
    return send_from_directory(GALLERY_FOLDER, filename)

@app.route('/api/photos')
def get_photos():
    photos_data = photo_manager.get_all_photos()
    for photo in photos_data:
        photo['url'] = f'/images/{photo["filename"]}'
    return jsonify(photos_data)

@app.route('/api/photos/by_location')
def get_photos_by_location():
    location_clusters = photo_manager.get_location_clusters()
    
    grouped_photos = []
    for cid, photos in sorted(location_clusters.items()):
        # Determine the most common location name for the cluster title
        location_names = [p.get('location_name') for p in photos if p.get('location_name')]
        if location_names:
            most_common_loc = Counter(location_names).most_common(1)[0][0]
            display_title = most_common_loc
        else:
            # Fallback to coordinates if no names are available
            lats = [p['latitude'] for p in photos if p.get('latitude') is not None]
            lons = [p['longitude'] for p in photos if p.get('longitude') is not None]
            if lats and lons:
                avg_lat = sum(lats) / len(lats)
                avg_lon = sum(lons) / len(lons)
                display_title = f"Location near ({avg_lat:.4f}, {avg_lon:.4f})"
            else:
                display_title = f"Unknown Location Cluster #{cid}"

        # Add url to each photo
        for photo in photos:
            photo['url'] = f'/images/{os.path.basename(photo["filepath"])}'

        grouped_photos.append({
            'title': display_title,
            'photos': photos
        })
        
    return jsonify(grouped_photos)

@app.route('/api/photos/by_date')
def get_photos_by_date():
    time_clusters = photo_manager.get_time_clusters()
    
    grouped_photos = []
    for cid, photos in sorted(time_clusters.items()):
        # Determine the date for the cluster title from the first photo
        date_str = photos[0].get('capture_date')
        if date_str:
            try:
                date = datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S').strftime('%Y-%m-%d')
                display_title = f"Album from {date}"
            except ValueError:
                display_title = f"Unknown Date Cluster #{cid}"
        else:
            display_title = f"Unknown Date Cluster #{cid}"

        # Add url to each photo
        for photo in photos:
            photo['url'] = f'/images/{os.path.basename(photo["filepath"])}'

        grouped_photos.append({
            'title': display_title,
            'photos': photos
        })
        
    return jsonify(grouped_photos)

@app.route('/api/upload', methods=['POST'])
def upload_photos():
    if 'images' not in request.files:
        return jsonify({'error': 'No images part in the request'}), 400
    files = request.files.getlist('images')
    if not files or all(f.filename == '' for f in files):
        return jsonify({'error': 'No selected files'}), 400

    for file in files:
        if file:
            file.save(os.path.join(GALLERY_FOLDER, file.filename))

    try:
        photo_manager.index_gallery()
    except Exception as e:
        print(f"Error during gallery indexing: {e}")
        return jsonify({'success': False, 'error': 'Upload successful, but failed to re-index gallery.'}), 500

    return jsonify({'success': True, 'message': 'Upload successful, gallery has been updated.'}), 200

@app.route('/api/search_by_face', methods=['POST'])
def search_by_face():
    if 'image' not in request.files:
        return jsonify({'error': 'No image for search in the request'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file for search'}), 400

    # Save to a temporary file
    temp_path = os.path.join(TEMP_FOLDER, file.filename)
    file.save(temp_path)

    try:
        matches = photo_manager.search_by_face(temp_path)
        
        # Format results for the frontend
        results = []
        for filepath, data in matches:
            filename = os.path.basename(filepath)
            results.append({
                'filename': filename,
                'url': f'/images/{filename}',
                'location_name': data.get('location'),
                'capture_date': data.get('date')
            })
        return jsonify(results)
    except Exception as e:
        print(f"Error during face search: {e}")
        return jsonify({'error': 'Failed to perform face search.'}), 500
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.route('/api/persons')
def get_persons():
    try:
        names = photo_manager.get_all_person_names()
        return jsonify(names)
    except Exception as e:
        print(f"Error getting person names: {e}")
        return jsonify({'error': 'Failed to retrieve person names.'}), 500

@app.route('/api/search_by_name')
def search_by_name():
    name = request.args.get('name')
    if not name:
        return jsonify({'error': 'Name parameter is required'}), 400
    
    try:
        photos_data = photo_manager.search_by_person_name(name)
        for photo in photos_data:
            photo['url'] = f'/images/{photo["filename"]}'
        return jsonify(photos_data)
    except Exception as e:
        print(f"Error during search by name: {e}")
        return jsonify({'error': 'Failed to perform search by name.'}), 500

@app.route('/api/photo/<int:photo_id>/faces')
def get_photo_faces(photo_id):
    try:
        faces = photo_manager.get_faces_for_photo(photo_id)
        return jsonify(faces)
    except Exception as e:
        print(f"Error getting faces for photo {photo_id}: {e}")
        return jsonify({'error': 'Failed to retrieve face data.'}), 500

@app.route('/api/tag_face', methods=['POST'])
def tag_face():
    data = request.json
    print(f"[DEBUG] Received tag request data: {data}")
    face_id = data.get('face_id')
    name = data.get('name')

    if not face_id or not name:
        return jsonify({'error': 'face_id and name are required'}), 400

    try:
        result = photo_manager.assign_name_to_face(face_id, name)
        print(f"[DEBUG] Returning result for tag request: {result}")
        return jsonify(result)
    except Exception as e:
        print(f"Error tagging face: {e}")
        return jsonify({'error': 'Failed to tag face.'}), 500

@app.route('/api/unassign_face_name', methods=['POST'])
def unassign_face_name():
    data = request.json
    face_id = data.get('face_id')
    if not face_id:
        return jsonify({'error': 'face_id is required'}), 400
    try:
        result = photo_manager.unassign_name_from_face(face_id)
        if result.get('status') == 'success':
            return jsonify(result), 200
        else:
            return jsonify(result), 404 # Or appropriate error code
    except Exception as e:
        print(f"Error unassigning name from face: {e}")
        return jsonify({'error': 'Failed to unassign name.'}), 500

@app.route('/api/photo/<int:photo_id>', methods=['DELETE'])
def delete_photo(photo_id):
    try:
        result = photo_manager.delete_photo(photo_id)
        if result.get('status') == 'success':
            return jsonify(result), 200
        else:
            return jsonify(result), 404
    except Exception as e:
        print(f"Error deleting photo: {e}")
        return jsonify({'error': 'Failed to delete photo.'}), 500

@app.route('/albums')
def albums_page():
    google_maps_api_key = os.getenv('GOOGLE_MAPS_API_KEY')
    return render_template('albums.html', gmaps_api_key=google_maps_api_key)

if __name__ == '__main__':
    print("Starting server...")
    app.run(debug=True, port=5001)
