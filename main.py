import streamlit as st
import os
from datetime import datetime
from collections import Counter
from PIL import Image, ImageOps
from photo_manager import PhotoManager

def orient_image(image_path):
    """Corrects image orientation based on EXIF data."""
    try:
        image = Image.open(image_path)
        return ImageOps.exif_transpose(image)
    except Exception:
        # Fallback to the original path if orientation fails
        return image_path

def main():
    st.set_page_config(layout="wide")
    st.title("Local Photo Manager & Face Recognition")

    # Initialize the photo manager
    photo_manager = PhotoManager(db_path='gallery.db', gallery_path='images')

    # --- Sidebar for Gallery Management ---
    st.sidebar.title("Gallery Management")
    if st.sidebar.button("Index Gallery"):
        with st.spinner('Indexing new or modified photos...'):
            photo_manager.index_gallery(force_reindex=False)
        st.sidebar.success("Gallery indexing complete.")

    if st.sidebar.button("Force Re-index"):
        with st.spinner('Force re-indexing all images... This may take a while.'):
            photo_manager.index_gallery(force_reindex=True)
        st.sidebar.success("Gallery has been re-indexed.")

    if st.sidebar.button("Cluster Photos"):
        with st.spinner('Clustering photos by time and location...'):
            photo_manager.cluster_photos()
        st.sidebar.success("Photo clustering complete.")

    # --- Main Content ---
    st.header("Search by Face")

    uploaded_file = st.file_uploader("Upload an image to find similar faces...", type=["jpg", "jpeg", "png"])
    threshold = st.slider("Similarity Threshold", 0.0, 1.0, 0.6, 0.05)

    if uploaded_file is not None:
        # Ensure 'uploads' directory exists
        if not os.path.exists("uploads"):
            os.makedirs("uploads")
        
        query_image_path = os.path.join("uploads", uploaded_file.name)
        with open(query_image_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.image(orient_image(query_image_path), caption='Query Image', width=150)

        with st.spinner('Searching for similar faces...'):
            matches = photo_manager.search_by_face(query_image_path, threshold)

        if not matches:
            st.write("No matches found.")
        else:
            st.header(f"{len(matches)} matching photos found:")
            for filepath, data in matches:
                st.subheader(f"Photo: {os.path.basename(filepath)}")
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.image(orient_image(filepath), use_column_width=True)
                with col2:
                    st.write(f"**Location:** {data.get('location', 'N/A')}")
                    st.write(f"**Date:** {data.get('date', 'N/A')}")
                    st.write(f"**Found {len(data['faces'])} matching faces:**")
                    for face in data['faces']:
                        st.write(f"- Score: {face['score']:.2f}")

        # Clean up the uploaded file
        os.remove(query_image_path)

    st.divider()

    # --- Photo Albums Display ---
    st.header("Photo Albums")

    time_clusters = photo_manager.get_time_clusters()
    location_clusters = photo_manager.get_location_clusters()

    if not time_clusters and not location_clusters:
        st.info(
            "No photo albums to display. To create albums, please do the following:\n\n" 
            "1.  Ensure your photos have **GPS and date information** in their metadata.\n"
            "2.  Click the **'Index Gallery'** or **'Force Re-index'** button in the sidebar.\n"
            "3.  Click the **'Cluster Photos'** button to generate albums."
        )
    else:
        if time_clusters:
            st.subheader("Albums by Date")
            for cid, photos in time_clusters.items():
                date_str = photos[0].get('capture_date')
                if date_str:
                    date = datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S').strftime('%Y-%m-%d')
                    with st.expander(f"Album from {date} ({len(photos)} photos)"):
                        st.image([orient_image(p['filepath']) for p in photos], width=150)

        if location_clusters:
            st.subheader("Albums by Location")
            
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

                with st.expander(f"{display_title} ({len(photos)} photos)"):
                    cols = st.columns(5)  # Display thumbnails in columns
                    for i, p in enumerate(photos):
                        cols[i % 5].image(orient_image(p['filepath']), width=150)

if __name__ == "__main__":
    main()

