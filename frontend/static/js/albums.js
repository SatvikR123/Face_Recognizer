function initPage() {
    loadDateAlbums();
    loadLocationAlbums();
}

function loadDateAlbums() {
    fetch('/api/photos/by_date')
        .then(response => response.json())
        .then(data => {
            const dateGrid = document.getElementById('date-grid');
            if (data.length === 0) {
                dateGrid.innerHTML = '<p>No date albums found.</p>';
                return;
            }
            data.forEach(group => {
                const groupContainer = document.createElement('div');
                groupContainer.className = 'album-group';

                const groupTitle = document.createElement('h2');
                groupTitle.className = 'album-group-title';
                groupTitle.textContent = group.title;
                groupContainer.appendChild(groupTitle);

                const photoGrid = document.createElement('div');
                photoGrid.className = 'photo-grid';

                group.photos.forEach(photo => {
                    const photoItem = document.createElement('div');
                    photoItem.className = 'photo-item';
                    photoItem.innerHTML = `<img src="${photo.url}" alt="Photo" loading="lazy">`;
                    photoGrid.appendChild(photoItem);
                });

                groupContainer.appendChild(photoGrid);
                dateGrid.appendChild(groupContainer);
            });
        })
        .catch(error => console.error('Error loading date albums:', error));
}

function loadLocationAlbums() {
    fetch('/api/photos/by_location')
        .then(response => response.json())
        .then(data => {
            const locationGrid = document.getElementById('location-grid');
            if (data.length === 0) {
                locationGrid.innerHTML = '<p>No location albums found.</p>';
                return;
            }
            data.forEach((group, index) => {
                const groupContainer = document.createElement('div');
                groupContainer.className = 'album-group';

                const groupTitle = document.createElement('h2');
                groupTitle.className = 'album-group-title';
                groupTitle.textContent = group.title;
                groupContainer.appendChild(groupTitle);

                const mapContainer = document.createElement('div');
                mapContainer.id = `map-${index}`;
                mapContainer.className = 'map-container';
                groupContainer.appendChild(mapContainer);

                const photoGrid = document.createElement('div');
                photoGrid.className = 'photo-grid';

                group.photos.forEach(photo => {
                    const photoItem = document.createElement('div');
                    photoItem.className = 'photo-item';
                    photoItem.innerHTML = `<img src="${photo.url}" alt="Photo" loading="lazy">`;
                    photoGrid.appendChild(photoItem);
                });

                groupContainer.appendChild(photoGrid);
                locationGrid.appendChild(groupContainer);

                // Initialize map
                if (group.photos.length > 0) {
                    const lats = group.photos.map(p => p.latitude).filter(lat => lat !== null);
                    const lons = group.photos.map(p => p.longitude).filter(lon => lon !== null);
                    if (lats.length > 0 && lons.length > 0) {
                        const centerLat = lats.reduce((a, b) => a + b, 0) / lats.length;
                        const centerLon = lons.reduce((a, b) => a + b, 0) / lons.length;

                        const map = new google.maps.Map(document.getElementById(`map-${index}`), {
                            center: { lat: centerLat, lng: centerLon },
                            zoom: 12
                        });

                        group.photos.forEach(p => {
                            if (p.latitude !== null && p.longitude !== null) {
                                new google.maps.Marker({
                                    position: { lat: p.latitude, lng: p.longitude },
                                    map: map,
                                    title: p.filename
                                });
                            }
                        });
                    }
                }
            });
        })
        .catch(error => console.error('Error loading location albums:', error));
}
