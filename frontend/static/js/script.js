document.addEventListener('DOMContentLoaded', function() {
    // --- DOM Elements ---
    const photoGrid = document.getElementById('photo-grid');
    const modal = document.getElementById('photo-modal');
    const modalImage = document.getElementById('modal-image');
    const modalCaption = document.getElementById('modal-caption');
    const faceBoxContainer = document.getElementById('face-box-container');
    const closeBtn = document.querySelector('.close-btn');

    const nameSearchForm = document.getElementById('name-search-form');
    const nameSearchInput = document.getElementById('name-search-input');
    const personNameList = document.getElementById('person-names');

    const uploadForm = document.getElementById('upload-form');
    const imageUploadInput = document.getElementById('image-upload');

    const searchForm = document.getElementById('search-form');
    const searchInput = document.getElementById('search-input');

    const clearSearchBtn = document.getElementById('clear-search-btn');

    // --- Initial Load ---
    loadAllPhotos();
    loadPersonNames();

    // --- Core Functions ---
    function loadAllPhotos() {
        fetch('/api/photos')
            .then(response => response.json())
            .then(displayPhotos)
            .catch(error => console.error('Error loading photos:', error));
    }

    function loadPersonNames() {
        fetch('/api/persons')
            .then(response => response.json())
            .then(names => {
                personNameList.innerHTML = '';
                names.forEach(name => {
                    const option = document.createElement('option');
                    option.value = name;
                    personNameList.appendChild(option);
                });
            })
            .catch(error => console.error('Error loading person names:', error));
    }

    function displayPhotos(photos) {
        photoGrid.innerHTML = '';
        if (photos.length === 0) {
            photoGrid.innerHTML = '<p class="no-photos-msg">No photos found. Try a different search or upload some photos!</p>';
            return;
        }
        clearSearchBtn.style.display = 'block';
        photos.forEach(photo => {
            const photoItemContainer = document.createElement('div');
            photoItemContainer.className = 'photo-item-container';

            const photoItem = document.createElement('div');
            photoItem.className = 'photo-item';
            photoItem.innerHTML = `<img src="/images/${photo.filename}" alt="${photo.filename}" loading="lazy">`;
            photoItem.addEventListener('click', () => openModal(photo));

            const photoActions = document.createElement('div');
            photoActions.className = 'photo-actions';
            
            const deleteBtn = document.createElement('button');
            deleteBtn.className = 'delete-photo-btn';

            deleteBtn.title = 'Delete Photo';
            deleteBtn.onclick = (e) => {
                e.stopPropagation();
                deletePhoto(photo.id, photoItemContainer);
            };

            photoActions.appendChild(deleteBtn);
            photoItemContainer.appendChild(photoItem);
            photoItemContainer.appendChild(photoActions);
            photoGrid.appendChild(photoItemContainer);
        });
    }

    // --- Modal & Face Tagging ---
    function openModal(photo) {
        modal.style.display = 'block';
        modalImage.src = `/images/${photo.filename}`;
        let captionText = `Filename: ${photo.filename}`;
        if (photo.capture_date) captionText += `<br>Captured: ${new Date(photo.capture_date).toLocaleString()}`;
        if (photo.location_name) captionText += `<br>Location: ${photo.location_name}`;
        modalCaption.innerHTML = captionText;

        modalImage.onload = () => displayFaces(photo.id, modalImage);
        if (modalImage.complete) displayFaces(photo.id, modalImage);
    }

    function displayFaces(photoId, imageElement) {
        faceBoxContainer.innerHTML = '';
        fetch(`/api/photo/${photoId}/faces`)
            .then(response => response.json())
            .then(faces => {
                const imageRect = imageElement.getBoundingClientRect();
                const containerRect = faceBoxContainer.getBoundingClientRect();

                const scaleX = imageRect.width / imageElement.naturalWidth;
                const scaleY = imageRect.height / imageElement.naturalHeight;

                const offsetX = imageRect.left - containerRect.left;
                const offsetY = imageRect.top - containerRect.top;

                faces.forEach(face => {
                    const faceBox = document.createElement('div');
                    faceBox.className = 'face-box';
                    
                    faceBox.style.left = `${(face.box_x * scaleX) + offsetX}px`;
                    faceBox.style.top = `${(face.box_y * scaleY) + offsetY}px`;
                    faceBox.style.width = `${face.box_w * scaleX}px`;
                    faceBox.style.height = `${face.box_h * scaleY}px`;
                    
                    if (face.person_name) {
                        const nameTag = document.createElement('div');
                        nameTag.className = 'face-tag';
                        
                        const nameSpan = document.createElement('span');
                        nameSpan.textContent = face.person_name;
                        
                        const deleteBtn = document.createElement('span');
                        deleteBtn.className = 'delete-face-name';
                        deleteBtn.innerHTML = '&times;';
                        deleteBtn.title = 'Unassign Name';
                        deleteBtn.onclick = (e) => {
                            e.stopPropagation();
                            unassignNameFromFace(face.id, faceBox);
                        };
                        
                        nameTag.appendChild(nameSpan);
                        nameTag.appendChild(deleteBtn);
                        faceBox.appendChild(nameTag);
                    } else {
                        faceBox.title = 'Click to tag this face';
                        faceBox.addEventListener('click', () => showTagInput(faceBox, face.id), { once: true });
                    }
                    faceBoxContainer.appendChild(faceBox);
                });
            });
    }

    function showTagInput(faceBox, faceId) {
        if (faceBox.querySelector('.tag-input-container')) return;
        const inputContainer = document.createElement('div');
        inputContainer.className = 'tag-input-container';
        const input = document.createElement('input');
        input.type = 'text';
        input.className = 'tag-input';
        input.placeholder = 'Enter name';
        input.onclick = (e) => e.stopPropagation();
        const saveBtn = document.createElement('button');
        saveBtn.className = 'tag-save-btn';
        saveBtn.textContent = 'Save';
        saveBtn.onclick = (e) => {
            e.stopPropagation();
            const name = input.value.trim();
            if (name) assignNameToFace(faceId, name, faceBox);
        };
        inputContainer.appendChild(input);
        inputContainer.appendChild(saveBtn);
        faceBox.appendChild(inputContainer);
        input.focus();
    }

    function assignNameToFace(faceId, name, faceBox) {
        fetch('/api/tag_face', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ face_id: faceId, name: name })
        })
        .then(response => {
            if (!response.ok) {
                return response.text().then(text => {
                    throw new Error(`HTTP error! Status: ${response.status}, Body: ${text}`);
                });
            }
            return response.json();
        })
        .then(result => {
            if (result.status === 'success') {
                faceBox.innerHTML = ''; // Clear the input form
                const nameTag = document.createElement('div');
                nameTag.className = 'face-tag';
                
                const nameSpan = document.createElement('span');
                nameSpan.textContent = name;

                const deleteBtn = document.createElement('span');
                deleteBtn.className = 'delete-face-name';
                deleteBtn.innerHTML = '&times;';
                deleteBtn.title = 'Unassign Name';
                deleteBtn.onclick = (e) => {
                    e.stopPropagation();
                    unassignNameFromFace(faceId, faceBox);
                };

                nameTag.appendChild(nameSpan);
                nameTag.appendChild(deleteBtn);
                faceBox.appendChild(nameTag);
                faceBox.title = name;
                loadPersonNames();
            } else {
                alert('Failed to save name: ' + (result.error || 'Unknown error from server'));
                showTagInput(faceBox, faceId);
            }
        })
        .catch(error => {
            console.error('Error assigning name:', error);
            alert('Failed to save name. See browser console for details. Error: ' + error.message);
            showTagInput(faceBox, faceId);
        });
    }

    function unassignNameFromFace(faceId, faceBox) {
        if (!confirm('Are you sure you want to remove this name?')) {
            return;
        }
        fetch('/api/unassign_face_name', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ face_id: faceId })
        })
        .then(response => response.json())
        .then(result => {
            if (result.status === 'success') {
                faceBox.innerHTML = ''; // Clear the name tag
                faceBox.title = 'Click to tag this face';
                // Re-add the click listener to allow tagging again
                faceBox.addEventListener('click', () => showTagInput(faceBox, faceId), { once: true });
            } else {
                alert('Failed to unassign name: ' + (result.error || 'Unknown error'));
            }
        })
        .catch(error => {
            console.error('Error unassigning name:', error);
            alert('An error occurred while unassigning the name. Please check the console.');
        });
    }

    function deletePhoto(photoId, elementToRemove) {
        if (!confirm('Are you sure you want to delete this photo? This action cannot be undone.')) {
            return;
        }

        fetch(`/api/photo/${photoId}`, {
            method: 'DELETE'
        })
        .then(response => {
            if (!response.ok) {
                return response.json().then(err => { throw new Error(err.error || 'Server error') });
            }
            return response.json();
        })
        .then(result => {
            if (result.status === 'success') {
                elementToRemove.remove();
                // Optional: show a small success message
            } else {
                alert('Failed to delete photo: ' + result.message);
            }
        })
        .catch(error => {
            console.error('Error deleting photo:', error);
            alert('An error occurred while deleting the photo: ' + error.message);
        });
    }

    function closeModal() {
        modal.style.display = 'none';
        faceBoxContainer.innerHTML = '';
        faceBoxContainer.style.top = '0';
        faceBoxContainer.style.left = '0';
        faceBoxContainer.style.width = '100%';
        faceBoxContainer.style.height = '100%';
    }

    // --- Event Listeners ---
    closeBtn.addEventListener('click', closeModal);

    window.addEventListener('click', (event) => {
        if (event.target === modal) {
            closeModal();
        }
    });

    imageUploadInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            const formData = new FormData(uploadForm);
            const uploadBtnLabel = document.querySelector('label[for="image-upload"]');
            const originalText = uploadBtnLabel.textContent;
            uploadBtnLabel.textContent = 'Uploading...';

            fetch('/api/upload', { method: 'POST', body: formData })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        loadAllPhotos();
                        loadPersonNames();
                    } else {
                        alert('Upload failed: ' + (data.error || 'Unknown error'));
                    }
                })
                .catch(error => console.error('Error uploading files:', error))
                .finally(() => {
                    uploadBtnLabel.textContent = originalText;
                    uploadForm.reset();
                });
        }
    });

    searchInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            const formData = new FormData(searchForm);
            fetch('/api/search_by_face', { method: 'POST', body: formData })
                .then(response => response.json())
                .then(data => data.error ? alert('Search failed: ' + data.error) : displayPhotos(data))
                .catch(error => console.error('Error searching by face:', error));
        }
    });

    nameSearchForm.addEventListener('submit', (e) => {
        e.preventDefault();
        const name = nameSearchInput.value;
        if (!name) return;
        fetch(`/api/search_by_name?name=${encodeURIComponent(name)}`)
            .then(response => response.json())
            .then(data => data.error ? alert('Search failed: ' + data.error) : displayPhotos(data))
            .catch(error => console.error('Error searching by name:', error));
    });

    clearSearchBtn.addEventListener('click', () => {
        nameSearchInput.value = '';
        searchInput.value = '';
        clearSearchBtn.style.display = 'none';
        loadAllPhotos();
    });
});
