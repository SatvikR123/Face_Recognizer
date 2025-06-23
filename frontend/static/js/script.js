document.addEventListener('DOMContentLoaded', function() {
    // --- DOM Elements ---
    const photoGrid = document.querySelector('.photo-grid');
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



    let currentPage = 1;
    let isLoading = false;
    let allPhotosLoaded = false;

    // --- Initial Load ---
    loadPhotos();
    loadPersonNames();

    // --- Infinite Scroll ---
    window.addEventListener('scroll', () => {
        if (isLoading || allPhotosLoaded) return;
        if ((window.innerHeight + window.scrollY) >= document.body.offsetHeight - 500) {
            loadPhotos();
        }
    });

    function showLoadingIndicator(show) {
        let indicator = document.getElementById('loading-indicator');
        if (!indicator) {
            indicator = document.createElement('div');
            indicator.id = 'loading-indicator';
            indicator.textContent = 'Loading...';
            document.body.appendChild(indicator);
        }
        indicator.style.display = show ? 'block' : 'none';
    }

    // Toast Notifications
    function showToast(message, type = 'info', duration = 3000) {
        Toastify({
            text: message,
            duration: duration,
            gravity: 'top', // top or bottom
            position: 'right', // left, center or right
            close: true,
            style: {
                background:
                    type === 'error'
                        ? '#e74c3c' // red
                        : type === 'success'
                        ? '#2ecc71' // green
                        : type === 'warning'
                        ? '#f1c40f' // yellow
                        : '#3498db', // blue for info/default
                color: '#ffffff',
            },
        }).showToast();
    }

    // Confirmation Dialog
    function showConfirm(message) {
        return new Promise((resolve) => {
            const overlay = document.createElement('div');
            overlay.className = 'confirm-overlay';
            const dialog = document.createElement('div');
            dialog.className = 'confirm-dialog';

            const msg = document.createElement('p');
            msg.textContent = message;

            const buttons = document.createElement('div');
            buttons.className = 'confirm-buttons';

            const cancelBtn = document.createElement('button');
            cancelBtn.textContent = 'Cancel';
            cancelBtn.className = 'confirm-cancel-btn';
            const okBtn = document.createElement('button');
            okBtn.textContent = 'OK';
            okBtn.className = 'confirm-ok-btn';

            cancelBtn.onclick = () => {
                document.body.removeChild(overlay);
                resolve(false);
            };
            okBtn.onclick = () => {
                document.body.removeChild(overlay);
                resolve(true);
            };

            buttons.appendChild(cancelBtn);
            buttons.appendChild(okBtn);
            dialog.appendChild(msg);
            dialog.appendChild(buttons);
            overlay.appendChild(dialog);
            document.body.appendChild(overlay);
        });
    }

    // --- Core Functions ---
    function loadPhotos() {
        if (isLoading || allPhotosLoaded) return;
        isLoading = true;
        showLoadingIndicator(true);

        fetch(`/api/photos?page=${currentPage}`)
            .then(response => response.json())
            .then(newPhotos => {
                if (newPhotos.length > 0) {
                    displayPhotos(newPhotos);
                    currentPage++;
                } else {
                    allPhotosLoaded = true;
                }
            })
            .catch(error => console.error('Error loading photos:', error))
            .finally(() => {
                isLoading = false;
                showLoadingIndicator(false);
                // Check if the content is scrollable and load more if it's not.
                setTimeout(() => {
                    const isScrollable = document.documentElement.scrollHeight > document.documentElement.clientHeight;
                    if (!isScrollable && !allPhotosLoaded) {
                        loadPhotos();
                    }
                }, 200);
            });
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
        if (photos.length === 0 && photoGrid.childElementCount === 0) {
            photoGrid.innerHTML = '<p class="no-photos-msg">No photos found.</p>';
            return;
        }

        photos.forEach(photo => {
            const photoItemContainer = document.createElement('div');
            photoItemContainer.className = 'photo-item-container';

            const photoItem = document.createElement('div');
            photoItem.className = 'photo-item';
            photoItem.innerHTML = `<img src="${photo.url}" alt="Photo" loading="lazy">`;
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
        modalImage.src = photo.url;
        let captionText = `Filename: ${photo.filepath.split('/').pop()}`;
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
                showToast('Failed to save name: ' + (result.error || 'Unknown error from server'), 'error');
                showTagInput(faceBox, faceId);
            }
        })
        .catch(error => {
            console.error('Error assigning name:', error);
            showToast('Failed to save name. See browser console for details.', 'error');
            showTagInput(faceBox, faceId);
        });
    }

    async function unassignNameFromFace(faceId, faceBox) {
        const confirmed = await showConfirm('Are you sure you want to remove this name?');
        if (!confirmed) {
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
                showToast('Failed to unassign name: ' + (result.error || 'Unknown error'), 'error');
            }
        })
        .catch(error => {
            console.error('Error unassigning name:', error);
            showToast('An error occurred while unassigning the name. Please check the console.', 'error');
        });
    }

    async function deletePhoto(photoId, elementToRemove) {
        const confirmed = await showConfirm('Are you sure you want to delete this photo? This action cannot be undone.');
        if (!confirmed) {
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
                showToast('Failed to delete photo: ' + result.message, 'error');
            }
        })
        .catch(error => {
            console.error('Error deleting photo:', error);
            showToast('An error occurred while deleting the photo.', 'error');
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
                        // Reset and reload for infinite scroll
                        currentPage = 1;
                        allPhotosLoaded = false;
                        photoGrid.innerHTML = '';
                        loadPhotos();
                        loadPersonNames();
                    } else {
                        showToast('Upload failed: ' + (data.error || 'Unknown error'), 'error');
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
                .then(photos => {
                    if (photos.error) {
                        showToast('Search failed: ' + photos.error, 'error');
                    } else {
                        allPhotosLoaded = true; // Disable infinite scroll for search results
                        photoGrid.innerHTML = '';
                        displayPhotos(photos);
                        clearSearchBtn.style.display = 'block';
                    }
                })
                .catch(error => console.error('Error searching by face:', error));
        }
    });

    nameSearchForm.addEventListener('submit', (e) => {
        e.preventDefault();
        const name = nameSearchInput.value;
        if (!name) return;
        fetch(`/api/search_by_name?name=${encodeURIComponent(name)}`)
            .then(response => response.json())
            .then(photos => {
                if (photos.error) {
                    showToast('Search failed: ' + photos.error, 'error');
                } else {
                    allPhotosLoaded = true; // Disable infinite scroll for search results
                    photoGrid.innerHTML = '';
                    displayPhotos(photos);
                    clearSearchBtn.style.display = 'block';
                }
            })
            .catch(error => console.error('Error searching by name:', error));
    });

    clearSearchBtn.addEventListener('click', () => {
        nameSearchInput.value = '';
        searchInput.value = '';
        clearSearchBtn.style.display = 'none';
        
        // Reset state for infinite scrolling
        currentPage = 1;
        allPhotosLoaded = false;
        photoGrid.innerHTML = '';
        loadPhotos();
    });


});
