let modelReady = false;

// Check model status on page load
document.addEventListener('DOMContentLoaded', async () => {
    await checkModelStatus();
    setupEventListeners();
});

async function checkModelStatus() {
    try {
        const response = await fetch('http://localhost:5678/api/status');
        if (response.ok) {
            const data = await response.json();
            if (data.status === 'ready') {
                modelReady = true;
                console.log('Model ready:', data);
            }
        }
    } catch (error) {
        console.warn('Could not connect to backend:', error);
    }
}

function setupEventListeners() {
    const uploadBox = document.getElementById('uploadBox');
    const fileInput = document.getElementById('fileInput');
    const browseBtn = document.getElementById('browseBtn');
    const resetBtn = document.getElementById('resetBtn');

    // Browse button
    browseBtn.addEventListener('click', () => fileInput.click());

    // File input change
    fileInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) {
            handleImageUpload(file);
        }
    });

    // Drag and drop
    uploadBox.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadBox.classList.add('dragover');
    });

    uploadBox.addEventListener('dragleave', () => {
        uploadBox.classList.remove('dragover');
    });

    uploadBox.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadBox.classList.remove('dragover');
        const file = e.dataTransfer.files[0];
        if (file && file.type.startsWith('image/')) {
            handleImageUpload(file);
        } else {
            showError('Please drop an image file');
        }
    });

    // Reset button
    resetBtn.addEventListener('click', () => {
        resetUI();
    });
}

async function handleImageUpload(file) {
    if (!modelReady) {
        showError('Model is not loaded. Make sure the backend server is running.');
        return;
    }

    // Show loading
    showLoading();

    try {
        // Create FormData for file upload
        const formData = new FormData();
        formData.append('image', file);

        // Send to backend (port 5678)
        const response = await fetch('http://localhost:5678/api/predict', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const error = await response.json();
            showError(error.error || 'Failed to process image');
            hideLoading();
            return;
        }

        const result = await response.json();
        displayResults(file, result);
        hideLoading();

    } catch (error) {
        console.error('Error:', error);
        showError('Error processing image: ' + error.message);
        hideLoading();
    }
}

function displayResults(file, result) {
    // Show preview
    const reader = new FileReader();
    reader.onload = (e) => {
        document.getElementById('previewImage').src = e.target.result;
    };
    reader.readAsDataURL(file);

    // Set results
    document.getElementById('equationOutput').textContent = result.equation || '-';
    document.getElementById('expressionOutput').textContent = result.expression || '-';
    document.getElementById('answerOutput').textContent = result.answer !== null ? result.answer : '-';

    // Show results, hide upload
    hideError();
    document.getElementById('uploadBox').style.display = 'none';
    document.getElementById('results').classList.remove('hidden');
}

function showLoading() {
    document.getElementById('loading').classList.remove('hidden');
}

function hideLoading() {
    document.getElementById('loading').classList.add('hidden');
}

function showError(message) {
    document.getElementById('errorMsg').textContent = message;
    document.getElementById('error').classList.remove('hidden');
}

function hideError() {
    document.getElementById('error').classList.add('hidden');
}

function resetUI() {
    document.getElementById('fileInput').value = '';
    document.getElementById('uploadBox').style.display = '';
    document.getElementById('results').classList.add('hidden');
    hideError();
}
