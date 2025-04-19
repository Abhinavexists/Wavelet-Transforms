document.addEventListener('DOMContentLoaded', function() {
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');
    const processButton = document.getElementById('processButton');
    const denoisingForm = document.getElementById('denoisingForm');
    const resultContainer = document.querySelector('.result-container');
    const resultImage = document.getElementById('resultImage');
    const loading = document.querySelector('.loading');
    const addNoiseCheckbox = document.getElementById('addNoise');
    const noiseLevelContainer = document.getElementById('noiseLevelContainer');
    const noiseLevelValue = document.getElementById('noiseLevelValue');
    const noiseLevelInput = document.querySelector('input[name="noise_level"]');
    
    let currentFile = null;
    
    // Drag and drop handlers
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    });
    
    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('dragover');
    });
    
    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        const file = e.dataTransfer.files[0];
        handleFile(file);
    });
    
    dropZone.addEventListener('click', () => {
        fileInput.click();
    });
    
    fileInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        handleFile(file);
    });
    
    function handleFile(file) {
        if (file && file.type.startsWith('image/')) {
            currentFile = file;
            const reader = new FileReader();
            reader.onload = (e) => {
                dropZone.innerHTML = `<img src="${e.target.result}" alt="Uploaded image">`;
                processButton.disabled = false;
            };
            reader.readAsDataURL(file);
        }
    }
    
    // Noise level slider
    noiseLevelInput.addEventListener('input', (e) => {
        noiseLevelValue.textContent = e.target.value;
    });
    
    addNoiseCheckbox.addEventListener('change', (e) => {
        noiseLevelContainer.style.display = e.target.checked ? 'block' : 'none';
    });
    
    // Form submission
    denoisingForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        if (!currentFile) return;
        
        const formData = new FormData();
        formData.append('image', currentFile);
        
        // Add form parameters
        const formElements = denoisingForm.elements;
        for (let element of formElements) {
            if (element.name) {
                if (element.type === 'checkbox') {
                    formData.append(element.name, element.checked);
                } else {
                    formData.append(element.name, element.value);
                }
            }
        }
        
        // Show loading
        loading.style.display = 'block';
        resultContainer.style.display = 'none';
        
        try {
            const response = await fetch('/process', {
                method: 'POST',
                body: formData
            });
            
            const data = await response.json();
            
            if (response.ok) {
                resultImage.src = `data:image/png;base64,${data.image}`;
                resultContainer.style.display = 'block';
                
                // Update download button
                const downloadBtn = document.getElementById('downloadBtn');
                if (downloadBtn) {
                    downloadBtn.href = resultImage.src;
                    downloadBtn.download = `denoised_${currentFile.name}`;
                }
            } else {
                alert(data.error || 'An error occurred');
            }
        } catch (error) {
            alert('An error occurred while processing the image');
        } finally {
            loading.style.display = 'none';
        }
    });
}); 