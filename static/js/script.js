document.addEventListener('DOMContentLoaded', function () {
  const dropArea = document.getElementById('dropArea');
  const fileInput = document.getElementById('fileInput');
  const browseBtn = document.getElementById('browseBtn');
  const previewImage = document.getElementById('previewImage');
  const resultsSection = document.getElementById('resultsSection');
  const diseaseName = document.getElementById('diseaseName');
  const confidenceFill = document.getElementById('confidenceFill');
  const confidenceValue = document.getElementById('confidenceValue');
  const diseaseDescription = document.getElementById('diseaseDescription');
  const recommendationsList = document.getElementById('recommendationsList');

  // Prevent default drag behaviors
  ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
    dropArea.addEventListener(eventName, preventDefaults, false);
  });

  function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
  }

  // Highlight drop area
  ['dragenter', 'dragover'].forEach(eventName => {
    dropArea.addEventListener(eventName, () => dropArea.classList.add('highlight'), false);
  });

  ['dragleave', 'drop'].forEach(eventName => {
    dropArea.addEventListener(eventName, () => dropArea.classList.remove('highlight'), false);
  });

  dropArea.addEventListener('drop', handleDrop, false);
  browseBtn.addEventListener('click', () => fileInput.click());
  fileInput.addEventListener('change', handleFiles);

  function handleDrop(e) {
    const dt = e.dataTransfer;
    const files = dt.files;
    handleFiles({ target: { files } });
  }

  function handleFiles(e) {
    const files = e.target.files;
    if (files.length) {
      const file = files[0];
      if (file.type.match(/image\/(jpeg|jpg|png)/)) {
        displayPreview(file);
        analyzeImage(file);
      } else {
        alert('Please upload a JPG, JPEG or PNG image file.');
      }
    }
  }

  function displayPreview(file) {
    const reader = new FileReader();
    reader.onload = function (e) {
      previewImage.src = e.target.result;
      resultsSection.classList.remove('hidden');
    };
    reader.readAsDataURL(file);
  }

  function analyzeImage(file) {
    diseaseName.textContent = 'Analyzing...';
    confidenceFill.style.width = '0%';
    confidenceValue.textContent = '0%';
    diseaseDescription.textContent = 'Processing your image...';
    recommendationsList.innerHTML = '<li>Processing recommendations...</li>';

    const formData = new FormData();
    formData.append("file", file);

    fetch("/predict", {
      method: "POST",
      body: formData
    })
      .then(res => res.json())
      .then(result => {
        if (result.error) {
          diseaseName.textContent = "Error: " + result.error;
          return;
        }
        diseaseName.textContent = result.disease;
        confidenceFill.style.width = `${result.confidence}%`;
        confidenceValue.textContent = `${result.confidence}%`;
        diseaseDescription.textContent = result.cause;

        recommendationsList.innerHTML = '';
        result.recommendations.forEach(rec => {
          const li = document.createElement('li');
          li.textContent = rec;
          recommendationsList.appendChild(li);
        });
      })
      .catch(err => {
        console.error(err);
        diseaseName.textContent = "Error connecting to server";
      });
  }
});
