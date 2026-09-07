// Global State Variables
let selectedFile = null;
let latestResult = null;
let isWebcamRunning = false;
let webcamStream = null;
let webcamInterval = null;
let lastFrameTime = performance.now();

// Tab Switcher
function switchTab(tabId) {
  document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
  document.querySelectorAll('.tab-panel').forEach(panel => panel.classList.remove('active'));

  event.currentTarget.classList.add('active');
  document.getElementById(`tab-${tabId}`).classList.add('active');

  if (tabId !== 'webcam' && isWebcamRunning) {
    stopWebcam();
  }
}

// Drag and Drop Event Setup
const dropZone = document.getElementById('dropZone');
if (dropZone) {
  ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
    dropZone.addEventListener(eventName, preventDefaults, false);
  });

  function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
  }

  ['dragenter', 'dragover'].forEach(eventName => {
    dropZone.addEventListener(eventName, () => dropZone.classList.add('dragover'), false);
  });

  ['dragleave', 'drop'].forEach(eventName => {
    dropZone.addEventListener(eventName, () => dropZone.classList.remove('dragover'), false);
  });

  dropZone.addEventListener('drop', (e) => {
    const dt = e.dataTransfer;
    const files = dt.files;
    if (files.length > 0) {
      selectedFile = files[0];
      previewSelectedFile(selectedFile);
    }
  });
}

function handleFileSelect(event) {
  const files = event.target.files;
  if (files.length > 0) {
    selectedFile = files[0];
    previewSelectedFile(selectedFile);
  }
}

function previewSelectedFile(file) {
  const reader = new FileReader();
  reader.onload = (e) => {
    const img = document.getElementById('annotatedPreview');
    img.src = e.target.result;
    img.style.display = 'block';
    document.getElementById('previewPlaceholder').style.display = 'none';

    // Also update studio preview if available
    const studioImg = document.getElementById('studioPreview');
    if (studioImg) {
      studioImg.src = e.target.result;
      studioImg.style.display = 'block';
      const studioPlaceholder = document.getElementById('studioPlaceholder');
      if (studioPlaceholder) studioPlaceholder.style.display = 'none';
    }
  };
  reader.readAsDataURL(file);
}

// Process Uploaded Image via API
async function processUploadedImage() {
  if (!selectedFile) {
    alert("Please select or drop an image file first!");
    return;
  }

  const formData = new FormData();
  formData.append('file', selectedFile);
  formData.append('conf_threshold', document.getElementById('confSlider').value);
  formData.append('gaussian_blur', document.getElementById('blurSlider') ? document.getElementById('blurSlider').value : 0);
  formData.append('clahe_contrast', document.getElementById('checkClahe') ? document.getElementById('checkClahe').checked : false);
  formData.append('contrast', document.getElementById('contrastSlider') ? document.getElementById('contrastSlider').value : 1.0);
  formData.append('brightness', document.getElementById('brightSlider') ? document.getElementById('brightSlider').value : 1.0);
  formData.append('sharpen', document.getElementById('checkSharpen') ? document.getElementById('checkSharpen').checked : false);

  try {
    const response = await fetch('/api/detect/image', {
      method: 'POST',
      body: formData
    });

    if (!response.ok) {
      throw new Error(`API Error: ${response.statusText}`);
    }

    const data = await response.json();
    latestResult = data;
    renderResults(data);
  } catch (err) {
    console.error("Pipeline request error:", err);
    alert(`Failed to run pipeline: ${err.message}`);
  }
}

// Render Results on UI
function renderResults(data) {
  // Update Annotated Image Preview
  if (data.annotated_image_base64) {
    const img = document.getElementById('annotatedPreview');
    img.src = data.annotated_image_base64;
    img.style.display = 'block';

    const studioImg = document.getElementById('studioPreview');
    if (studioImg) {
      studioImg.src = data.annotated_image_base64;
      studioImg.style.display = 'block';
    }
  }

  // Metrics
  document.getElementById('metricCount').innerText = data.total_detections;
  document.getElementById('metricInference').innerText = `${data.inference_time_ms} ms`;
  document.getElementById('metricTotal').innerText = `${data.total_time_ms} ms`;

  // Render Detections List
  const listContainer = document.getElementById('detectionsList');
  if (data.detections.length === 0) {
    listContainer.innerHTML = `<p style="color: var(--text-dim);">No objects detected at confidence threshold.</p>`;
  } else {
    listContainer.innerHTML = data.detections.map(det => `
      <div class="detection-card">
        <div>
          <span class="det-class">#${det.id} ${det.class_name}</span>
          <div style="font-size: 0.8rem; color: var(--text-muted); margin-top: 0.2rem;">
            Box [x1:${Math.round(det.bbox.x1)}, y1:${Math.round(det.bbox.y1)}, x2:${Math.round(det.bbox.x2)}, y2:${Math.round(det.bbox.y2)}] • 
            Norm [cx:${det.bbox.x_center_norm.toFixed(3)}, cy:${det.bbox.y_center_norm.toFixed(3)}, w:${det.bbox.width_norm.toFixed(3)}, h:${det.bbox.height_norm.toFixed(3)}]
          </div>
        </div>
        <span class="det-badge">${(det.confidence * 100).toFixed(1)}%</span>
      </div>
    `).join('');
  }

  // Update JSON Output Inspector
  document.getElementById('jsonOutputBlock').innerText = JSON.stringify(data, null, 2);
}

// Live Webcam Feed Implementation
async function startWebcam() {
  const video = document.getElementById('webcamVideo');
  const placeholder = document.getElementById('webcamPlaceholder');

  try {
    webcamStream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
    video.srcObject = webcamStream;
    video.style.display = 'block';
    placeholder.style.display = 'none';

    isWebcamRunning = true;
    document.getElementById('btnStartWebcam').disabled = true;
    document.getElementById('btnStopWebcam').disabled = false;

    // Canvas for hidden frame extraction
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');

    const captureAndDetect = async () => {
      if (!isWebcamRunning) return;

      canvas.width = video.videoWidth || 640;
      canvas.height = video.videoHeight || 480;
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

      const frameBase64 = canvas.toDataURL('image/jpeg', 0.8);
      const confThreshold = parseFloat(document.getElementById('confSlider').value);

      const now = performance.now();
      try {
        const res = await fetch('/api/detect/frame', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            image: frameBase64,
            conf_threshold: confThreshold
          })
        });

        if (res.ok) {
          const data = await res.json();
          const latency = Math.round(performance.now() - now);
          const fps = Math.round(1000 / (performance.now() - lastFrameTime));
          lastFrameTime = performance.now();

          // Overlay annotated image
          const overlay = document.getElementById('webcamAnnotated');
          overlay.src = data.annotated_image_base64;
          overlay.style.display = 'block';
          video.style.display = 'none';

          document.getElementById('webcamFps').innerText = `${fps} FPS`;
          document.getElementById('webcamLatency').innerText = `${latency} ms`;
          document.getElementById('webcamCount').innerText = data.total_detections;
        }
      } catch (err) {
        console.error("Webcam frame error:", err);
      }

      if (isWebcamRunning) {
        setTimeout(captureAndDetect, 50);  // Loop every 50ms (~20 FPS max)
      }
    };

    setTimeout(captureAndDetect, 500);
  } catch (err) {
    alert(`Could not access webcam: ${err.message}`);
    console.error(err);
  }
}

function stopWebcam() {
  isWebcamRunning = false;
  if (webcamStream) {
    webcamStream.getTracks().forEach(track => track.stop());
  }

  document.getElementById('webcamVideo').style.display = 'none';
  document.getElementById('webcamAnnotated').style.display = 'none';
  document.getElementById('webcamPlaceholder').style.display = 'block';
  document.getElementById('btnStartWebcam').disabled = false;
  document.getElementById('btnStopWebcam').disabled = true;
}

// Copy JSON Output to Clipboard
function copyJsonOutput() {
  const jsonText = document.getElementById('jsonOutputBlock').innerText;
  navigator.clipboard.writeText(jsonText).then(() => {
    alert("JSON payload copied to clipboard!");
  }).catch(err => {
    console.error("Could not copy:", err);
  });
}
