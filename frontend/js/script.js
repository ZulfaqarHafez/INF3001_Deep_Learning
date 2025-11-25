class PPEComplianceSystem {
  constructor() {
    this.API_BASE = 'http://127.0.0.1:8000';
    this.threshold = 120;
    this.cameraActive = false;
    this.cameraStream = null;
    this.animationFrame = null;
    this.fpsCounter = { frames: 0, lastTime: Date.now(), fps: 0 };
    this.stats = { totalAnalyzed: 0, compliantCount: 0 };
    
    this.initDOM();
    this.bindEvents();
    this.updateStats();
  }

  initDOM() {
    // Navigation
    this.navItems = document.querySelectorAll('.nav-item');
    this.uploadView = document.getElementById('uploadView');
    this.liveView = document.getElementById('liveView');
    
    // Upload elements
    this.uploadZone = document.getElementById('uploadZone');
    this.uploadPlaceholder = document.getElementById('uploadPlaceholder');
    this.imagePreview = document.getElementById('imagePreview');
    this.previewImage = document.getElementById('previewImage');
    this.fileInput = document.getElementById('fileInput');
    this.browseBtn = document.getElementById('browseBtn');
    this.removeImageBtn = document.getElementById('removeImage');
    this.analyzeBtn = document.getElementById('analyzeBtn');
    this.uploadProgress = document.getElementById('uploadProgress');
    this.uploadProgressFill = document.getElementById('uploadProgressFill');
    this.uploadProgressText = document.getElementById('uploadProgressText');
    this.uploadResults = document.getElementById('uploadResults');
    this.uploadStatusDot = document.getElementById('uploadStatusDot');
    this.uploadStatusText = document.getElementById('uploadStatusText');
    
    // Live camera elements
    this.cameraVideo = document.getElementById('cameraVideo');
    this.cameraCanvas = document.getElementById('cameraCanvas');
    this.cameraPlaceholder = document.getElementById('cameraPlaceholder');
    this.startCameraBtn = document.getElementById('startCamera');
    this.stopCameraBtn = document.getElementById('stopCamera');
    this.liveStatusDot = document.getElementById('liveStatusDot');
    this.liveStatusText = document.getElementById('liveStatusText');
    this.liveCompliant = document.getElementById('liveCompliant');
    this.liveNonCompliant = document.getElementById('liveNonCompliant');
    this.liveTotalPeople = document.getElementById('liveTotalPeople');
    this.liveDetectionsList = document.getElementById('liveDetectionsList');
    this.fpsDisplay = document.getElementById('fpsDisplay');
    this.processingTimeDisplay = document.getElementById('processingTime');
    this.thresholdDisplay = document.getElementById('thresholdDisplay');
    this.increaseThresholdBtn = document.getElementById('increaseThreshold');
    this.decreaseThresholdBtn = document.getElementById('decreaseThreshold');
    
    // Stats
    this.totalAnalyzedEl = document.getElementById('totalAnalyzed');
    this.complianceRateEl = document.getElementById('complianceRate');
  }

  bindEvents() {
    // Navigation
    this.navItems.forEach(item => {
      item.addEventListener('click', () => this.switchView(item.dataset.view));
    });
    
    // Upload events
    this.uploadZone.addEventListener('click', () => {
      if (!this.hasUploadedImage()) this.fileInput.click();
    });
    
    this.browseBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      this.fileInput.click();
    });
    
    this.fileInput.addEventListener('change', (e) => {
      if (e.target.files?.length) this.handleFileUpload(e.target.files[0]);
    });
    
    this.removeImageBtn.addEventListener('click', () => this.resetUpload());
    this.analyzeBtn.addEventListener('click', () => this.analyzeImage());
    
    // Drag and drop
    this.uploadZone.addEventListener('dragover', (e) => {
      e.preventDefault();
      this.uploadZone.style.borderColor = 'var(--primary)';
    });
    
    this.uploadZone.addEventListener('dragleave', () => {
      this.uploadZone.style.borderColor = '';
    });
    
    this.uploadZone.addEventListener('drop', (e) => {
      e.preventDefault();
      this.uploadZone.style.borderColor = '';
      if (e.dataTransfer.files?.length) this.handleFileUpload(e.dataTransfer.files[0]);
    });
    
    // Camera events
    this.startCameraBtn.addEventListener('click', () => this.startCamera());
    this.stopCameraBtn.addEventListener('click', () => this.stopCamera());
    
    // Threshold controls
    this.increaseThresholdBtn.addEventListener('click', () => this.adjustThreshold(10));
    this.decreaseThresholdBtn.addEventListener('click', () => this.adjustThreshold(-10));
  }

  // ===== View Management =====
  switchView(view) {
    this.navItems.forEach(item => item.classList.remove('active'));
    document.querySelector(`[data-view="${view}"]`).classList.add('active');
    
    this.uploadView.classList.remove('active');
    this.liveView.classList.remove('active');
    
    if (view === 'upload') {
      this.uploadView.classList.add('active');
      if (this.cameraActive) this.stopCamera();
    } else if (view === 'live') {
      this.liveView.classList.add('active');
    }
  }

  // ===== Image Upload =====
  hasUploadedImage() {
    return this.imagePreview.style.display !== 'none';
  }

  handleFileUpload(file) {
    if (!file || !file.type.startsWith('image/')) {
      alert('Please select a valid image file.');
      return;
    }

    const reader = new FileReader();
    reader.onload = (e) => {
      this.previewImage.src = e.target.result;
      this.previewImage.onload = () => {
        this.showImagePreview();
        this.currentFile = file;
      };
    };
    reader.readAsDataURL(file);
  }

  showImagePreview() {
    this.uploadPlaceholder.style.display = 'none';
    this.imagePreview.style.display = 'block';
    this.analyzeBtn.disabled = false;
    this.setUploadStatus('Ready to analyze', 'active');
  }

  resetUpload() {
    this.uploadPlaceholder.style.display = 'block';
    this.imagePreview.style.display = 'none';
    this.fileInput.value = '';
    this.currentFile = null;
    this.analyzeBtn.disabled = true;
    this.uploadResults.innerHTML = `
      <div class="empty-state">
        <svg width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="currentColor" opacity="0.3">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"/>
        </svg>
        <p>No analysis yet</p>
        <span>Upload an image to begin</span>
      </div>
    `;
    this.setUploadStatus('Ready', 'active');
  }

  async analyzeImage() {
    if (!this.currentFile) return;

    this.setUploadStatus('Analyzing...', 'loading');
    this.analyzeBtn.disabled = true;
    this.showUploadProgress(20);

    const formData = new FormData();
    formData.append('file', this.currentFile);

    try {
      const response = await fetch(`${this.API_BASE}/detect-helmet`, {
        method: 'POST',
        body: formData
      });

      this.showUploadProgress(80);

      if (!response.ok) {
        throw new Error('Analysis failed');
      }

      const data = await response.json();
      this.showUploadProgress(100);

      setTimeout(() => {
        this.displayUploadResults(data);
        this.setUploadStatus('Analysis complete', 'active');
        this.analyzeBtn.disabled = false;
        this.hideUploadProgress();
        this.updateStatsFromResult(data);
      }, 500);

    } catch (error) {
      console.error('Analysis error:', error);
      this.setUploadStatus('Analysis failed', 'error');
      this.analyzeBtn.disabled = false;
      this.hideUploadProgress();
      
      this.uploadResults.innerHTML = `
        <div class="empty-state">
          <svg width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="currentColor" opacity="0.3">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"/>
          </svg>
          <p>Analysis failed</p>
          <span>Please check your connection and try again</span>
        </div>
      `;
    }
  }

  displayUploadResults(data) {
    const { total_people, compliant_people, non_compliant_people, compliance_rate, person_analyses } = data;
    
    const isCompliant = compliant_people === total_people && total_people > 0;
    
    this.uploadResults.innerHTML = `
      <div class="compliance-card ${isCompliant ? 'compliant' : 'non-compliant'}">
        <div class="compliance-header">
          <div class="compliance-icon">
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="3">
              ${isCompliant 
                ? '<path stroke-linecap="round" stroke-linejoin="round" d="M5 13l4 4L19 7"/>'
                : '<path stroke-linecap="round" stroke-linejoin="round" d="M6 18L18 6M6 6l12 12"/>'}
            </svg>
          </div>
          <div>
            <div class="compliance-title">${isCompliant ? 'COMPLIANT' : 'NON-COMPLIANT'}</div>
            <div style="color: var(--text-secondary); font-size: 0.875rem;">
              ${isCompliant ? 'All workers wearing helmets' : 'Safety violation detected'}
            </div>
          </div>
        </div>
        
        <div class="compliance-details">
          <div class="detail-item">
            <div class="detail-label">Total People</div>
            <div class="detail-value">${total_people}</div>
          </div>
          <div class="detail-item">
            <div class="detail-label">Compliant</div>
            <div class="detail-value" style="color: var(--success);">${compliant_people}</div>
          </div>
          <div class="detail-item">
            <div class="detail-label">Non-Compliant</div>
            <div class="detail-value" style="color: var(--danger);">${non_compliant_people}</div>
          </div>
          <div class="detail-item">
            <div class="detail-label">Compliance Rate</div>
            <div class="detail-value">${(compliance_rate * 100).toFixed(1)}%</div>
          </div>
        </div>
      </div>
      
      ${person_analyses && person_analyses.length > 0 ? `
        <div class="person-list">
          ${person_analyses.map((person, i) => `
            <div class="person-card ${person.overall_compliant ? 'compliant' : 'non-compliant'}">
              <div class="person-header">
                <span class="person-id">Person ${i + 1}</span>
                <span class="person-status">${person.overall_compliant ? 'COMPLIANT' : 'NON-COMPLIANT'}</span>
              </div>
              <div class="person-reason">
                ${person.helmet_status?.reason || 'No helmet information'}
              </div>
            </div>
          `).join('')}
        </div>
      ` : ''}
    `;
  }

  setUploadStatus(text, state) {
    this.uploadStatusText.textContent = text;
    this.uploadStatusDot.className = `status-dot ${state}`;
  }

  showUploadProgress(percent) {
    this.uploadProgress.style.display = 'block';
    this.uploadProgressFill.style.width = `${percent}%`;
    this.uploadProgressText.textContent = `Processing... ${percent}%`;
  }

  hideUploadProgress() {
    setTimeout(() => {
      this.uploadProgress.style.display = 'none';
      this.uploadProgressFill.style.width = '0%';
    }, 300);
  }

  // ===== Live Camera =====
  async startCamera() {
    try {
      this.setLiveStatus('Starting camera...', 'loading');
      
      this.cameraStream = await navigator.mediaDevices.getUserMedia({
        video: { width: 1280, height: 720, facingMode: 'user' }
      });
      
      this.cameraVideo.srcObject = this.cameraStream;
      this.cameraVideo.style.display = 'block';
      this.cameraPlaceholder.style.display = 'none';
      
      this.cameraVideo.onloadedmetadata = () => {
        this.cameraCanvas.width = this.cameraVideo.videoWidth;
        this.cameraCanvas.height = this.cameraVideo.videoHeight;
        
        this.cameraActive = true;
        this.startCameraBtn.disabled = true;
        this.stopCameraBtn.disabled = false;
        
        this.setLiveStatus('Camera active', 'active');
        this.processFrame();
      };
      
    } catch (error) {
      console.error('Camera error:', error);
      this.setLiveStatus('Camera error', 'error');
      alert('Could not access camera. Please check permissions.');
    }
  }

  stopCamera() {
    this.cameraActive = false;
    
    if (this.animationFrame) {
      cancelAnimationFrame(this.animationFrame);
    }
    
    if (this.cameraStream) {
      this.cameraStream.getTracks().forEach(track => track.stop());
    }
    
    this.cameraVideo.style.display = 'none';
    this.cameraPlaceholder.style.display = 'flex';
    
    const ctx = this.cameraCanvas.getContext('2d');
    ctx.clearRect(0, 0, this.cameraCanvas.width, this.cameraCanvas.height);
    
    this.startCameraBtn.disabled = false;
    this.stopCameraBtn.disabled = true;
    
    this.setLiveStatus('Inactive', 'inactive');
    this.resetLiveStats();
  }

  async processFrame() {
    if (!this.cameraActive) return;

    const startTime = performance.now();

    // Capture frame
    const canvas = document.createElement('canvas');
    canvas.width = this.cameraVideo.videoWidth;
    canvas.height = this.cameraVideo.videoHeight;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(this.cameraVideo, 0, 0);

    // Convert to blob
    canvas.toBlob(async (blob) => {
      if (!this.cameraActive) return;

      try {
        const formData = new FormData();
        formData.append('file', blob, 'frame.jpg');
        formData.append('threshold', this.threshold.toString());

        const response = await fetch(`${this.API_BASE}/detect-helmet`, {
          method: 'POST',
          body: formData
        });

        if (response.ok) {
          const data = await response.json();
          this.drawDetections(data);
          this.updateLiveStats(data);
          
          const processingTime = performance.now() - startTime;
          this.processingTimeDisplay.textContent = `${Math.round(processingTime)}ms`;
        }

      } catch (error) {
        console.error('Detection error:', error);
      }

      // Update FPS
      this.updateFPS();

      // Schedule next frame
      if (this.cameraActive) {
        this.animationFrame = requestAnimationFrame(() => this.processFrame());
      }
    }, 'image/jpeg', 0.8);
  }

  drawDetections(data) {
    const ctx = this.cameraCanvas.getContext('2d');
    ctx.clearRect(0, 0, this.cameraCanvas.width, this.cameraCanvas.height);

    if (!data.person_analyses) return;

    const scaleX = this.cameraCanvas.width / this.cameraVideo.videoWidth;
    const scaleY = this.cameraCanvas.height / this.cameraVideo.videoHeight;

    data.person_analyses.forEach((person) => {
      const [x1, y1, x2, y2] = person.person_box;
      const isCompliant = person.overall_compliant;

      // Draw box
      ctx.strokeStyle = isCompliant ? '#10b981' : '#ef4444';
      ctx.lineWidth = 3;
      ctx.strokeRect(x1 * scaleX, y1 * scaleY, (x2 - x1) * scaleX, (y2 - y1) * scaleY);

      // Draw label
      const label = isCompliant ? 'COMPLIANT' : 'NON-COMPLIANT';
      ctx.font = 'bold 16px Inter';
      const textWidth = ctx.measureText(label).width;
      
      ctx.fillStyle = isCompliant ? '#10b981' : '#ef4444';
      ctx.fillRect(x1 * scaleX, Math.max(0, y1 * scaleY - 30), textWidth + 16, 30);
      
      ctx.fillStyle = 'white';
      ctx.fillText(label, x1 * scaleX + 8, Math.max(20, y1 * scaleY - 8));

      // Draw head position if available
      if (person.head_detected && person.head_position) {
        const [hx, hy] = person.head_position;
        ctx.fillStyle = '#06b6d4';
        ctx.beginPath();
        ctx.arc(hx * scaleX, hy * scaleY, 6, 0, 2 * Math.PI);
        ctx.fill();
        ctx.strokeStyle = 'white';
        ctx.lineWidth = 2;
        ctx.stroke();
      }
    });
  }

  updateLiveStats(data) {
    this.liveTotalPeople.textContent = data.total_people || 0;
    this.liveCompliant.textContent = data.compliant_people || 0;
    this.liveNonCompliant.textContent = data.non_compliant_people || 0;

    // Update detections list
    if (data.person_analyses && data.person_analyses.length > 0) {
      this.liveDetectionsList.innerHTML = data.person_analyses.map((person, i) => `
        <div class="detection-item ${person.overall_compliant ? 'compliant' : 'non-compliant'}">
          <div class="detection-info">
            <span class="detection-label">Person ${i + 1}</span>
          </div>
          <span class="detection-confidence">${person.overall_compliant ? '✓ Compliant' : '✗ Non-compliant'}</span>
        </div>
      `).join('');
    } else {
      this.liveDetectionsList.innerHTML = '<div class="empty-state-small"><p>No people detected</p></div>';
    }
  }

  resetLiveStats() {
    this.liveTotalPeople.textContent = '0';
    this.liveCompliant.textContent = '0';
    this.liveNonCompliant.textContent = '0';
    this.liveDetectionsList.innerHTML = '<div class="empty-state-small"><p>No detections</p></div>';
  }

  updateFPS() {
    this.fpsCounter.frames++;
    const now = Date.now();
    const elapsed = now - this.fpsCounter.lastTime;

    if (elapsed >= 1000) {
      this.fpsCounter.fps = Math.round(this.fpsCounter.frames / (elapsed / 1000));
      this.fpsDisplay.textContent = this.fpsCounter.fps;
      this.fpsCounter.frames = 0;
      this.fpsCounter.lastTime = now;
    }
  }

  setLiveStatus(text, state) {
    this.liveStatusText.textContent = text;
    this.liveStatusDot.className = `status-dot ${state}`;
  }

  adjustThreshold(amount) {
    this.threshold = Math.max(40, Math.min(300, this.threshold + amount));
    this.thresholdDisplay.textContent = `${this.threshold}px`;
  }

  // ===== Stats Management =====
  updateStatsFromResult(data) {
    this.stats.totalAnalyzed += data.total_people || 0;
    this.stats.compliantCount += data.compliant_people || 0;
    this.updateStats();
  }

  updateStats() {
    this.totalAnalyzedEl.textContent = this.stats.totalAnalyzed;
    
    if (this.stats.totalAnalyzed > 0) {
      const rate = (this.stats.compliantCount / this.stats.totalAnalyzed * 100).toFixed(1);
      this.complianceRateEl.textContent = `${rate}%`;
    } else {
      this.complianceRateEl.textContent = '--%';
    }
  }
}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
  new PPEComplianceSystem();
});