/**
 * PPE Helmet Compliance System
 * Frontend JavaScript with Supabase History Integration
 * 
 * Author: Zulfaqar
 * Project: INF3001 Deep Learning - PPE Detection
 */

class PPEComplianceSystem {
  constructor() {
    // API Configuration
    this.API_BASE = 'http://127.0.0.1:8000';
    
    // Detection settings
    this.threshold = 120;
    
    // Camera state
    this.cameraActive = false;
    this.cameraStream = null;
    this.animationFrame = null;
    this.fpsCounter = { frames: 0, lastTime: Date.now(), fps: 0 };
    
    // Session stats
    this.stats = { totalAnalyzed: 0, compliantCount: 0 };
    
    // History state
    this.historyOffset = 0;
    this.historyLimit = 12;
    this.currentFilter = 'all';
    this.historyRecords = [];
    
    // Current file for upload
    this.currentFile = null;
    
    // Initialize
    this.initDOM();
    this.bindEvents();
    this.checkAPIConnection();
    this.updateStats();
  }

  // =========================================================================
  // DOM INITIALIZATION
  // =========================================================================
  
  initDOM() {
    this.clearAllHistoryBtn = document.getElementById('clearAllHistory');
    // Navigation
    this.navItems = document.querySelectorAll('.nav-item');
    this.uploadView = document.getElementById('uploadView');
    this.liveView = document.getElementById('liveView');
    this.historyView = document.getElementById('historyView');
    
    // Upload elements
    this.uploadZone = document.getElementById('uploadZone');
    this.uploadPlaceholder = document.getElementById('uploadPlaceholder');
    this.imagePreview = document.getElementById('imagePreview');
    this.previewImage = document.getElementById('previewImage');
    this.fileInput = document.getElementById('fileInput');
    this.browseBtn = document.getElementById('browseBtn');
    this.removeImageBtn = document.getElementById('removeImage');
    this.analyzeBtn = document.getElementById('analyzeBtn');
    this.saveToHistoryCheckbox = document.getElementById('saveToHistory');
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
    
    // History elements
    this.historyGrid = document.getElementById('historyGrid');
    this.refreshHistoryBtn = document.getElementById('refreshHistory');
    this.filterBtns = document.querySelectorAll('.filter-btn');
    this.prevPageBtn = document.getElementById('prevPage');
    this.nextPageBtn = document.getElementById('nextPage');
    this.pageInfo = document.getElementById('pageInfo');
    this.historyPagination = document.getElementById('historyPagination');
    
    // History stats
    this.statsTotalScans = document.getElementById('statsTotalScans');
    this.statsTotalCompliant = document.getElementById('statsTotalCompliant');
    this.statsTotalViolations = document.getElementById('statsTotalViolations');
    this.statsOverallRate = document.getElementById('statsOverallRate');
    
    // Modal
    this.detailModal = document.getElementById('detailModal');
    this.modalBody = document.getElementById('modalBody');
    this.closeModalBtn = document.getElementById('closeModal');
    
    // Sidebar stats
    this.totalAnalyzedEl = document.getElementById('totalAnalyzed');
    this.complianceRateEl = document.getElementById('complianceRate');
    
    // Connection status
    this.apiStatusDot = document.getElementById('apiStatusDot');
    this.apiStatusText = document.getElementById('apiStatusText');
  }

  // =========================================================================
  // EVENT BINDING
  // =========================================================================
  
  bindEvents() {
    this.clearAllHistoryBtn?.addEventListener('click', () => this.clearAllHistory());
    // Navigation
    this.navItems.forEach(item => {
      item.addEventListener('click', () => this.switchView(item.dataset.view));
    });
    
    // Upload events
    this.uploadZone.addEventListener('click', (e) => {
      if (!this.hasUploadedImage() && e.target !== this.browseBtn) {
        this.fileInput.click();
      }
    });
    
    this.browseBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      this.fileInput.click();
    });
    
    this.fileInput.addEventListener('change', (e) => {
      if (e.target.files?.length) this.handleFileUpload(e.target.files[0]);
    });
    
    this.removeImageBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      this.resetUpload();
    });
    
    this.analyzeBtn.addEventListener('click', () => this.analyzeImage());
    
    // Drag and drop
    this.uploadZone.addEventListener('dragover', (e) => {
      e.preventDefault();
      this.uploadZone.classList.add('dragover');
    });
    
    this.uploadZone.addEventListener('dragleave', () => {
      this.uploadZone.classList.remove('dragover');
    });
    
    this.uploadZone.addEventListener('drop', (e) => {
      e.preventDefault();
      this.uploadZone.classList.remove('dragover');
      if (e.dataTransfer.files?.length) this.handleFileUpload(e.dataTransfer.files[0]);
    });
    
    // Camera events
    this.startCameraBtn.addEventListener('click', () => this.startCamera());
    this.stopCameraBtn.addEventListener('click', () => this.stopCamera());
    
    // Threshold controls
    this.increaseThresholdBtn.addEventListener('click', () => this.adjustThreshold(10));
    this.decreaseThresholdBtn.addEventListener('click', () => this.adjustThreshold(-10));
    
    // History events
    this.refreshHistoryBtn?.addEventListener('click', () => {
      this.loadHistory();
      this.loadHistoryStats();
    });
    
    this.filterBtns?.forEach(btn => {
      btn.addEventListener('click', () => this.setFilter(btn.dataset.filter));
    });
    
    this.prevPageBtn?.addEventListener('click', () => this.changePage(-1));
    this.nextPageBtn?.addEventListener('click', () => this.changePage(1));
    
    // Modal events
    this.closeModalBtn?.addEventListener('click', () => this.closeModal());
    this.detailModal?.addEventListener('click', (e) => {
      if (e.target === this.detailModal) this.closeModal();
    });
    
    // Keyboard events
    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape' && this.detailModal.style.display !== 'none') {
        this.closeModal();
      }
    });
  }

  // =========================================================================
  // API CONNECTION
  // =========================================================================
  
  async checkAPIConnection() {
    try {
      const response = await fetch(`${this.API_BASE}/health`);
      if (response.ok) {
        const data = await response.json();
        this.setAPIStatus('connected', `API Connected`);
        console.log('API Health:', data);
      } else {
        this.setAPIStatus('error', 'API Error');
      }
    } catch (error) {
      console.error('API Connection Error:', error);
      this.setAPIStatus('error', 'API Offline');
    }
  }
  
  setAPIStatus(status, text) {
    if (this.apiStatusDot) {
      this.apiStatusDot.className = `status-dot ${status === 'connected' ? 'active' : 'error'}`;
    }
    if (this.apiStatusText) {
      this.apiStatusText.textContent = text;
    }
  }

  // =========================================================================
  // VIEW MANAGEMENT
  // =========================================================================
  
  switchView(view) {
    // Update navigation
    this.navItems.forEach(item => item.classList.remove('active'));
    document.querySelector(`[data-view="${view}"]`)?.classList.add('active');
    
    // Hide all views
    this.uploadView.classList.remove('active');
    this.liveView.classList.remove('active');
    this.historyView.classList.remove('active');
    
    // Show selected view
    if (view === 'upload') {
      this.uploadView.classList.add('active');
      if (this.cameraActive) this.stopCamera();
    } else if (view === 'live') {
      this.liveView.classList.add('active');
    } else if (view === 'history') {
      this.historyView.classList.add('active');
      this.loadHistory();
      this.loadHistoryStats();
    }
  }

  // =========================================================================
  // IMAGE UPLOAD
  // =========================================================================
  
  hasUploadedImage() {
    return this.imagePreview.style.display !== 'none';
  }

  handleFileUpload(file) {
    if (!file || !file.type.startsWith('image/')) {
      this.showNotification('Please select a valid image file.', 'error');
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
    this.imagePreview.style.display = 'flex';
    this.analyzeBtn.disabled = false;
    this.setUploadStatus('Ready to analyze', 'active');
  }

  resetUpload() {
    this.uploadPlaceholder.style.display = 'flex';
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
    formData.append('save_to_history', this.saveToHistoryCheckbox?.checked ?? true);

    try {
      const response = await fetch(`${this.API_BASE}/detect-helmet`, {
        method: 'POST',
        body: formData
      });

      this.showUploadProgress(80);

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Analysis failed');
      }

      const data = await response.json();
      this.showUploadProgress(100);

      setTimeout(() => {
        this.displayUploadResults(data);
        this.setUploadStatus('Analysis complete', 'active');
        this.analyzeBtn.disabled = false;
        this.hideUploadProgress();
        this.updateStatsFromResult(data);
        
        if (data.saved_to_history) {
          this.showNotification('Detection saved to history', 'success');
        }
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
          <span>${error.message || 'Please check your connection and try again'}</span>
        </div>
      `;
    }
  }

  async drawDetectionsOnImage(data) {
    return new Promise((resolve) => {
      const img = new Image();
      img.onload = () => {
        const canvas = document.createElement('canvas');
        canvas.width = img.width;
        canvas.height = img.height;
        const ctx = canvas.getContext('2d');
        
        // Draw original image
        ctx.drawImage(img, 0, 0);
        
        // Draw detections
        if (data.person_analyses && data.person_analyses.length > 0) {
          data.person_analyses.forEach((person, idx) => {
            const [x1, y1, x2, y2] = person.person_box;
            const isCompliant = person.overall_compliant;
            
            // Draw person bounding box
            ctx.strokeStyle = isCompliant ? '#10b981' : '#ef4444';
            ctx.lineWidth = Math.max(4, img.width / 300);
            ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
            
            // Draw label background
            const label = isCompliant ? 'COMPLIANT' : 'NON-COMPLIANT';
            const fontSize = Math.max(16, img.width / 60);
            ctx.font = `bold ${fontSize}px Inter, sans-serif`;
            const textWidth = ctx.measureText(label).width;
            
            ctx.fillStyle = isCompliant ? '#10b981' : '#ef4444';
            ctx.fillRect(x1, Math.max(0, y1 - fontSize - 15), textWidth + 20, fontSize + 10);
            
            // Draw label text
            ctx.fillStyle = 'white';
            ctx.fillText(label, x1 + 10, Math.max(fontSize + 2, y1 - 8));
            
            // Draw person number
            ctx.font = `bold ${fontSize * 0.8}px Inter, sans-serif`;
            ctx.fillStyle = 'white';
            ctx.fillText(`Person ${idx + 1}`, x1 + 10, y2 - 10);
            
            // Draw head position if available
            if (person.head_detected && person.head_position) {
              const [hx, hy] = person.head_position;
              const dotSize = Math.max(6, img.width / 150);
              ctx.fillStyle = '#06b6d4';
              ctx.beginPath();
              ctx.arc(hx, hy, dotSize, 0, 2 * Math.PI);
              ctx.fill();
              ctx.strokeStyle = 'white';
              ctx.lineWidth = 3;
              ctx.stroke();
            }
            
            // Draw helmet bounding box if available
            if (person.helmet_bbox) {
              const [hx1, hy1, hx2, hy2] = person.helmet_bbox;
              ctx.strokeStyle = '#fbbf24';
              ctx.lineWidth = 3;
              ctx.setLineDash([10, 5]);
              ctx.strokeRect(hx1, hy1, hx2 - hx1, hy2 - hy1);
              ctx.setLineDash([]);
              
              // Helmet label
              ctx.font = `${fontSize * 0.7}px Inter, sans-serif`;
              ctx.fillStyle = '#fbbf24';
              ctx.fillText('Helmet', hx1, hy1 - 5);
            }
          });
        }
        
        resolve(canvas.toDataURL('image/jpeg', 0.9));
      };
      img.src = this.previewImage.src;
    });
  }

  async displayUploadResults(data) {
    const { total_people, compliant_people, non_compliant_people, compliance_rate, person_analyses } = data;
    
    const isCompliant = compliant_people === total_people && total_people > 0;
    
    // Generate annotated image
    const annotatedImageSrc = await this.drawDetectionsOnImage(data);
    
    this.uploadResults.innerHTML = `
      <!-- Detection Visualization -->
      <div class="detection-visualization">
        <div class="visualization-header">
          <h4>Detection Results</h4>
          <button class="btn btn-sm btn-secondary" onclick="this.closest('.detection-visualization').querySelector('img').requestFullscreen()">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 8V4m0 0h4M4 4l5 5m11-1V4m0 0h-4m4 0l-5 5M4 16v4m0 0h4m-4 0l5-5m11 5l-5-5m5 5v-4m0 4h-4"/>
            </svg>
            Fullscreen
          </button>
        </div>
        <div class="visualization-image">
          <img src="${annotatedImageSrc}" alt="Detection Result" />
        </div>
        <div class="visualization-legend">
          <div class="legend-item">
            <div class="legend-color" style="background: #10b981;"></div>
            <span>Compliant (Green)</span>
          </div>
          <div class="legend-item">
            <div class="legend-color" style="background: #ef4444;"></div>
            <span>Non-Compliant (Red)</span>
          </div>
          <div class="legend-item">
            <div class="legend-color" style="background: #fbbf24;"></div>
            <span>Helmet Detection (Yellow)</span>
          </div>
          <div class="legend-item">
            <div class="legend-color" style="background: #06b6d4;"></div>
            <span>Head Position (Cyan)</span>
          </div>
        </div>
      </div>
      
      <!-- Compliance Summary -->
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
      
      ${data.saved_to_history ? `
        <div class="history-saved-notice">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"/>
          </svg>
          Saved to history
        </div>
      ` : ''}
      
      ${person_analyses && person_analyses.length > 0 ? `
        <div class="person-list">
          <h4 style="margin-bottom: var(--spacing-md); color: var(--text-primary);">Individual Analysis</h4>
          ${person_analyses.map((person, i) => `
            <div class="person-card ${person.overall_compliant ? 'compliant' : 'non-compliant'}">
              <div class="person-header">
                <span class="person-id">Person ${i + 1}</span>
                <span class="person-status">${person.overall_compliant ? 'COMPLIANT' : 'NON-COMPLIANT'}</span>
              </div>
              <div class="person-details">
                <div class="person-detail-row">
                  <span class="detail-key">Status:</span>
                  <span class="detail-val">${person.status || 'Unknown'}</span>
                </div>
                <div class="person-detail-row">
                  <span class="detail-key">Head Detected:</span>
                  <span class="detail-val">${person.head_detected ? 'Yes ✓' : 'No ✗'}</span>
                </div>
                ${person.has_helmet ? `
                  <div class="person-detail-row">
                    <span class="detail-key">Helmet Found:</span>
                    <span class="detail-val">Yes${person.helmet_confidence ? ` (${(person.helmet_confidence * 100).toFixed(1)}% conf)` : ''}</span>
                  </div>
                ` : ''}
                ${person.distance_to_head !== undefined && person.distance_to_head !== null ? `
                  <div class="person-detail-row">
                    <span class="detail-key">Distance to Head:</span>
                    <span class="detail-val">${person.distance_to_head.toFixed(0)}px</span>
                  </div>
                ` : ''}
              </div>
              <div class="person-reason">
                ${person.reason || 'No additional information'}
              </div>
            </div>
          `).join('')}
        </div>
      ` : ''}
    `;
  }

  setUploadStatus(text, state) {
    if (this.uploadStatusText) this.uploadStatusText.textContent = text;
    if (this.uploadStatusDot) this.uploadStatusDot.className = `status-dot ${state}`;
  }

  showUploadProgress(percent) {
    if (this.uploadProgress) this.uploadProgress.style.display = 'block';
    if (this.uploadProgressFill) this.uploadProgressFill.style.width = `${percent}%`;
    if (this.uploadProgressText) this.uploadProgressText.textContent = `Processing... ${percent}%`;
  }

  hideUploadProgress() {
    setTimeout(() => {
      if (this.uploadProgress) this.uploadProgress.style.display = 'none';
      if (this.uploadProgressFill) this.uploadProgressFill.style.width = '0%';
    }, 300);
  }

  // =========================================================================
  // LIVE CAMERA
  // =========================================================================
  
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
      this.showNotification('Could not access camera. Please check permissions.', 'error');
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

    const canvas = document.createElement('canvas');
    canvas.width = this.cameraVideo.videoWidth;
    canvas.height = this.cameraVideo.videoHeight;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(this.cameraVideo, 0, 0);

    canvas.toBlob(async (blob) => {
      if (!this.cameraActive) return;

      try {
        const formData = new FormData();
        formData.append('file', blob, 'frame.jpg');
        formData.append('threshold', this.threshold.toString());
        formData.append('save_to_history', 'false'); // Don't save live frames to history

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

      this.updateFPS();

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

      ctx.strokeStyle = isCompliant ? '#10b981' : '#ef4444';
      ctx.lineWidth = 3;
      ctx.strokeRect(x1 * scaleX, y1 * scaleY, (x2 - x1) * scaleX, (y2 - y1) * scaleY);

      const label = isCompliant ? 'COMPLIANT' : 'NON-COMPLIANT';
      ctx.font = 'bold 16px Inter, sans-serif';
      const textWidth = ctx.measureText(label).width;
      
      ctx.fillStyle = isCompliant ? '#10b981' : '#ef4444';
      ctx.fillRect(x1 * scaleX, Math.max(0, y1 * scaleY - 30), textWidth + 16, 30);
      
      ctx.fillStyle = 'white';
      ctx.fillText(label, x1 * scaleX + 8, Math.max(20, y1 * scaleY - 8));

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
    if (this.liveStatusText) this.liveStatusText.textContent = text;
    if (this.liveStatusDot) this.liveStatusDot.className = `status-dot ${state}`;
  }

  adjustThreshold(amount) {
    this.threshold = Math.max(40, Math.min(300, this.threshold + amount));
    this.thresholdDisplay.textContent = `${this.threshold}px`;
  }

  // =========================================================================
  // HISTORY MANAGEMENT
  // =========================================================================
  
  async loadHistory() {
    try {
      let url = `${this.API_BASE}/history?limit=${this.historyLimit}&offset=${this.historyOffset}`;
      
      if (this.currentFilter === 'compliant') {
        url += '&compliant_only=true';
      } else if (this.currentFilter === 'violations') {
        url += '&compliant_only=false';
      }
      
      const response = await fetch(url);
      
      if (!response.ok) {
        throw new Error('Failed to load history');
      }
      
      const data = await response.json();
      this.historyRecords = data.records;
      this.renderHistory(data.records);
      this.updatePagination(data);
      
    } catch (error) {
      console.error('Failed to load history:', error);
      this.historyGrid.innerHTML = `
        <div class="empty-state">
          <svg width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="currentColor" opacity="0.3">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"/>
          </svg>
          <p>Failed to load history</p>
          <span>Check your connection and try again</span>
        </div>
      `;
    }
  }
  
  async loadHistoryStats() {
    try {
      const response = await fetch(`${this.API_BASE}/history/stats/summary`);
      
      if (!response.ok) return;
      
      const stats = await response.json();
      
      if (this.statsTotalScans) this.statsTotalScans.textContent = stats.total_scans || 0;
      if (this.statsTotalCompliant) this.statsTotalCompliant.textContent = stats.total_compliant || 0;
      if (this.statsTotalViolations) this.statsTotalViolations.textContent = stats.total_non_compliant || 0;
      if (this.statsOverallRate) {
        const rate = stats.overall_compliance_rate ? (stats.overall_compliance_rate * 100).toFixed(1) : 0;
        this.statsOverallRate.textContent = `${rate}%`;
      }
      
    } catch (error) {
      console.error('Failed to load history stats:', error);
    }
  }

  renderHistory(records) {
    if (!records || records.length === 0) {
      this.historyGrid.innerHTML = `
        <div class="empty-state">
          <svg width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="currentColor" opacity="0.3">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z"/>
          </svg>
          <p>No detection history found</p>
          <span>Analyze an image to start building history</span>
        </div>
      `;
      this.historyPagination.style.display = 'none';
      return;
    }
    
    this.historyPagination.style.display = 'flex';
    
    this.historyGrid.innerHTML = records.map(record => {
      const date = new Date(record.created_at);
      const isFullyCompliant = record.non_compliant_people === 0;
      
      return `
        <div class="history-card ${isFullyCompliant ? 'compliant' : 'non-compliant'}" data-id="${record.id}">
          <div class="history-image">
            <img src="${record.image_url}" alt="Detection ${record.id}" loading="lazy" onerror="this.src='data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 width=%22100%22 height=%22100%22><rect fill=%22%23334155%22 width=%22100%22 height=%22100%22/><text fill=%22%2394a3b8%22 x=%2250%22 y=%2250%22 text-anchor=%22middle%22 dy=%22.3em%22 font-size=%2212%22>No Image</text></svg>'"/>
            <div class="history-badge ${isFullyCompliant ? 'success' : 'danger'}">
              ${isFullyCompliant ? '✓ Compliant' : '⚠ Violation'}
            </div>
          </div>
          <div class="history-content">
            <div class="history-date">
              ${date.toLocaleDateString()} at ${date.toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'})}
            </div>
            <div class="history-stats">
              <span class="stat-mini">
                <strong>${record.total_people}</strong> people
              </span>
              <span class="stat-mini success">
                <strong>${record.compliant_people}</strong> compliant
              </span>
              <span class="stat-mini danger">
                <strong>${record.non_compliant_people}</strong> violations
              </span>
            </div>
            <div class="history-rate">
              Compliance: ${((record.compliance_rate || 0) * 100).toFixed(1)}%
            </div>
          </div>
          <div class="history-actions">
            <button class="btn btn-sm btn-secondary" onclick="ppeSystem.viewHistoryDetail('${record.id}')">
              View Details
            </button>
            <button class="btn btn-sm btn-danger" onclick="ppeSystem.deleteHistoryRecord('${record.id}')">
              Delete
            </button>
          </div>
        </div>
      `;
    }).join('');
  }

  setFilter(filter) {
    this.currentFilter = filter;
    this.historyOffset = 0;
    
    this.filterBtns.forEach(btn => {
      btn.classList.toggle('active', btn.dataset.filter === filter);
    });
    
    this.loadHistory();
  }

  changePage(direction) {
    this.historyOffset += direction * this.historyLimit;
    if (this.historyOffset < 0) this.historyOffset = 0;
    this.loadHistory();
  }

  updatePagination(data) {
    const currentPage = Math.floor(this.historyOffset / this.historyLimit) + 1;
    this.pageInfo.textContent = `Page ${currentPage}`;
    this.prevPageBtn.disabled = this.historyOffset === 0;
    this.nextPageBtn.disabled = data.count < this.historyLimit;
  }

  async viewHistoryDetail(recordId) {
    try {
      const response = await fetch(`${this.API_BASE}/history/${recordId}`);
      
      if (!response.ok) {
        throw new Error('Failed to fetch record');
      }
      
      const record = await response.json();
      this.showDetailModal(record);
      
    } catch (error) {
      console.error('Failed to fetch record:', error);
      this.showNotification('Failed to load record details', 'error');
    }
  }
  
  showDetailModal(record) {
    const date = new Date(record.created_at);
    const isCompliant = record.non_compliant_people === 0;
    
    this.modalBody.innerHTML = `
      <div class="modal-image">
        <img src="${record.image_url}" alt="Detection" />
      </div>
      
      <div class="modal-info">
        <div class="modal-timestamp">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z"/>
          </svg>
          ${date.toLocaleString()}
        </div>
        
        <div class="modal-compliance ${isCompliant ? 'compliant' : 'non-compliant'}">
          <div class="modal-compliance-icon">
            ${isCompliant 
              ? '<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"/></svg>'
              : '<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"/></svg>'}
          </div>
          <div>
            <div class="modal-compliance-title">${isCompliant ? 'FULLY COMPLIANT' : 'VIOLATIONS DETECTED'}</div>
            <div class="modal-compliance-subtitle">
              ${record.compliant_people} of ${record.total_people} workers compliant
            </div>
          </div>
        </div>
        
        <div class="modal-stats-grid">
          <div class="modal-stat">
            <span class="modal-stat-value">${record.total_people}</span>
            <span class="modal-stat-label">Total People</span>
          </div>
          <div class="modal-stat success">
            <span class="modal-stat-value">${record.compliant_people}</span>
            <span class="modal-stat-label">Compliant</span>
          </div>
          <div class="modal-stat danger">
            <span class="modal-stat-value">${record.non_compliant_people}</span>
            <span class="modal-stat-label">Violations</span>
          </div>
          <div class="modal-stat">
            <span class="modal-stat-value">${((record.compliance_rate || 0) * 100).toFixed(1)}%</span>
            <span class="modal-stat-label">Compliance Rate</span>
          </div>
        </div>
        
        ${record.person_analyses && record.person_analyses.length > 0 ? `
          <div class="modal-analyses">
            <h4>Individual Analysis</h4>
            ${record.person_analyses.map((person, i) => `
              <div class="modal-person ${person.overall_compliant ? 'compliant' : 'non-compliant'}">
                <div class="modal-person-header">
                  <span>Person ${i + 1}</span>
                  <span class="modal-person-status">${person.overall_compliant ? '✓ Compliant' : '✗ Non-compliant'}</span>
                </div>
                <div class="modal-person-reason">${person.reason || 'No details'}</div>
              </div>
            `).join('')}
          </div>
        ` : ''}
        
        <div class="modal-meta">
          <span>Threshold: ${record.threshold_used || 'N/A'}px</span>
          ${record.original_filename ? `<span>File: ${record.original_filename}</span>` : ''}
        </div>
      </div>
    `;
    
    this.detailModal.style.display = 'flex';
    document.body.style.overflow = 'hidden';
  }
  
  closeModal() {
    this.detailModal.style.display = 'none';
    document.body.style.overflow = '';
  }

  async deleteHistoryRecord(recordId) {
    if (!confirm('Delete this detection record? This cannot be undone.')) return;
    
    try {
      const response = await fetch(`${this.API_BASE}/history/${recordId}`, {
        method: 'DELETE'
      });
      
      if (response.ok) {
        this.showNotification('Record deleted successfully', 'success');
        this.loadHistory();
        this.loadHistoryStats();
      } else {
        const data = await response.json();
        throw new Error(data.error || 'Failed to delete record');
      }
    } catch (error) {
      console.error('Delete failed:', error);
      this.showNotification(error.message || 'Failed to delete record', 'error');
    }
  }
  async clearAllHistory() {
  const confirmed = confirm(
    'DELETE ALL HISTORY?\n\n' +
    'This will permanently delete:\n' +
    '• All detection records\n' +
    '• All uploaded images\n' +
    '• All statistics data\n\n' +
    'This action CANNOT be undone!\n\n' +
    'Are you absolutely sure?'
  );
  
  if (!confirmed) return;
  
  const doubleConfirm = confirm(
    'FINAL WARNING!\n\n' +
    'You are about to delete ALL history permanently.\n\n' +
    'Click OK to proceed with deletion.'
  );
  
  if (!doubleConfirm) return;
  
  try {
    this.clearAllHistoryBtn.disabled = true;
    this.clearAllHistoryBtn.innerHTML = `
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" class="spin">
        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"/>
      </svg>
      Deleting...
    `;
    
    const response = await fetch(`${this.API_BASE}/history?confirm=true`, {
      method: 'DELETE'
    });
    
    if (response.ok) {
      const data = await response.json();
      this.showNotification(
        `Successfully deleted ${data.deleted_records} records and ${data.deleted_images} images`,
        'success'
      );
      this.loadHistory();
      this.loadHistoryStats();
      this.historyOffset = 0;
    } else {
      const errorData = await response.json();
      throw new Error(errorData.error || 'Failed to clear history');
    }
  } catch (error) {
    console.error('Clear all history failed:', error);
    this.showNotification(
      error.message || 'Failed to clear history. Please try again.',
      'error'
    );
  } finally {
    this.clearAllHistoryBtn.disabled = false;
    this.clearAllHistoryBtn.innerHTML = `
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor">
        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"/>
      </svg>
      Clear All History
    `;
    }
  }

  // =========================================================================
  // STATISTICS
  // =========================================================================
  
  updateStatsFromResult(data) {
    this.stats.totalAnalyzed += data.total_people || 0;
    this.stats.compliantCount += data.compliant_people || 0;
    this.updateStats();
  }

  updateStats() {
    if (this.totalAnalyzedEl) {
      this.totalAnalyzedEl.textContent = this.stats.totalAnalyzed;
    }
    
    if (this.complianceRateEl) {
      if (this.stats.totalAnalyzed > 0) {
        const rate = (this.stats.compliantCount / this.stats.totalAnalyzed * 100).toFixed(1);
        this.complianceRateEl.textContent = `${rate}%`;
      } else {
        this.complianceRateEl.textContent = '--%';
      }
    }
  }

  // =========================================================================
  // NOTIFICATIONS
  // =========================================================================
  
  showNotification(message, type = 'info') {
    // Remove existing notification
    const existing = document.querySelector('.notification');
    if (existing) existing.remove();
    
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.innerHTML = `
      <span>${message}</span>
      <button class="notification-close">&times;</button>
    `;
    
    document.body.appendChild(notification);
    
    // Trigger animation
    setTimeout(() => notification.classList.add('show'), 10);
    
    // Auto-dismiss
    const timeout = setTimeout(() => {
      notification.classList.remove('show');
      setTimeout(() => notification.remove(), 300);
    }, 4000);
    
    // Manual dismiss
    notification.querySelector('.notification-close').addEventListener('click', () => {
      clearTimeout(timeout);
      notification.classList.remove('show');
      setTimeout(() => notification.remove(), 300);
    });
  }
}

// Initialize the system
let ppeSystem;
document.addEventListener('DOMContentLoaded', () => {
  ppeSystem = new PPEComplianceSystem();
});

// Export for global access (for inline onclick handlers)
window.ppeSystem = ppeSystem;