/**
 * PPE Helmet Compliance System
 * Frontend JavaScript with Supabase History Integration
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
    
    console.log("PPE System Initialized");
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
    this.captureFrameBtn = document.getElementById('captureFrame');
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
    this.captureFrameBtn?.addEventListener('click', () => this.captureSnapshot());
    
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
      } else {
        this.setAPIStatus('error', 'API Error');
      }
    } catch (error) {
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
    this.navItems.forEach(item => item.classList.remove('active'));
    document.querySelector(`[data-view="${view}"]`)?.classList.add('active');
    
    this.uploadView.classList.remove('active');
    this.liveView.classList.remove('active');
    this.historyView.classList.remove('active');
    
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

  // =========================================================================
  // UPDATED VISUALIZATION LOGIC (Shared for Upload and History)
  // =========================================================================

  /**
   * Helper to draw skeleton lines
   */
  drawSkeleton(ctx, landmarks, scaleX = 1, scaleY = 1) {
    if (!landmarks) return;
    
    // Points map
    const pts = {
      N: landmarks.nose,
      LS: landmarks.shoulders[0], RS: landmarks.shoulders[1],
      LE: landmarks.elbows[0],    RE: landmarks.elbows[1],
      LW: landmarks.wrists[0],    RW: landmarks.wrists[1],
      LH: landmarks.hips[0],      RH: landmarks.hips[1],
      TC: landmarks.torso_center
    };
    
    // Connections to draw
    const connections = [
      ['LS', 'RS'], ['LS', 'LE'], ['LE', 'LW'], // Left Arm
      ['RS', 'RE'], ['RE', 'RW'],               // Right Arm
      ['LS', 'LH'], ['RS', 'RH'],               // Torso Sides
      ['LH', 'RH'], ['LS', 'TC'], ['RS', 'TC'], // Hips & Neck
      ['N', 'TC']                               // Nose to Torso
    ];
    
    // Draw Bones (Yellow)
    ctx.strokeStyle = 'cyan';
    ctx.lineWidth = 2;
    
    connections.forEach(([p1, p2]) => {
      if (pts[p1] && pts[p2]) {
        ctx.beginPath();
        ctx.moveTo(pts[p1][0] * scaleX, pts[p1][1] * scaleY);
        ctx.lineTo(pts[p2][0] * scaleX, pts[p2][1] * scaleY);
        ctx.stroke();
      }
    });
    
    // Draw Joints (Red)
    ctx.fillStyle = 'red';
    Object.values(pts).forEach(pt => {
      if (pt) {
        ctx.beginPath();
        ctx.arc(pt[0] * scaleX, pt[1] * scaleY, 4, 0, 2 * Math.PI);
        ctx.fill();
      }
    });
  }

  /**
   * Generates an annotated image with bounding boxes and labels
   * Works for both local blob URLs (Upload) and external URLs (History/Supabase)
   */
  async generateAnnotatedImage(imageSource, personAnalyses) {
    return new Promise((resolve) => {
      const img = new Image();
      // 'anonymous' is crucial for drawing external images (Supabase) onto canvas
      // without tainting it, assuming Supabase bucket has CORS configured.
      img.crossOrigin = "anonymous"; 
      
      img.onload = () => {
        try {
          const canvas = document.createElement('canvas');
          canvas.width = img.width;
          canvas.height = img.height;
          const ctx = canvas.getContext('2d');
          
          // 1. Draw original image
          ctx.drawImage(img, 0, 0);
          
          const fontSize = Math.max(16, img.width / 60);
          ctx.font = `bold ${fontSize}px Inter, sans-serif`;

          // 2. Define Colors for different Classes
          const colors = {
            'hardhat': '#fbbf24', // Yellow
            'helmet': '#fbbf24',
            'safety vest': '#f97316', // Orange
            'vest': '#f97316',
            'gloves': '#a855f7', // Purple
            'boots': '#854d0e', // Brown
            'mask': '#3b82f6', // Blue
            'person': '#94a3b8',
            'default': '#06b6d4' // Cyan
          };

          // 3. Draw Person Analysis
          if (personAnalyses && personAnalyses.length > 0) {
            personAnalyses.forEach((person, idx) => {
              
              // --- NEW: Draw Skeleton ---
              if (person.landmarks) {
                this.drawSkeleton(ctx, person.landmarks);
              }

              const [x1, y1, x2, y2] = person.person_box;
              const isCompliant = person.overall_compliant;
              
              // Draw person bounding box
              ctx.strokeStyle = isCompliant ? '#10b981' : '#ef4444';
              ctx.lineWidth = Math.max(4, img.width / 300);
              ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
              
              // Draw COMPLIANCE Status Label
              const label = isCompliant ? 'COMPLIANT' : 'NON-COMPLIANT';
              const textWidth = ctx.measureText(label).width;
              
              ctx.fillStyle = isCompliant ? '#10b981' : '#ef4444';
              ctx.fillRect(x1, Math.max(0, y1 - fontSize - 15), textWidth + 20, fontSize + 10);
              
              ctx.fillStyle = 'white';
              ctx.fillText(label, x1 + 10, Math.max(fontSize + 2, y1 - 8));
              
              // Draw Person ID
              ctx.font = `bold ${fontSize * 0.8}px Inter, sans-serif`;
              ctx.fillText(`P${idx + 1}`, x1, y2 + fontSize);

              // 4. Draw ALL associated PPE items for this person
              if (person.ppe_items && person.ppe_items.length > 0) {
                person.ppe_items.forEach(item => {
                  const [ix1, iy1, ix2, iy2] = item.bbox;
                  const className = item.class_name.toLowerCase();
                  
                  // Determine color
                  let itemColor = colors['default'];
                  for (const key in colors) {
                    if (className.includes(key)) itemColor = colors[key];
                  }

                  // Draw item box
                  ctx.strokeStyle = itemColor;
                  ctx.lineWidth = 2;
                  ctx.setLineDash([]); // Solid line
                  ctx.strokeRect(ix1, iy1, ix2 - ix1, iy2 - iy1);

                  // Draw item label WITH CONFIDENCE
                  const confPercent = Math.round(item.confidence * 100);
                  const itemLabel = `${item.class_name} ${confPercent}%`;

                  ctx.font = `${fontSize * 0.6}px Inter, sans-serif`;
                  ctx.fillStyle = itemColor;
                  ctx.fillText(itemLabel, ix1, iy1 - 5);
                });
              }
            });
          }
          
          resolve(canvas.toDataURL('image/jpeg', 0.9));
        } catch(e) {
          console.warn("Could not draw on canvas (likely CORS issue). Returning original image.", e);
          resolve(imageSource); // Fallback to raw image if canvas fails
        }
      };
      
      img.onerror = () => {
        console.error("Failed to load image for annotation");
        resolve(imageSource);
      };
      
      img.src = imageSource;
    });
  }

  async displayUploadResults(data) {
    const { total_people, compliant_people, non_compliant_people, compliance_rate, person_analyses } = data;
    const isCompliant = compliant_people === total_people && total_people > 0;
    
    // Reuse the new generalized function
    const annotatedImageSrc = await this.generateAnnotatedImage(this.previewImage.src, person_analyses);
    
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
          <div class="legend-item"><div class="legend-color" style="background: #10b981;"></div><span>Compliant</span></div>
          <div class="legend-item"><div class="legend-color" style="background: #ef4444;"></div><span>Non-Compliant</span></div>
          <div class="legend-item"><div class="legend-color" style="background: #fbbf24;"></div><span>Helmet</span></div>
          <div class="legend-item"><div class="legend-color" style="background: #f97316;"></div><span>Vest</span></div>
          <div class="legend-item"><div class="legend-color" style="background: #a855f7;"></div><span>Gloves</span></div>
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
            <div class="detail-label">Compliance Rate</div>
            <div class="detail-value">${(compliance_rate * 100).toFixed(1)}%</div>
          </div>
        </div>
      </div>

      <!-- Person List with New Gear Badges -->
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
                
                <!-- NEW SECTION: Detected Gear List -->
                <div class="person-detail-row" style="flex-direction: column; align-items: flex-start; gap: 5px;">
                  <span class="detail-key">Detected Gear:</span>
                  <div style="display: flex; flex-wrap: wrap; gap: 5px;">
                    ${person.detected_gear && person.detected_gear.length > 0 
                      ? person.detected_gear.map(gear => 
                          `<span style="background: #f1f5f9; color: #475569; padding: 2px 8px; border-radius: 4px; font-size: 0.75rem; border: 1px solid #e2e8f0;">${gear}</span>`
                        ).join('') 
                      : '<span style="color: #94a3b8; font-style: italic; font-size: 0.85rem;">None detected</span>'
                    }
                  </div>
                </div>

                <div class="person-detail-row">
                  <span class="detail-key">Reason:</span>
                  <span class="detail-val">${person.reason || 'N/A'}</span>
                </div>
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
        if(this.captureFrameBtn) this.captureFrameBtn.disabled = false; // Enable capture
        
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
    if(this.captureFrameBtn) this.captureFrameBtn.disabled = true; // Disable capture
    
    this.setLiveStatus('Inactive', 'inactive');
    this.resetLiveStats();
  }

  async captureSnapshot() {
    if (!this.cameraActive) return;
    
    // Disable button to prevent double clicks
    if(this.captureFrameBtn) {
        this.captureFrameBtn.disabled = true;
        this.captureFrameBtn.innerHTML = `Saving...`;
    }

    const canvas = document.createElement('canvas');
    canvas.width = this.cameraVideo.videoWidth;
    canvas.height = this.cameraVideo.videoHeight;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(this.cameraVideo, 0, 0);

    canvas.toBlob(async (blob) => {
      try {
        const formData = new FormData();
        formData.append('file', blob, 'snapshot.jpg');
        formData.append('threshold', this.threshold.toString());
        formData.append('save_to_history', 'true'); // FORCE SAVE

        const response = await fetch(`${this.API_BASE}/detect-helmet`, {
          method: 'POST',
          body: formData
        });

        if (response.ok) {
          const data = await response.json();
          if (data.saved_to_history) {
            this.showNotification('Snapshot saved to history!', 'success');
          } else {
            this.showNotification('Snapshot captured but save failed.', 'error');
          }
        } else {
             throw new Error("API Error");
        }
      } catch (error) {
        console.error('Snapshot error:', error);
        this.showNotification('Failed to save snapshot.', 'error');
      } finally {
        // Re-enable button
        if(this.captureFrameBtn) {
            this.captureFrameBtn.disabled = false;
            this.captureFrameBtn.innerHTML = `
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 9a2 2 0 012-2h.93a2 2 0 001.664-.89l.812-1.22A2 2 0 0110.07 4h3.86a2 2 0 011.664.89l.812 1.22A2 2 0 0018.07 7H19a2 2 0 012 2v9a2 2 0 01-2 2H5a2 2 0 01-2-2V9z"/>
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 13a3 3 0 11-6 0 3 3 0 016 0z"/>
              </svg>
              Capture
            `;
        }
      }
    }, 'image/jpeg', 0.9);
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

    // Define Colors map for live view as well
    const colors = {
      'hardhat': '#fbbf24', 'helmet': '#fbbf24',
      'safety vest': '#f97316', 'vest': '#f97316',
      'gloves': '#a855f7', 'boots': '#854d0e',
      'mask': '#3b82f6', 'default': '#06b6d4'
    };

    data.person_analyses.forEach((person) => {
      // --- NEW: Draw Skeleton in Live View ---
      if (person.landmarks) {
        this.drawSkeleton(ctx, person.landmarks, scaleX, scaleY);
      }

      const [x1, y1, x2, y2] = person.person_box;
      const isCompliant = person.overall_compliant;

      // Draw Person Box
      ctx.strokeStyle = isCompliant ? '#10b981' : '#ef4444';
      ctx.lineWidth = 3;
      ctx.strokeRect(x1 * scaleX, y1 * scaleY, (x2 - x1) * scaleX, (y2 - y1) * scaleY);

      // Draw Label
      const label = isCompliant ? 'COMPLIANT' : 'NON-COMPLIANT';
      ctx.font = 'bold 16px Inter, sans-serif';
      const textWidth = ctx.measureText(label).width;
      
      ctx.fillStyle = isCompliant ? '#10b981' : '#ef4444';
      ctx.fillRect(x1 * scaleX, Math.max(0, y1 * scaleY - 30), textWidth + 16, 30);
      
      ctx.fillStyle = 'white';
      ctx.fillText(label, x1 * scaleX + 8, Math.max(20, y1 * scaleY - 8));

      // Draw Head
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

      // Draw Associated Gear with Confidence
      if (person.ppe_items && person.ppe_items.length > 0) {
         person.ppe_items.forEach(item => {
            const [ix1, iy1, ix2, iy2] = item.bbox;
            const className = item.class_name.toLowerCase();
            let color = colors['default'];
            for (const k in colors) { if(className.includes(k)) color = colors[k]; }
            
            ctx.strokeStyle = color;
            ctx.lineWidth = 2;
            ctx.setLineDash([]); // Solid line
            ctx.strokeRect(ix1 * scaleX, iy1 * scaleY, (ix2 - ix1) * scaleX, (iy2 - iy1) * scaleY);
            
            // --- ADDED CONFIDENCE ---
            const confPercent = Math.round(item.confidence * 100);
            ctx.fillStyle = color;
            ctx.font = '12px Inter, sans-serif';
            ctx.fillText(`${item.class_name} ${confPercent}%`, ix1 * scaleX, (iy1 * scaleY) - 5);
         });
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
            <span class="detection-sub">${person.detected_gear ? person.detected_gear.join(', ') : ''}</span>
          </div>
          <span class="detection-confidence">${person.overall_compliant ? '✓' : '✗'}</span>
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
      if (this.currentFilter === 'compliant') url += '&compliant_only=true';
      else if (this.currentFilter === 'violations') url += '&compliant_only=false';
      
      const response = await fetch(url);
      if (!response.ok) throw new Error('Failed to load history');
      
      const data = await response.json();
      this.historyRecords = data.records;
      this.renderHistory(data.records);
      this.updatePagination(data);
      
    } catch (error) {
      console.error('Failed to load history:', error);
      this.historyGrid.innerHTML = `
        <div class="empty-state">
          <p>Failed to load history</p>
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
    } catch (error) { console.error(error); }
  }

  renderHistory(records) {
    if (!records || records.length === 0) {
      this.historyGrid.innerHTML = `
        <div class="empty-state">
          <p>No detection history found</p>
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
            <img src="${record.image_url}" loading="lazy" />
            <div class="history-badge ${isFullyCompliant ? 'success' : 'danger'}">
              ${isFullyCompliant ? '✓ Pass' : '⚠ Fail'}
            </div>
          </div>
          <div class="history-content">
            <div class="history-date">
              ${date.toLocaleDateString()} at ${date.toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'})}
            </div>
            <div class="history-actions">
              <button class="btn btn-sm btn-secondary" onclick="window.ppeSystem.viewHistoryDetail('${record.id}')">Details</button>
              <button class="btn btn-sm btn-danger" onclick="window.ppeSystem.deleteHistoryRecord('${record.id}')">Delete</button>
            </div>
          </div>
        </div>
      `;
    }).join('');
  }

  setFilter(filter) {
    this.currentFilter = filter;
    this.historyOffset = 0;
    this.filterBtns.forEach(btn => btn.classList.toggle('active', btn.dataset.filter === filter));
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
      if (!response.ok) throw new Error('Failed');
      const record = await response.json();
      this.showDetailModal(record);
    } catch (error) { console.error(error); }
  }
  
  async showDetailModal(record) {
    const date = new Date(record.created_at);
    const isCompliant = record.non_compliant_people === 0;
    
    // Safety check: Ensure person_analyses is an object/array, not a string
    let analyses = record.person_analyses;
    if (typeof analyses === 'string') {
        try {
            analyses = JSON.parse(analyses);
        } catch (e) {
            console.error("Failed to parse person_analyses JSON:", e);
            analyses = []; // Fallback to empty array
        }
    }
    
    // Generate annotated image dynamically
    // Note: We use the raw URL from Supabase and overlay data from 'person_analyses'
    const annotatedImage = await this.generateAnnotatedImage(record.image_url, analyses);
    
    // Render Modal Content
    this.modalBody.innerHTML = `
      <div class="modal-image">
        <img src="${annotatedImage}" alt="Detection" />
      </div>
      <div class="modal-info">
        <h3>${isCompliant ? 'COMPLIANT' : 'NON-COMPLIANT'}</h3>
        <p class="modal-timestamp">${date.toLocaleString()}</p>
        
        <!-- NEW: Threshold Display -->
        <div style="background: var(--bg-dark); padding: 10px; border-radius: 8px; margin-bottom: 15px; border: 1px solid var(--border);">
            <strong style="color: var(--text-primary);">Config Used:</strong>
            <span style="color: var(--text-secondary); margin-left: 5px;">
                Threshold: ${record.threshold_used || 'N/A'}px
            </span>
        </div>
        
        <div class="modal-analyses">
            ${analyses && analyses.length > 0 ? analyses.map((p, i) => `
              <div class="modal-person ${p.overall_compliant ? 'compliant' : 'non-compliant'}">
                <div class="modal-person-header">
                  <span>Person ${i+1}: ${p.overall_compliant ? 'Pass' : 'Fail'}</span>
                </div>
                
                <div class="modal-person-reason">
                   Reason: ${p.reason || 'N/A'}
                </div>
                
                <!-- Updated Gear Visualization -->
                <div style="margin-top: 10px;">
                  <strong style="font-size: 0.75rem; color: #94a3b8; display: block; margin-bottom: 5px;">DETECTED GEAR:</strong>
                  
                  ${p.ppe_items && p.ppe_items.length > 0 
                    ? `<div class="gear-list">
                        ${p.ppe_items.map(item => {
                           const confidence = Math.round(item.confidence * 100);
                           return `
                             <div class="gear-badge">
                               ${item.class_name} 
                               <span class="confidence">${confidence}%</span>
                             </div>
                           `;
                        }).join('')}
                       </div>`
                    : '<span style="color: #64748b; font-size: 0.8rem; font-style: italic;">No gear detected</span>'
                  }
                </div>
              </div>
            `).join('') : '<p>No analysis data available</p>'}
        </div>
      </div>
    `;
    this.detailModal.style.display = 'flex';
  }
  
  closeModal() {
    this.detailModal.style.display = 'none';
  }

  async deleteHistoryRecord(recordId) {
    if (!confirm('Delete this record?')) return;
    try {
      await fetch(`${this.API_BASE}/history/${recordId}`, { method: 'DELETE' });
      this.loadHistory();
      this.loadHistoryStats();
    } catch (error) { console.error(error); }
  }

  async clearAllHistory() {
    if(!confirm('Delete ALL history?')) return;
    try {
        await fetch(`${this.API_BASE}/history?confirm=true`, { method: 'DELETE' });
        this.loadHistory();
        this.loadHistoryStats();
    } catch(e) { console.error(e); }
  }

  updateStatsFromResult(data) {
    this.stats.totalAnalyzed += data.total_people || 0;
    this.stats.compliantCount += data.compliant_people || 0;
    this.updateStats();
  }

  updateStats() {
    if (this.totalAnalyzedEl) this.totalAnalyzedEl.textContent = this.stats.totalAnalyzed;
    if (this.complianceRateEl) {
      if (this.stats.totalAnalyzed > 0) {
        const rate = (this.stats.compliantCount / this.stats.totalAnalyzed * 100).toFixed(1);
        this.complianceRateEl.textContent = `${rate}%`;
      } else {
        this.complianceRateEl.textContent = '--%';
      }
    }
  }
  
  showNotification(message, type = 'info') {
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.innerHTML = `<span>${message}</span>`;
    document.body.appendChild(notification);
    setTimeout(() => notification.classList.add('show'), 10);
    setTimeout(() => notification.remove(), 4000);
  }
}

// Corrected Initialization for Module Scope
document.addEventListener('DOMContentLoaded', () => {
  window.ppeSystem = new PPEComplianceSystem();
});