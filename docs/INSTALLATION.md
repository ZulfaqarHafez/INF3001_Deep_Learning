# Installation & Setup Guide

> **Complete guide for setting up the PPE Detection System from scratch**

---

## Table of Contents
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Detailed Setup](#detailed-setup)
  - [Backend Setup](#backend-setup)
  - [Frontend Setup](#frontend-setup)
  - [Supabase Configuration](#supabase-configuration)
- [Model Download](#model-download)
- [Troubleshooting](#troubleshooting)
- [Verification](#verification)

---

## Prerequisites

### System Requirements

**Minimum:**
- OS: Windows 10/11, macOS 12+, or Ubuntu 20.04+
- RAM: 8GB
- CPU: 4-core processor
- Storage: 2GB free space
- Webcam (for live detection)

**Recommended:**
- RAM: 16GB
- GPU: NVIDIA GPU with 4GB+ VRAM (optional, for faster inference)
- CPU: 6-core processor
- Storage: 5GB free space

### Software Requirements

1. **Python 3.11+**
   ```bash
   python --version  # Should show 3.11 or higher
   ```

2. **Node.js (Optional - for frontend development)**
   ```bash
   node --version  # Should show 16.0 or higher
   ```

3. **Git**
   ```bash
   git --version
   ```

4. **Modern Web Browser**
   - Chrome 90+
   - Firefox 88+
   - Safari 14+
   - Edge 90+

---

## Quick Start

For those who want to get running immediately:

```bash
# 1. Clone repository
git clone https://github.com/ZulfaqarHafez/AAI3001_Deep_Learning.git
cd AAI3001_Deep_Learning

# 2. Install backend dependencies
cd backend
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env
# Edit .env with your Supabase credentials

# 4. Run backend
python backend_api.py

# 5. Open frontend (in new terminal)
cd ../frontend
# Open index.html in browser or use Live Server
```

**Server runs at:** `http://127.0.0.1:8000`
**API Docs:** `http://127.0.0.1:8000/docs`

---

## Detailed Setup

### Backend Setup

#### Step 1: Clone Repository

```bash
git clone https://github.com/ZulfaqarHafez/AAI3001_Deep_Learning.git
cd AAI3001_Deep_Learning
```

#### Step 2: Create Virtual Environment

**On macOS/Linux:**
```bash
cd backend
python3 -m venv venv
source venv/bin/activate
```

**On Windows:**
```cmd
cd backend
python -m venv venv
venv\Scripts\activate
```

You should see `(venv)` in your terminal prompt.

#### Step 3: Install Dependencies

```bash
pip install --break-system-packages -r requirements.txt
```

**Key Packages Installed:**
- `ultralytics==8.0.196` - YOLOv8 framework
- `mediapipe==0.10.7` - Pose estimation
- `fastapi==0.110.0` - API framework
- `opencv-python==4.8.1` - Image processing
- `supabase==2.0.0` - Cloud storage
- `huggingface-hub` - Model download
- `python-dotenv==1.0.0` - Environment variables
- `uvicorn` - ASGI server

**Note:** If you encounter the error `externally-managed-environment`, use the `--break-system-packages` flag as shown above.

#### Step 4: Configure Environment Variables

Create `.env` file in `backend/` directory:

```bash
cp .env.example .env  # If example exists
# OR create manually:
nano .env  # or use your preferred editor
```

**Add the following configuration:**

```env
# === Supabase Cloud Configuration ===
SUPABASE_URL=https://your-project-id.supabase.co
SUPABASE_KEY=your-anon-public-api-key

# === Model Configuration ===
USE_HUGGINGFACE=true
HUGGINGFACE_REPO=iMaximusiV/yolo-ppe-detector
HUGGINGFACE_FILENAME=best_100Epoch.pt

# === Alternative: Local Model ===
# If you want to use local model instead:
# USE_HUGGINGFACE=false
# LOCAL_MODEL_PATH=../models/best_100Epoch.pt
```

**How to get Supabase credentials:**
1. Go to [supabase.com](https://supabase.com)
2. Create a new project (or use existing)
3. Go to Settings → API
4. Copy `Project URL` → Use as `SUPABASE_URL`
5. Copy `anon public` key → Use as `SUPABASE_KEY`

---

### Frontend Setup

The frontend uses vanilla JavaScript and doesn't require build steps.

#### Option 1: Simple HTTP Server (Recommended)

**Using Python:**
```bash
cd frontend
python -m http.server 5500
```
Then open: `http://localhost:5500`

**Using Node.js:**
```bash
cd frontend
npx http-server -p 5500
```

**Using PHP:**
```bash
cd frontend
php -S localhost:5500
```

#### Option 2: VS Code Live Server

1. Install "Live Server" extension in VS Code
2. Open `frontend/index.html`
3. Right-click → "Open with Live Server"

#### Option 3: Direct File Opening (Not Recommended)

You can open `index.html` directly in browser, but some features (camera access, CORS) may not work properly.

---

### Supabase Configuration

#### Database Setup

1. **Go to Supabase Dashboard** → Your Project → SQL Editor

2. **Create Detection History Table:**

```sql
-- Create table for storing detection records
CREATE TABLE detection_history (
  id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  image_url TEXT NOT NULL,
  image_path TEXT NOT NULL,
  original_filename TEXT,
  total_people INTEGER DEFAULT 0,
  compliant_people INTEGER DEFAULT 0,
  non_compliant_people INTEGER DEFAULT 0,
  compliance_rate NUMERIC(5,4) DEFAULT 0,
  person_analyses JSONB,
  threshold_used INTEGER
);

-- Create indexes for faster queries
CREATE INDEX idx_created_at ON detection_history(created_at DESC);
CREATE INDEX idx_compliance ON detection_history(non_compliant_people);
CREATE INDEX idx_compliance_rate ON detection_history(compliance_rate);
```

3. **Verify table created:**
```sql
SELECT * FROM detection_history LIMIT 1;
```

#### Storage Setup

1. **Go to Supabase Dashboard** → Storage

2. **Create New Bucket:**
   - Click "New bucket"
   - Name: `detection-images`
   - Public bucket: **YES** (toggle on)
   - Click "Create bucket"

3. **Configure CORS (Important!):**

Go to Storage → `detection-images` → Policies → CORS Configuration:

```json
{
  "allowedOrigins": [
    "http://localhost:3000",
    "http://127.0.0.1:8000",
    "http://127.0.0.1:5500",
    "http://localhost:5500"
  ],
  "allowedMethods": ["GET", "POST", "PUT", "DELETE"],
  "allowedHeaders": ["*"],
  "maxAge": 3600
}
```

4. **Set Public Access Policy:**

Go to Policies → New Policy:

**Policy Name:** "Public Read Access"
**Target roles:** `public`
**Policy definition:**
```sql
SELECT
  *
FROM
  storage.objects
WHERE
  bucket_id = 'detection-images';
```

**Create another policy for authenticated uploads:**

**Policy Name:** "Authenticated Upload"
**Target roles:** `authenticated`, `anon`
**Policy definition:**
```sql
INSERT INTO
  storage.objects
  (bucket_id, name, owner, metadata)
VALUES
  ('detection-images', *, *, *);
```

---

## Model Download

The system automatically downloads the model from Hugging Face on first run if `USE_HUGGINGFACE=true`.

### Automatic Download (Recommended)

1. Ensure `.env` has:
   ```env
   USE_HUGGINGFACE=true
   HUGGINGFACE_REPO=iMaximusiV/yolo-ppe-detector
   HUGGINGFACE_FILENAME=best_100Epoch.pt
   ```

2. Start backend - model downloads automatically:
   ```bash
   python backend_api.py
   ```

   You'll see:
   ```
   Downloading from Hugging Face: iMaximusiV/yolo-ppe-detector/best_100Epoch.pt
   Model size: 22.5 MB
   Download complete!
   ```

### Manual Download

If automatic download fails:

1. **Visit:** https://huggingface.co/iMaximusiV/yolo-ppe-detector

2. **Download `best_100Epoch.pt`** (22.5 MB)

3. **Create models folder:**
   ```bash
   mkdir -p models
   ```

4. **Move downloaded file:**
   ```bash
   mv ~/Downloads/best_100Epoch.pt models/
   ```

5. **Update `.env`:**
   ```env
   USE_HUGGINGFACE=false
   LOCAL_MODEL_PATH=../models/best_100Epoch.pt
   ```

---

## Running the System

### Start Backend Server

```bash
cd backend
source venv/bin/activate  # Windows: venv\Scripts\activate
python backend_api.py
```

**Expected Output:**
```
======================================================================
INITIALIZING HELMET COMPLIANCE DETECTOR
======================================================================

Loading Person Detector (YOLOv8n)...
✓ Person detector loaded

Loading PPE Detector (best_100Epoch.pt)...
Model source: Hugging Face (iMaximusiV/yolo-ppe-detector)
✓ PPE detector loaded

Initializing MediaPipe Pose...
✓ MediaPipe initialized

Connecting to Supabase...
✓ Supabase connected

======================================================================
SYSTEM READY
======================================================================

INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```

### Start Frontend

In a new terminal:

```bash
cd frontend
python -m http.server 5500
```

**Open browser:** http://localhost:5500

---

## Verification

### Test Backend Health

```bash
curl http://127.0.0.1:8000/health
```

**Expected Response:**
```json
{
  "status": "healthy",
  "person_detector": "loaded",
  "ppe_detector": "loaded",
  "mediapipe": "initialized",
  "supabase": "connected"
}
```

### Test API Documentation

Open: http://127.0.0.1:8000/docs

You should see FastAPI's interactive documentation (Swagger UI).

### Test Image Upload

1. Go to http://localhost:5500
2. Click "Image Upload" tab
3. Drag and drop a test image
4. Click "Analyze Image"
5. Should see detection results with skeleton overlay

### Test Live Camera

1. Go to "Live Camera" tab
2. Click "Start Camera"
3. Allow camera access when prompted
4. Should see live video with real-time detection

### Test History

1. Upload a few images with "Save to history" checked
2. Go to "History" tab
3. Should see saved detections in grid view

---

## Troubleshooting

### Issue: "Module not found" errors

**Solution:**
```bash
pip install --break-system-packages -r requirements.txt
```

### Issue: Supabase connection fails

**Symptoms:**
```
Error: 401 Unauthorized
```

**Solution:**
1. Check `.env` has correct `SUPABASE_URL` and `SUPABASE_KEY`
2. Verify key is the `anon public` key, not service role key
3. Ensure no extra spaces in `.env` file

### Issue: Model download fails

**Symptoms:**
```
Error downloading from Hugging Face
```

**Solution:**
1. Check internet connection
2. Try manual download: https://huggingface.co/iMaximusiV/yolo-ppe-detector
3. Use local model instead (update `.env`)

### Issue: Camera not working

**Symptoms:**
- "Camera not found" error
- Black screen in Live Camera view

**Solution:**
1. Check browser permissions (allow camera access)
2. Ensure frontend is served via HTTP (not file://)
3. Try different browser (Chrome recommended)
4. Check if another application is using camera

### Issue: CORS errors in browser console

**Symptoms:**
```
Access to fetch at 'http://127.0.0.1:8000' blocked by CORS
```

**Solution:**

Backend already has CORS configured. If still seeing errors:

1. **Check frontend is served via HTTP**, not file://
2. **Verify backend is running** on port 8000
3. **Clear browser cache**

### Issue: Images not displaying in History

**Symptoms:**
- History cards show broken image icons
- Console shows CORS errors for Supabase images

**Solution:**

1. **Verify Supabase bucket is public:**
   - Go to Storage → detection-images
   - Check "Public bucket" is enabled

2. **Configure CORS properly** (see Supabase Configuration section above)

3. **Check image URLs** in network tab - should be accessible

### Issue: Low FPS / Slow processing

**Symptoms:**
- Live camera shows <1 FPS
- Long processing times

**Solution:**

**CPU Optimization:**
- Processing 1-2 FPS on CPU is normal
- Reduce video resolution in camera settings
- Close other applications

**GPU Acceleration (Advanced):**
```bash
# Install CUDA-enabled PyTorch (if you have NVIDIA GPU)
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Issue: Port already in use

**Symptoms:**
```
ERROR:    [Errno 48] Address already in use
```

**Solution:**

**Find and kill process using port 8000:**
```bash
# macOS/Linux
lsof -ti:8000 | xargs kill -9

# Windows
netstat -ano | findstr :8000
taskkill /PID <PID> /F
```

**Or use different port:**
```bash
uvicorn backend_api:app --port 8080
```

---

## Advanced Configuration

### GPU Acceleration

If you have an NVIDIA GPU:

1. **Install CUDA Toolkit:**
   - Download from: https://developer.nvidia.com/cuda-downloads
   - Install appropriate version (11.8 recommended)

2. **Install PyTorch with CUDA:**
   ```bash
   pip uninstall torch torchvision
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```

3. **Verify GPU is detected:**
   ```python
   import torch
   print(torch.cuda.is_available())  # Should be True
   print(torch.cuda.get_device_name(0))  # Shows GPU name
   ```

### Custom Model Training

To use your own trained model:

1. **Train model** using YOLOv8:
   ```bash
   yolo train data=your_dataset.yaml model=yolov8s.pt epochs=100
   ```

2. **Copy best weights:**
   ```bash
   cp runs/detect/train/weights/best.pt models/custom_model.pt
   ```

3. **Update `.env`:**
   ```env
   USE_HUGGINGFACE=false
   LOCAL_MODEL_PATH=../models/custom_model.pt
   ```

### Production Deployment

For deploying to production server:

1. **Use Gunicorn** instead of uvicorn:
   ```bash
   pip install gunicorn
   gunicorn backend_api:app -w 4 -k uvicorn.workers.UvicornWorker
   ```

2. **Set up reverse proxy** (nginx):
   ```nginx
   server {
       listen 80;
       server_name your-domain.com;
       
       location / {
           proxy_pass http://127.0.0.1:8000;
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
       }
   }
   ```

3. **Use environment-specific .env:**
   ```env
   # Production .env
   DEBUG=false
   ALLOWED_ORIGINS=https://your-domain.com
   ```

---

## Next Steps

After successful installation:

1. **Read the main README** for feature overview
2. **Check [MODEL_TRAINING.md](MODEL_TRAINING.md)** for training details
3. **Explore [TECHNICAL_IMPLEMENTATION.md](TECHNICAL_IMPLEMENTATION.md)** for architecture
4. **Review [API_DOCUMENTATION.md](API_DOCUMENTATION.md)** for API specs

---

## Getting Help

### Common Resources:
- **GitHub Issues:** https://github.com/ZulfaqarHafez/AAI3001_Deep_Learning/issues
- **YOLOv8 Docs:** https://docs.ultralytics.com
- **MediaPipe Docs:** https://google.github.io/mediapipe
- **FastAPI Docs:** https://fastapi.tiangolo.com
- **Supabase Docs:** https://supabase.com/docs

### System Information for Bug Reports

When reporting issues, include:

```bash
# Python version
python --version

# Installed packages
pip list

# System info
uname -a  # macOS/Linux
systeminfo  # Windows

# GPU info (if applicable)
nvidia-smi
```

---

<p align="center">
  <strong>🚀 Installation complete! Start detecting PPE compliance! 🚀</strong>
</p>

<p align="center">
  <sub>Installation Guide | Last Updated: November 29, 2025</sub>
</p>
