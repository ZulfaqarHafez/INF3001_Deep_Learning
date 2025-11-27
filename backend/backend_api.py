"""
FastAPI Backend for Helmet Compliance Detection
With Supabase integration for image storage and history logging

Author: Zulfaqar
Project: INF3001 Deep Learning - PPE Detection API v2.0

Updated: Now loads PPE model from Hugging Face automatically
"""

from fastapi import FastAPI, File, UploadFile, Form, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import cv2
import numpy as np
from helmet_compliance_detector import HelmetComplianceDetector
from supabase import create_client, Client
from typing import Optional, List
from datetime import datetime
import uuid
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="PPE Helmet Compliance API",
    description="Real-time helmet compliance detection with Supabase history logging",
    version="2.0.0"
)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# SUPABASE CONFIGURATION
# =============================================================================
SUPABASE_URL = os.getenv("SUPABASE_URL", "https://your-project-id.supabase.co")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "your-anon-public-key")

# Initialize Supabase client
try:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    print(f"✓ Supabase client initialized")
    print(f"  URL: {SUPABASE_URL[:50]}...")
except Exception as e:
    print(f"✗ Failed to initialize Supabase: {e}")
    supabase = None

# =============================================================================
# DETECTOR INITIALIZATION - Using Hugging Face Model
# =============================================================================
print("\n" + "="*70)
print("INITIALIZING HELMET COMPLIANCE DETECTOR")
print("="*70)

# Configuration - Choose your model source
USE_HUGGINGFACE = os.getenv("USE_HUGGINGFACE", "true").lower() == "true"
HUGGINGFACE_REPO = os.getenv("HUGGINGFACE_REPO", "iMaximusiV/yolo-ppe-detector")
HUGGINGFACE_FILENAME = os.getenv("HUGGINGFACE_FILENAME", "best.pt")
LOCAL_MODEL_PATH = os.getenv("LOCAL_MODEL_PATH", "../ppe-4080-v12/weights/best.pt")

try:
    if USE_HUGGINGFACE:
        print(f"\n[Config] Using Hugging Face model: {HUGGINGFACE_REPO}/{HUGGINGFACE_FILENAME}")
        detector = HelmetComplianceDetector(
            huggingface_repo=HUGGINGFACE_REPO,
            huggingface_filename=HUGGINGFACE_FILENAME,
            use_huggingface=True,
            confidence_threshold=0.5
        )
    else:
        print(f"\n[Config] Using local model: {LOCAL_MODEL_PATH}")
        detector = HelmetComplianceDetector(
            ppe_model_path=LOCAL_MODEL_PATH,
            confidence_threshold=0.5
        )
    print("✓ Detector ready!")
except Exception as e:
    print(f"✗ Failed to initialize detector: {e}")
    print("  Check model path or Hugging Face connection")
    detector = None


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

async def upload_image_to_supabase(image_bytes: bytes, filename: str) -> dict:
    """
    Upload image to Supabase Storage bucket.
    """
    if not supabase:
        return None
    
    try:
        # Generate unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        extension = filename.split('.')[-1] if filename and '.' in filename else 'jpg'
        storage_path = f"detections/{timestamp}_{unique_id}.{extension}"
        
        # Determine content type
        content_type = "image/jpeg"
        if extension.lower() == "png":
            content_type = "image/png"
        elif extension.lower() == "webp":
            content_type = "image/webp"
        
        # Upload to Supabase Storage
        supabase.storage.from_('detection-images').upload(
            path=storage_path,
            file=image_bytes,
            file_options={"content-type": content_type}
        )
        
        # Get public URL
        public_url = supabase.storage.from_('detection-images').get_public_url(storage_path)
        
        return {
            "path": storage_path,
            "url": public_url
        }
        
    except Exception as e:
        print(f"Error uploading image: {e}")
        return None


async def save_detection_record(
    image_url: str,
    image_path: str,
    original_filename: str,
    results: dict,
    threshold: int
) -> dict:
    """
    Save detection record to Supabase database.
    """
    if not supabase:
        return None
    
    try:
        record = {
            "image_url": image_url,
            "image_path": image_path,
            "original_filename": original_filename,
            "total_people": results.get('total_people', 0),
            "compliant_people": results.get('compliant_people', 0),
            "non_compliant_people": results.get('non_compliant_people', 0),
            "compliance_rate": results.get('compliance_rate', 0),
            "person_analyses": results.get('person_analyses', []),
            "threshold_used": threshold
        }
        
        response = supabase.table('detection_history').insert(record).execute()
        
        if response.data and len(response.data) > 0:
            return response.data[0]
        return None
        
    except Exception as e:
        print(f"Error saving detection record: {e}")
        return None


# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.get("/")
async def root():
    """Health check and API info endpoint"""
    return {
        "status": "online",
        "service": "PPE Helmet Compliance API",
        "version": "2.0.0"
    }


@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "api": "healthy",
        "detector": "ready" if detector else "unavailable",
        "supabase": "connected" if supabase else "disconnected"
    }


# =============================================================================
# DETECTION ENDPOINTS
# =============================================================================

@app.post("/detect-helmet")
async def detect_helmet(
    file: UploadFile = File(...),
    threshold: Optional[int] = Form(None),
    save_to_history: Optional[bool] = Form(True)
):
    """
    Detect helmet compliance in uploaded image.
    Updated to return full PPE audit data AND landmarks.
    """
    if not detector:
        return JSONResponse(
            status_code=503,
            content={"error": "Detection service unavailable"}
        )
    
    try:
        # Read image file
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return JSONResponse(
                status_code=400,
                content={"error": "Invalid image file. Please upload a valid JPG or PNG."}
            )
        
        # Store original threshold
        original_threshold = detector.HELMET_ON_HEAD_THRESHOLD
        
        # Adjust threshold if provided
        if threshold is not None:
            if threshold < 20 or threshold > 500:
                return JSONResponse(
                    status_code=400,
                    content={"error": "Threshold must be between 20 and 500"}
                )
            detector.HELMET_ON_HEAD_THRESHOLD = int(threshold)
        
        # Process frame (visualize=False because we draw on frontend)
        _, results = detector.process_frame(frame, visualize=False)
        
        # Build response
        response = {
            "total_people": results['total_people'],
            "compliant_people": results['compliant'],
            "non_compliant_people": results['non_compliant'],
            "compliance_rate": results['compliance_rate'],
            "threshold_used": detector.HELMET_ON_HEAD_THRESHOLD,
            "person_analyses": [],
            "raw_ppe_items": results.get('all_ppe_detected', [])  # Raw items for potential debugging
        }
        
        # Add detailed analysis for each person
        for i, analysis in enumerate(results['analyses']):
            person_data = {
                "person_id": i + 1,
                "person_box": analysis['person_box'],
                "head_detected": analysis['head_detected'],
                "has_helmet": analysis.get('has_helmet', False),
                "overall_compliant": analysis.get('compliant', False),
                "status": analysis.get('status', 'Unknown'),
                "reason": analysis.get('reason', ''),
                
                # New Fields for Full Audit
                "detected_gear": analysis.get('detected_gear', []),
                "ppe_items": analysis.get('ppe_items', [])
            }
            
            # Add optional fields if available
            if 'head_position' in analysis:
                person_data['head_position'] = analysis['head_position']
            
            if 'distance_to_head' in analysis and analysis['distance_to_head'] is not None:
                person_data['distance_to_head'] = round(analysis['distance_to_head'], 2)
                
            if 'helmet' in analysis:
                person_data['helmet_bbox'] = analysis['helmet']['bbox']
                person_data['helmet_confidence'] = round(analysis['helmet']['confidence'], 3)
            
            # --- CRITICAL UPDATE: PASS LANDMARKS ---
            if 'landmarks' in analysis:
                person_data['landmarks'] = analysis['landmarks']
            
            response['person_analyses'].append(person_data)
        
        # Restore original threshold
        detector.HELMET_ON_HEAD_THRESHOLD = original_threshold
        
        # Save to history if requested and Supabase is available
        if save_to_history and supabase:
            # Upload image
            upload_result = await upload_image_to_supabase(contents, file.filename)
            
            if upload_result:
                # Save record
                record = await save_detection_record(
                    image_url=upload_result['url'],
                    image_path=upload_result['path'],
                    original_filename=file.filename,
                    results=response,
                    threshold=response['threshold_used']
                )
                
                if record:
                    response['history_id'] = record.get('id')
                    response['image_url'] = record.get('image_url')
                    response['saved_to_history'] = True
                else:
                    response['saved_to_history'] = False
                    response['save_error'] = "Failed to save record"
            else:
                response['saved_to_history'] = False
                response['save_error'] = "Failed to upload image"
        else:
            response['saved_to_history'] = False
        
        return response
        
    except Exception as e:
        print(f"Error processing image: {e}")
        import traceback
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"error": f"Processing failed: {str(e)}"}
        )


# =============================================================================
# HISTORY ENDPOINTS
# =============================================================================

@app.get("/history")
async def get_history(
    limit: int = Query(default=20, ge=1, le=100, description="Number of records to return"),
    offset: int = Query(default=0, ge=0, description="Number of records to skip"),
    compliant_only: Optional[bool] = Query(default=None, description="Filter by compliance status")
):
    """
    Fetch detection history with pagination and filtering.
    """
    if not supabase:
        return JSONResponse(
            status_code=503,
            content={"error": "History service unavailable"}
        )
    
    try:
        query = supabase.table('detection_history')\
            .select('*')\
            .order('created_at', desc=True)
        
        if compliant_only is True:
            query = query.eq('non_compliant_people', 0)
        elif compliant_only is False:
            query = query.gt('non_compliant_people', 0)
        
        response = query.range(offset, offset + limit - 1).execute()
        
        return {
            "records": response.data,
            "count": len(response.data),
            "offset": offset,
            "limit": limit,
            "has_more": len(response.data) == limit
        }
        
    except Exception as e:
        print(f"Error fetching history: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to fetch history: {str(e)}"}
        )


@app.get("/history/stats/summary")
async def get_stats_summary():
    """Get overall detection statistics summary."""
    if not supabase:
        return JSONResponse(
            status_code=503,
            content={"error": "History service unavailable"}
        )
    
    try:
        response = supabase.table('detection_history')\
            .select('total_people, compliant_people, non_compliant_people, compliance_rate')\
            .execute()
        
        records = response.data
        
        if not records:
            return {
                "total_scans": 0,
                "total_people": 0,
                "total_compliant": 0,
                "total_non_compliant": 0,
                "overall_compliance_rate": 0,
                "average_people_per_scan": 0
            }
        
        total_people = sum(r['total_people'] for r in records)
        total_compliant = sum(r['compliant_people'] for r in records)
        total_non_compliant = sum(r['non_compliant_people'] for r in records)
        
        return {
            "total_scans": len(records),
            "total_people": total_people,
            "total_compliant": total_compliant,
            "total_non_compliant": total_non_compliant,
            "overall_compliance_rate": round(total_compliant / total_people, 4) if total_people > 0 else 0,
            "average_people_per_scan": round(total_people / len(records), 2) if records else 0,
            "scans_with_violations": sum(1 for r in records if r['non_compliant_people'] > 0),
            "fully_compliant_scans": sum(1 for r in records if r['non_compliant_people'] == 0)
        }
        
    except Exception as e:
        print(f"Error fetching stats: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to fetch stats: {str(e)}"}
        )


@app.get("/history/{record_id}")
async def get_history_record(record_id: str):
    """Fetch a specific detection record by ID."""
    if not supabase:
        return JSONResponse(
            status_code=503,
            content={"error": "History service unavailable"}
        )
    
    try:
        response = supabase.table('detection_history')\
            .select('*')\
            .eq('id', record_id)\
            .single()\
            .execute()
        
        if response.data:
            return response.data
        else:
            return JSONResponse(
                status_code=404,
                content={"error": "Record not found"}
            )
            
    except Exception as e:
        print(f"Error fetching record: {e}")
        return JSONResponse(
            status_code=404,
            content={"error": "Record not found"}
        )


@app.delete("/history/{record_id}")
async def delete_history_record(record_id: str):
    """Delete a detection record and its associated image."""
    if not supabase:
        return JSONResponse(
            status_code=503,
            content={"error": "History service unavailable"}
        )
    
    try:
        # First, get the record to find the image path
        record_response = supabase.table('detection_history')\
            .select('image_path')\
            .eq('id', record_id)\
            .single()\
            .execute()
        
        if not record_response.data:
            return JSONResponse(
                status_code=404,
                content={"error": "Record not found"}
            )
        
        image_path = record_response.data.get('image_path')
        
        # Delete image from storage
        if image_path:
            try:
                supabase.storage.from_('detection-images').remove([image_path])
            except Exception as e:
                print(f"Warning: Could not delete image {image_path}: {e}")
        
        # Delete record from database
        supabase.table('detection_history')\
            .delete()\
            .eq('id', record_id)\
            .execute()
        
        return {
            "success": True,
            "deleted_id": record_id,
            "message": "Record and associated image deleted successfully"
        }
        
    except Exception as e:
        print(f"Error deleting record: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to delete record: {str(e)}"}
        )


@app.delete("/history")
async def clear_all_history(confirm: bool = Query(default=False)):
    """Delete all detection history records and images."""
    if not confirm:
        return JSONResponse(
            status_code=400,
            content={"error": "Set confirm=true to delete all history"}
        )
    
    if not supabase:
        return JSONResponse(
            status_code=503,
            content={"error": "History service unavailable"}
        )
    
    try:
        # Get all image paths
        records_response = supabase.table('detection_history')\
            .select('image_path')\
            .execute()
        
        image_paths = [r['image_path'] for r in records_response.data if r.get('image_path')]
        
        # Delete all images
        if image_paths:
            try:
                supabase.storage.from_('detection-images').remove(image_paths)
            except Exception as e:
                print(f"Warning: Error deleting some images: {e}")
        
        # Delete all records
        supabase.table('detection_history')\
            .delete()\
            .neq('id', '00000000-0000-0000-0000-000000000000')\
            .execute()
        
        return {
            "success": True,
            "deleted_records": len(records_response.data),
            "deleted_images": len(image_paths)
        }
        
    except Exception as e:
        print(f"Error clearing history: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to clear history: {str(e)}"}
        )


# =============================================================================
# CONFIGURATION ENDPOINTS
# =============================================================================

@app.get("/config")
async def get_config():
    """Get current detector configuration"""
    if not detector:
        return JSONResponse(
            status_code=503,
            content={"error": "Detector unavailable"}
        )
    
    return {
        "helmet_threshold": detector.HELMET_ON_HEAD_THRESHOLD,
        "confidence_threshold": detector.confidence_threshold,
        "model_loaded": detector.ppe_model is not None,
        "model_source": "huggingface" if USE_HUGGINGFACE else "local",
        "huggingface_repo": HUGGINGFACE_REPO if USE_HUGGINGFACE else None,
        "supabase_connected": supabase is not None
    }


@app.post("/config/threshold")
async def update_threshold(threshold: int = Query(..., ge=20, le=500)):
    """Update helmet detection threshold."""
    if not detector:
        return JSONResponse(
            status_code=503,
            content={"error": "Detector unavailable"}
        )
    
    old_threshold = detector.HELMET_ON_HEAD_THRESHOLD
    detector.HELMET_ON_HEAD_THRESHOLD = threshold
    
    return {
        "success": True,
        "old_threshold": old_threshold,
        "new_threshold": threshold
    }


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "="*70)
    print("PPE HELMET COMPLIANCE API SERVER v2.0")
    print("="*70)
    print("\nConfiguration:")
    print(f"  • Detector: {'Ready' if detector else 'Unavailable'}")
    print(f"  • Model Source: {'Hugging Face' if USE_HUGGINGFACE else 'Local'}")
    if USE_HUGGINGFACE:
        print(f"  • HF Repo: {HUGGINGFACE_REPO}")
    print(f"  • Supabase: {'Connected' if supabase else 'Disconnected'}")
    print("\nEndpoints:")
    print("  • API: http://127.0.0.1:8000")
    print("  • Docs: http://127.0.0.1:8000/docs")
    print("  • Health: http://127.0.0.1:8000/health")
    print("\nPress CTRL+C to stop")
    print("="*70 + "\n")
    
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")