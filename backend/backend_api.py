"""
FastAPI Backend for Helmet Compliance Detection
Provides endpoints for image upload and real-time camera detection

Author: Zulfaqar
Project: INF3001 Deep Learning - PPE Detection API
"""

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import cv2
import numpy as np
from helmet_compliance_detector import HelmetComplianceDetector
import io
from typing import Optional

# Initialize FastAPI app
app = FastAPI(
    title="PPE Helmet Compliance API",
    description="Real-time helmet compliance detection system",
    version="1.0.0"
)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize detector (load models once at startup)
print("Initializing Helmet Compliance Detector...")
detector = HelmetComplianceDetector(
    ppe_model_path='../ppe-4080-v12/weights/best.pt',
    confidence_threshold=0.5
)
print("✓ Detector ready!")


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "service": "PPE Helmet Compliance API",
        "version": "1.0.0"
    }


@app.post("/detect-helmet")
async def detect_helmet(
    file: UploadFile = File(...),
    threshold: Optional[int] = Form(None)
):
    """
    Detect helmet compliance in uploaded image.
    
    Parameters:
    - file: Image file (JPG, PNG)
    - threshold: Optional threshold in pixels (default: 120)
    
    Returns:
    - total_people: Number of people detected
    - compliant_people: Number wearing helmets
    - non_compliant_people: Number not wearing helmets
    - compliance_rate: Percentage (0-1)
    - person_analyses: Detailed analysis for each person
    """
    try:
        # Read image file
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return JSONResponse(
                status_code=400,
                content={"error": "Invalid image file"}
            )
        
        # Adjust threshold if provided
        if threshold is not None:
            detector.HELMET_ON_HEAD_THRESHOLD = int(threshold)
        
        # Process frame
        _, results = detector.process_frame(frame, visualize=False)
        
        # Format response
        response = {
            "total_people": results['total_people'],
            "compliant_people": results['compliant'],
            "non_compliant_people": results['non_compliant'],
            "compliance_rate": results['compliance_rate'],
            "person_analyses": []
        }
        
        # Add detailed analysis for each person
        for i, analysis in enumerate(results['analyses']):
            person_data = {
                "person_id": i + 1,
                "person_box": analysis['person_box'],
                "head_detected": analysis['head_detected'],
                "has_helmet": analysis.get('has_helmet', False),
                "overall_compliant": analysis['compliant'],
                "status": analysis['status'],
                "reason": analysis['reason']
            }
            
            # Add optional fields if available
            if 'head_position' in analysis:
                person_data['head_position'] = analysis['head_position']
            
            if 'distance_to_head' in analysis and analysis['distance_to_head'] is not None:
                person_data['distance_to_head'] = round(analysis['distance_to_head'], 2)
            
            if 'helmet' in analysis:
                person_data['helmet_bbox'] = analysis['helmet']['bbox']
                person_data['helmet_confidence'] = round(analysis['helmet']['confidence'], 3)
            
            response['person_analyses'].append(person_data)
        
        return response
        
    except Exception as e:
        print(f"Error processing image: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Processing failed: {str(e)}"}
        )


@app.get("/config")
async def get_config():
    """Get current detector configuration"""
    return {
        "helmet_threshold": detector.HELMET_ON_HEAD_THRESHOLD,
        "confidence_threshold": detector.confidence_threshold,
        "model_loaded": detector.ppe_model is not None
    }


@app.post("/config/threshold")
async def update_threshold(threshold: int):
    """Update helmet detection threshold"""
    if threshold < 20 or threshold > 500:
        return JSONResponse(
            status_code=400,
            content={"error": "Threshold must be between 20 and 500"}
        )
    
    detector.HELMET_ON_HEAD_THRESHOLD = threshold
    return {
        "success": True,
        "new_threshold": threshold
    }


if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*70)
    print("PPE HELMET COMPLIANCE API SERVER")
    print("="*70)
    print("\nStarting server...")
    print("API will be available at: http://127.0.0.1:8000")
    print("API docs at: http://127.0.0.1:8000/docs")
    print("\nPress CTRL+C to stop")
    print("="*70 + "\n")
    
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
