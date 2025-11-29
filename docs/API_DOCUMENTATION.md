# API Documentation

> **Complete REST API reference for the PPE Detection System**

---

## Table of Contents
- [Base URL](#base-url)
- [Authentication](#authentication)
- [Core Endpoints](#core-endpoints)
  - [Detection API](#detection-api)
  - [History API](#history-api)
  - [Health Check](#health-check)
- [Request/Response Formats](#requestresponse-formats)
- [Error Handling](#error-handling)
- [Code Examples](#code-examples)

---

## Base URL

**Development:**
```
http://127.0.0.1:8000
```

**Production:**
```
https://your-domain.com/api
```

---

## Authentication

Currently, the API is **publicly accessible** without authentication for development purposes.

For production deployment, consider implementing:
- API key authentication
- JWT tokens
- Rate limiting

---

## Core Endpoints

### Detection API

#### POST /detect-helmet

Analyze an image for PPE compliance.

**Request:**
- **Method:** POST
- **Content-Type:** multipart/form-data
- **Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `file` | File | Yes | - | Image file (JPEG, PNG) |
| `threshold` | int | No | 120 | Distance threshold in pixels |
| `save_to_history` | bool | No | false | Save detection to database |
| `require_helmet` | bool | No | true | Helmet required for compliance |
| `require_vest` | bool | No | true | Vest required for compliance |
| `require_gloves` | bool | No | false | Gloves required for compliance |

**Example Request:**

```bash
curl -X POST "http://127.0.0.1:8000/detect-helmet" \
  -F "file=@worker.jpg" \
  -F "threshold=120" \
  -F "save_to_history=true" \
  -F "require_helmet=true" \
  -F "require_vest=true" \
  -F "require_gloves=false"
```

**Response (200 OK):**

```json
{
  "total_people": 2,
  "compliant_people": 1,
  "non_compliant_people": 1,
  "compliance_rate": 0.5000,
  "threshold_used": 120,
  "person_analyses": [
    {
      "person_id": 1,
      "person_box": [120, 80, 380, 560],
      "overall_compliant": true,
      "status": "COMPLIANT",
      "reason": "Helmet OK, Vest OK",
      "detected_gear": [
        "Hardhat 95% [OK]",
        "Safety Vest 87% [OK]"
      ],
      "ppe_items": [
        {
          "bbox": [150, 90, 220, 160],
          "confidence": 0.95,
          "class_name": "Hardhat",
          "is_valid_location": true,
          "location_reason": "On Head (Dist 12<147)"
        },
        {
          "bbox": [140, 200, 360, 420],
          "confidence": 0.87,
          "class_name": "Safety Vest",
          "is_valid_location": true,
          "location_reason": "On Torso (Dist 8<180)"
        }
      ],
      "landmarks": {
        "nose": [190, 125],
        "ears": [[170, 130], [210, 130]],
        "torso_center": [250, 310],
        "shoulders": [[220, 240], [280, 240]],
        "wrists": [[180, 440], [320, 440]]
      }
    },
    {
      "person_id": 2,
      "person_box": [450, 100, 680, 590],
      "overall_compliant": false,
      "status": "NON-COMPLIANT",
      "reason": "Helmet OK, Missing Vest",
      "detected_gear": [
        "Hardhat 88% [OK]",
        "No-vest 72% [BAD]"
      ],
      "ppe_items": [
        {
          "bbox": [480, 110, 550, 180],
          "confidence": 0.88,
          "class_name": "Hardhat",
          "is_valid_location": true,
          "location_reason": "On Head (Dist 15<147)"
        },
        {
          "bbox": [500, 320, 630, 450],
          "confidence": 0.72,
          "class_name": "No-vest",
          "is_valid_location": false,
          "location_reason": "Missing Required PPE"
        }
      ],
      "landmarks": {
        "nose": [515, 145],
        "ears": [[495, 150], [535, 150]],
        "torso_center": [565, 350],
        "shoulders": [[535, 280], [595, 280]],
        "wrists": [[510, 480], [620, 480]]
      }
    }
  ],
  "saved_to_history": true,
  "history_id": "550e8400-e29b-41d4-a716-446655440000",
  "image_url": "https://your-supabase-url.supabase.co/storage/v1/object/public/detection-images/2025-11-29_14-30-45_abc123.jpg"
}
```

**Error Responses:**

```json
// 400 Bad Request - No file uploaded
{
  "detail": "No file provided"
}

// 400 Bad Request - Invalid file type
{
  "detail": "Invalid file type. Supported: jpg, jpeg, png"
}

// 500 Internal Server Error - Processing failed
{
  "detail": "Error processing image: [error message]"
}
```

---

### History API

#### GET /history

Retrieve detection history with pagination and filtering.

**Request:**
- **Method:** GET
- **Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `limit` | int | No | 20 | Records per page (max 100) |
| `offset` | int | No | 0 | Pagination offset |
| `sort_by` | str | No | created_at | Sort field |
| `sort_order` | str | No | desc | asc or desc |
| `filter_compliance` | str | No | all | all, compliant, non_compliant |

**Example Request:**

```bash
curl "http://127.0.0.1:8000/history?limit=10&offset=0&filter_compliance=non_compliant"
```

**Response (200 OK):**

```json
{
  "total_count": 46,
  "records": [
    {
      "id": "550e8400-e29b-41d4-a716-446655440000",
      "created_at": "2025-11-29T14:30:45.123Z",
      "image_url": "https://...",
      "image_path": "detection-images/2025-11-29_14-30-45_abc123.jpg",
      "original_filename": "worker1.jpg",
      "total_people": 2,
      "compliant_people": 1,
      "non_compliant_people": 1,
      "compliance_rate": 0.5000,
      "threshold_used": 120,
      "person_analyses": [...] // Full analysis array
    }
    // ... more records
  ],
  "limit": 10,
  "offset": 0,
  "has_more": true
}
```

---

#### GET /history/{record_id}

Get specific detection record by ID.

**Request:**
```bash
curl "http://127.0.0.1:8000/history/550e8400-e29b-41d4-a716-446655440000"
```

**Response (200 OK):**

```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "created_at": "2025-11-29T14:30:45.123Z",
  "image_url": "https://...",
  "total_people": 2,
  "compliant_people": 1,
  "non_compliant_people": 1,
  "compliance_rate": 0.5000,
  "person_analyses": [...]
}
```

**Error Response:**

```json
// 404 Not Found
{
  "detail": "Record not found"
}
```

---

#### GET /history/stats/summary

Get overall statistics across all detections.

**Request:**
```bash
curl "http://127.0.0.1:8000/history/stats/summary"
```

**Response (200 OK):**

```json
{
  "total_scans": 46,
  "total_people_analyzed": 65,
  "total_compliant": 29,
  "total_violations": 36,
  "overall_compliance_rate": 0.4462,
  "average_people_per_scan": 1.41,
  "date_range": {
    "earliest": "2025-11-15T10:20:30.000Z",
    "latest": "2025-11-29T14:30:45.123Z"
  }
}
```

---

#### DELETE /history/{record_id}

Delete specific detection record.

**Request:**
```bash
curl -X DELETE "http://127.0.0.1:8000/history/550e8400-e29b-41d4-a716-446655440000"
```

**Response (200 OK):**

```json
{
  "message": "Record deleted successfully",
  "deleted_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

**Error Response:**

```json
// 404 Not Found
{
  "detail": "Record not found"
}
```

---

#### DELETE /history

Clear all detection history (requires confirmation).

**Request:**

```bash
curl -X DELETE "http://127.0.0.1:8000/history?confirm=true"
```

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `confirm` | bool | Yes | Must be true to confirm deletion |

**Response (200 OK):**

```json
{
  "message": "All history cleared successfully",
  "deleted_count": 46
}
```

**Error Response:**

```json
// 400 Bad Request - Missing confirmation
{
  "detail": "Confirmation required. Add ?confirm=true to URL"
}
```

---

### Health Check

#### GET /health

Check API and service health status.

**Request:**
```bash
curl "http://127.0.0.1:8000/health"
```

**Response (200 OK):**

```json
{
  "status": "healthy",
  "timestamp": "2025-11-29T14:30:45.123Z",
  "services": {
    "person_detector": "loaded",
    "ppe_detector": "loaded",
    "mediapipe": "initialized",
    "supabase": "connected"
  },
  "version": "2.0.0"
}
```

**Error Response:**

```json
// 503 Service Unavailable - Service down
{
  "status": "unhealthy",
  "timestamp": "2025-11-29T14:30:45.123Z",
  "services": {
    "person_detector": "loaded",
    "ppe_detector": "loaded",
    "mediapipe": "initialized",
    "supabase": "connection_failed"
  },
  "error": "Supabase connection timeout"
}
```

---

## Request/Response Formats

### Image Upload Format

**Supported Formats:**
- JPEG (.jpg, .jpeg)
- PNG (.png)

**Size Limits:**
- Maximum file size: 10 MB
- Recommended: < 5 MB for faster processing

**Resolution:**
- Minimum: 320×240
- Maximum: 4096×4096
- Recommended: 640×480 to 1920×1080

---

### Response Status Codes

| Code | Description | Usage |
|------|-------------|-------|
| 200 | OK | Successful request |
| 400 | Bad Request | Invalid parameters or missing file |
| 404 | Not Found | Resource doesn't exist |
| 422 | Unprocessable Entity | Validation error |
| 500 | Internal Server Error | Processing error |
| 503 | Service Unavailable | Service down or overloaded |

---

### Error Response Format

All errors follow this structure:

```json
{
  "detail": "Human-readable error message",
  "error_code": "OPTIONAL_ERROR_CODE",
  "timestamp": "2025-11-29T14:30:45.123Z"
}
```

**Common Error Codes:**

| Code | Description |
|------|-------------|
| `FILE_MISSING` | No file provided in request |
| `INVALID_FILE_TYPE` | Unsupported file format |
| `FILE_TOO_LARGE` | File exceeds size limit |
| `PROCESSING_ERROR` | Model inference failed |
| `DATABASE_ERROR` | Supabase operation failed |
| `VALIDATION_ERROR` | Invalid parameter values |

---

## Code Examples

### Python (requests)

```python
import requests

# Upload image for detection
url = "http://127.0.0.1:8000/detect-helmet"
files = {'file': open('worker.jpg', 'rb')}
data = {
    'threshold': 120,
    'save_to_history': True,
    'require_helmet': True,
    'require_vest': True,
    'require_gloves': False
}

response = requests.post(url, files=files, data=data)
result = response.json()

print(f"Total people: {result['total_people']}")
print(f"Compliant: {result['compliant_people']}")
print(f"Compliance rate: {result['compliance_rate']:.2%}")

for person in result['person_analyses']:
    print(f"\nPerson {person['person_id']}: {person['status']}")
    print(f"Reason: {person['reason']}")
    for gear in person['detected_gear']:
        print(f"  - {gear}")
```

---

### JavaScript (Fetch API)

```javascript
// Upload image from file input
const fileInput = document.getElementById('imageFile');
const file = fileInput.files[0];

const formData = new FormData();
formData.append('file', file);
formData.append('threshold', 120);
formData.append('save_to_history', true);
formData.append('require_helmet', true);
formData.append('require_vest', true);
formData.append('require_gloves', false);

fetch('http://127.0.0.1:8000/detect-helmet', {
  method: 'POST',
  body: formData
})
.then(response => response.json())
.then(data => {
  console.log('Total people:', data.total_people);
  console.log('Compliant:', data.compliant_people);
  console.log('Compliance rate:', (data.compliance_rate * 100).toFixed(2) + '%');
  
  data.person_analyses.forEach(person => {
    console.log(`\nPerson ${person.person_id}: ${person.status}`);
    console.log(`Reason: ${person.reason}`);
    person.detected_gear.forEach(gear => {
      console.log(`  - ${gear}`);
    });
  });
})
.catch(error => console.error('Error:', error));
```

---

### Python (Async)

```python
import aiohttp
import asyncio

async def detect_ppe(image_path):
    """Async PPE detection"""
    async with aiohttp.ClientSession() as session:
        data = aiohttp.FormData()
        data.add_field('file',
                      open(image_path, 'rb'),
                      filename='image.jpg',
                      content_type='image/jpeg')
        data.add_field('threshold', '120')
        data.add_field('save_to_history', 'true')
        
        async with session.post(
            'http://127.0.0.1:8000/detect-helmet',
            data=data
        ) as response:
            return await response.json()

# Usage
result = asyncio.run(detect_ppe('worker.jpg'))
print(result)
```

---

### cURL (Advanced)

```bash
# Save response to file
curl -X POST "http://127.0.0.1:8000/detect-helmet" \
  -F "file=@worker.jpg" \
  -F "threshold=120" \
  -F "save_to_history=true" \
  -o result.json

# Pretty print JSON response
curl -X POST "http://127.0.0.1:8000/detect-helmet" \
  -F "file=@worker.jpg" \
  -F "threshold=120" | jq '.'

# Check only compliance rate
curl -X POST "http://127.0.0.1:8000/detect-helmet" \
  -F "file=@worker.jpg" | jq '.compliance_rate'
```

---

### Batch Processing (Python)

```python
import requests
from pathlib import Path
import time

def batch_detect(image_folder, save_results=True):
    """Process multiple images"""
    url = "http://127.0.0.1:8000/detect-helmet"
    results = []
    
    for image_path in Path(image_folder).glob('*.jpg'):
        print(f"Processing {image_path.name}...")
        
        files = {'file': open(image_path, 'rb')}
        data = {
            'threshold': 120,
            'save_to_history': save_results
        }
        
        try:
            response = requests.post(url, files=files, data=data)
            response.raise_for_status()
            result = response.json()
            results.append({
                'filename': image_path.name,
                'compliance_rate': result['compliance_rate'],
                'total_people': result['total_people']
            })
        except Exception as e:
            print(f"Error processing {image_path.name}: {e}")
        
        time.sleep(0.5)  # Rate limiting
    
    return results

# Usage
results = batch_detect('images/')
for r in results:
    print(f"{r['filename']}: {r['compliance_rate']:.2%} ({r['total_people']} people)")
```

---

## WebSocket Support (Future)

**Planned for Phase 3:**

Real-time detection updates via WebSocket connection:

```javascript
// Future implementation
const ws = new WebSocket('ws://127.0.0.1:8000/ws/detection');

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('Real-time detection:', data);
};

// Send video frame
ws.send(JSON.stringify({
  type: 'frame',
  data: base64EncodedFrame
}));
```

---

## Rate Limiting

**Current Limits (Development):**
- No rate limiting implemented

**Recommended for Production:**
- 100 requests per minute per IP
- 1000 requests per hour per API key
- Concurrent requests: 5 per client

**Implementation Example:**

```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@app.post("/detect-helmet")
@limiter.limit("100/minute")
async def detect_helmet(...):
    # Endpoint logic
    pass
```

---

## CORS Configuration

**Current Settings:**

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:8000",
        "http://127.0.0.1:5500"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

**For Production:**
- Restrict `allow_origins` to your domain
- Consider API key authentication
- Enable HTTPS only

---

<p align="center">
  <strong>📡 Complete API reference for PPE Detection System 📡</strong>
</p>

<p align="center">
  <sub>API Documentation | Last Updated: November 29, 2025</sub>
</p>
