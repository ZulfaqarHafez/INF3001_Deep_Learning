# Technical Implementation

> **Deep dive into system architecture, algorithms, and technical design decisions**

---

## Table of Contents
- [System Architecture](#system-architecture)
- [Core Algorithms](#core-algorithms)
- [Detection Pipeline](#detection-pipeline)
- [Pose-Aware Validation](#pose-aware-validation)
- [Dynamic Requirements System](#dynamic-requirements-system)
- [Performance Optimization](#performance-optimization)

---

## System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      Frontend (Vanilla JS)                       │
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ Image Upload │  │ Live Camera  │  │   History    │          │
│  │   - Drag/    │  │  - WebRTC    │  │  - Filter    │          │
│  │     Drop     │  │  - Snapshot  │  │  - Paginate  │          │
│  │   - Analyze  │  │  - Real-time │  │  - Detail    │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │             Sidebar (Dynamic Requirements)                │   │
│  │  ☑ Helmet/Hardhat  ☑ Safety Vest  ☐ Gloves             │   │
│  └──────────────────────────────────────────────────────────┘   │
└────────────────────────────┬─────────────────────────────────────┘
                             │ REST API (JSON over HTTP)
                             │
┌────────────────────────────▼─────────────────────────────────────┐
│                    FastAPI Backend (Python)                       │
│                                                                   │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │              Detection Pipeline Manager                     │  │
│  │                                                             │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌────────────┐       │  │
│  │  │   YOLOv8n    │  │  YOLOv8 PPE  │  │ MediaPipe  │       │  │
│  │  │ Person Det   │→ │  Gear Det    │→ │ Pose Est   │       │  │
│  │  │ (COCO)       │  │ (Custom)     │  │ (33 points)│       │  │
│  │  └──────────────┘  └──────────────┘  └────────────┘       │  │
│  │          ↓                  ↓                ↓             │  │
│  │  ┌────────────────────────────────────────────────────┐   │  │
│  │  │        Spatial Validation Engine                   │   │  │
│  │  │  • Euclidean Distance Calculation                  │   │  │
│  │  │  • Bounding Box Containment Check                  │   │  │
│  │  │  • Adaptive Threshold Scaling                      │   │  │
│  │  └────────────────────────────────────────────────────┘   │  │
│  │                          ↓                                 │  │
│  │  ┌────────────────────────────────────────────────────┐   │  │
│  │  │       Dynamic Compliance Evaluator                 │   │  │
│  │  │  • Configurable Requirements (Helmet/Vest/Gloves)  │   │  │
│  │  │  • Per-Person Analysis                             │   │  │
│  │  │  • Aggregate Statistics                            │   │  │
│  │  └────────────────────────────────────────────────────┘   │  │
│  └────────────────────────────────────────────────────────────┘  │
│                              ↓                                    │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │              Response Formatter                             │  │
│  │  • JSON serialization                                       │  │
│  │  • Image annotation coordinates                             │  │
│  │  • Confidence scores                                        │  │
│  └────────────────────────────────────────────────────────────┘  │
└────────────────────────────┬──────────────────────────────────────┘
                             │
┌────────────────────────────▼──────────────────────────────────────┐
│                    Supabase Cloud Storage                          │
│                                                                    │
│  ┌──────────────────┐          ┌──────────────────┐              │
│  │   PostgreSQL     │          │  Object Storage   │              │
│  │  detection_      │          │  detection-       │              │
│  │  history table   │          │  images bucket    │              │
│  │                  │          │  (Public, CORS)   │              │
│  └──────────────────┘          └──────────────────┘              │
└───────────────────────────────────────────────────────────────────┘
```

---

## Core Algorithms

### 1. Dual-Model Detection Pipeline

#### Person Detection (YOLOv8n)
```python
def detect_persons(image):
    """
    Step 1: Detect all people in the scene
    
    Model: YOLOv8n (pretrained on COCO)
    Input: BGR image (H x W x 3)
    Output: List of person bounding boxes
    
    Returns:
        person_boxes: List[Tuple[x1, y1, x2, y2]]
        confidences: List[float]
    """
    results = person_detector.predict(
        image,
        classes=[0],  # Class 0 = person in COCO
        conf=0.5,     # Minimum confidence threshold
        verbose=False
    )
    
    person_boxes = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = box.conf[0].cpu().numpy()
            person_boxes.append({
                'bbox': (x1, y1, x2, y2),
                'confidence': conf
            })
    
    return person_boxes
```

#### PPE Detection (Custom YOLOv8)
```python
def detect_ppe(image):
    """
    Step 2: Detect all PPE items in the scene
    
    Model: Custom YOLOv8 (trained on construction PPE)
    Classes: Hardhat, Safety Vest, Mask, Cone, Machinery, Vehicle
    
    Returns:
        ppe_detections: List[Dict]
            - bbox: (x1, y1, x2, y2)
            - class_name: str
            - confidence: float
    """
    results = ppe_detector.predict(
        image,
        conf=0.4,  # Lower threshold for PPE
        iou=0.5,   # Non-max suppression threshold
        verbose=False
    )
    
    ppe_detections = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = box.conf[0].cpu().numpy()
            cls = int(box.cls[0].cpu().numpy())
            class_name = result.names[cls]
            
            ppe_detections.append({
                'bbox': (x1, y1, x2, y2),
                'class_name': class_name,
                'confidence': conf
            })
    
    return ppe_detections
```

---

### 2. Pose Estimation (MediaPipe)

#### Landmark Extraction
```python
def extract_pose_landmarks(image, person_box):
    """
    Step 3: Extract 33 body landmarks using MediaPipe
    
    Landmarks include:
    - Face: Nose, eyes, ears, mouth (11 points)
    - Torso: Shoulders, hips (6 points)
    - Arms: Elbows, wrists (4 points)
    - Legs: Knees, ankles, feet (12 points)
    
    Returns:
        landmarks: Dict[str, Tuple[int, int]]
            Key anatomical points for validation
    """
    # Crop person region
    x1, y1, x2, y2 = person_box
    person_img = image[int(y1):int(y2), int(x1):int(x2)]
    
    # MediaPipe expects RGB
    rgb_img = cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB)
    
    # Process with MediaPipe
    results = pose_estimator.process(rgb_img)
    
    if not results.pose_landmarks:
        return None
    
    # Extract key landmarks
    h, w = person_img.shape[:2]
    landmarks = {}
    
    # Nose (landmark 0)
    nose = results.pose_landmarks.landmark[0]
    landmarks['nose'] = (
        int(nose.x * w + x1),
        int(nose.y * h + y1)
    )
    
    # Left/Right Ears (landmarks 7, 8)
    left_ear = results.pose_landmarks.landmark[7]
    right_ear = results.pose_landmarks.landmark[8]
    landmarks['ears'] = [
        (int(left_ear.x * w + x1), int(left_ear.y * h + y1)),
        (int(right_ear.x * w + x1), int(right_ear.y * h + y1))
    ]
    
    # Shoulders (landmarks 11, 12)
    left_shoulder = results.pose_landmarks.landmark[11]
    right_shoulder = results.pose_landmarks.landmark[12]
    landmarks['shoulders'] = [
        (int(left_shoulder.x * w + x1), int(left_shoulder.y * h + y1)),
        (int(right_shoulder.x * w + x1), int(right_shoulder.y * h + y1))
    ]
    
    # Hips (landmarks 23, 24)
    left_hip = results.pose_landmarks.landmark[23]
    right_hip = results.pose_landmarks.landmark[24]
    landmarks['hips'] = [
        (int(left_hip.x * w + x1), int(left_hip.y * h + y1)),
        (int(right_hip.x * w + x1), int(right_hip.y * h + y1))
    ]
    
    # Wrists (landmarks 15, 16)
    left_wrist = results.pose_landmarks.landmark[15]
    right_wrist = results.pose_landmarks.landmark[16]
    landmarks['wrists'] = [
        (int(left_wrist.x * w + x1), int(left_wrist.y * h + y1)),
        (int(right_wrist.x * w + x1), int(right_wrist.y * h + y1))
    ]
    
    # Compute torso center (average of shoulders and hips)
    shoulders = landmarks['shoulders']
    hips = landmarks['hips']
    torso_center_x = int(np.mean([s[0] for s in shoulders] + [h[0] for h in hips]))
    torso_center_y = int(np.mean([s[1] for s in shoulders] + [h[1] for h in hips]))
    landmarks['torso_center'] = (torso_center_x, torso_center_y)
    
    return landmarks
```

---

### 3. Spatial Validation Engine

#### Euclidean Distance Calculation
```python
def euclidean_distance(point1: Tuple[int, int], point2: Tuple[int, int]) -> float:
    """
    Calculate straight-line distance between two points in pixel space
    
    Formula: d = √[(x₂ - x₁)² + (y₂ - y₁)²]
    
    Args:
        point1: (x1, y1) coordinates
        point2: (x2, y2) coordinates
    
    Returns:
        float: Distance in pixels
    
    Example:
        point1 = (150, 85)  # Helmet center
        point2 = (155, 90)  # Nose position
        distance = euclidean_distance(point1, point2)
        # Returns: 7.07 pixels
    """
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
```

#### Dual Validation Strategy
```python
def validate_ppe_location(ppe_box, ppe_class, landmarks, person_height):
    """
    Validates PPE location using TWO independent checks:
    1. Distance Check: Item center within threshold of target landmark
    2. Containment Check: Target landmark inside item bounding box
    
    Args:
        ppe_box: (x1, y1, x2, y2) - PPE bounding box
        ppe_class: str - "Hardhat", "Safety Vest", etc.
        landmarks: Dict - Pose landmarks from MediaPipe
        person_height: int - Person bbox height in pixels
    
    Returns:
        is_valid: bool
        reason: str - Explanation of validation result
    """
    # Get PPE center point
    x1, y1, x2, y2 = ppe_box
    ppe_center = ((x1 + x2) / 2, (y1 + y2) / 2)
    
    # Define target landmarks and threshold based on PPE class
    if ppe_class == "Hardhat":
        target_landmarks = [landmarks['nose']] + landmarks['ears']
        threshold_ratio = 0.35  # 35% of person height
        body_part = "Head"
        
    elif ppe_class == "Safety Vest":
        target_landmarks = [landmarks['torso_center']] + landmarks['shoulders']
        threshold_ratio = 0.45  # 45% (more lenient for torso)
        body_part = "Torso"
        
    else:
        # Other PPE classes (Mask, Gloves) - not yet validated
        return True, "Detection Only (No Pose Validation)"
    
    # Calculate dynamic threshold
    threshold = person_height * threshold_ratio
    
    # Check 1: Minimum distance to any target landmark
    distances = [
        euclidean_distance(ppe_center, landmark)
        for landmark in target_landmarks
    ]
    min_distance = min(distances)
    
    if min_distance < threshold:
        return True, f"On {body_part} (Dist {int(min_distance)}<{int(threshold)})"
    
    # Check 2: Containment (fallback for edge cases)
    for landmark in target_landmarks:
        if is_point_in_box(landmark, ppe_box, buffer=20):
            return True, f"On {body_part} (Overlaps)"
    
    return False, f"Off {body_part} (Dist {int(min_distance)}>{int(threshold)})"


def is_point_in_box(point, box, buffer=0):
    """
    Check if a point is inside a bounding box (with optional buffer)
    
    Args:
        point: (x, y) coordinates
        box: (x1, y1, x2, y2) bounding box
        buffer: int - Pixels to expand box (for edge tolerance)
    
    Returns:
        bool: True if point is inside box
    """
    x, y = point
    x1, y1, x2, y2 = box
    return (x1 - buffer <= x <= x2 + buffer and
            y1 - buffer <= y <= y2 + buffer)
```

---

### 4. PPE-Specific Validation Rules

#### Hardhat Validation
```python
def validate_hardhat(hardhat_box, landmarks, person_height):
    """
    Hardhat validation against head landmarks
    
    Targets: Nose + Both Ears (3 points)
    Threshold: 35% of person height
    
    Rationale:
    - Nose: Center reference point for head
    - Ears: Lateral references ensure hardhat covers head width
    - 35%: Allows for various hardhat sizes and slight tilts
    
    Example Scenarios:
    
    VALID (Properly Worn):
        Hardhat Center: (150, 80)
        Nose: (155, 85) → Distance: 7.07px ✓
        Left Ear: (135, 88) → Distance: 18.03px ✓
        Person Height: 420px → Threshold: 147px
        Result: VALID (all distances < threshold)
    
    INVALID (Held in Hand):
        Hardhat Center: (200, 350)
        Nose: (155, 85) → Distance: 267.39px ✗
        Person Height: 420px → Threshold: 147px
        Result: INVALID (distance > threshold)
    """
    return validate_ppe_location(
        ppe_box=hardhat_box,
        ppe_class="Hardhat",
        landmarks=landmarks,
        person_height=person_height
    )
```

#### Safety Vest Validation
```python
def validate_vest(vest_box, landmarks, person_height):
    """
    Safety vest validation against torso landmarks
    
    Targets: Torso Center + Both Shoulders (3 points)
    Threshold: 45% of person height
    
    Rationale:
    - Torso Center: Computed as average of shoulders + hips
    - Shoulders: Ensure vest covers upper torso properly
    - 45%: More lenient than hardhat (vests have larger variation)
    
    Example:
        Vest Center: (200, 250)
        Torso Center: (198, 255) → Distance: 5.39px ✓
        Left Shoulder: (180, 200) → Distance: 53.85px ✓
        Person Height: 400px → Threshold: 180px
        Result: VALID (vest properly worn on torso)
    """
    return validate_ppe_location(
        ppe_box=vest_box,
        ppe_class="Safety Vest",
        landmarks=landmarks,
        person_height=person_height
    )
```

---

### 5. Dynamic Requirements System

#### Compliance Evaluation
```python
def evaluate_compliance(person_analysis, requirements):
    """
    Evaluate person's compliance against dynamic requirements
    
    Args:
        person_analysis: Dict containing detected PPE for one person
        requirements: Dict with configurable rules
            {
                'helmet': bool,  # Is helmet required?
                'vest': bool,    # Is vest required?
                'gloves': bool   # Are gloves required?
            }
    
    Returns:
        is_compliant: bool
        reason: str - Human-readable explanation
    
    Example:
        person_analysis = {
            'has_helmet': True,
            'has_vest': False,
            'has_gloves': True
        }
        
        requirements = {
            'helmet': True,
            'vest': True,
            'gloves': False
        }
        
        Result:
            is_compliant = False
            reason = "Helmet OK, Missing Vest"
    """
    missing_items = []
    present_items = []
    
    # Check each requirement
    if requirements.get('helmet', True):
        if person_analysis['has_helmet']:
            present_items.append("Helmet OK")
        else:
            missing_items.append("Missing Helmet")
    
    if requirements.get('vest', True):
        if person_analysis['has_vest']:
            present_items.append("Vest OK")
        else:
            missing_items.append("Missing Vest")
    
    if requirements.get('gloves', False):
        if person_analysis['has_gloves']:
            present_items.append("Gloves OK")
        else:
            missing_items.append("Missing Gloves")
    
    # Determine compliance
    is_compliant = len(missing_items) == 0
    
    # Build reason string
    if is_compliant:
        reason = ", ".join(present_items)
    else:
        reason = ", ".join(present_items + missing_items)
    
    return is_compliant, reason
```

---

## Detection Pipeline

### Complete End-to-End Flow

```python
def process_image(image, threshold=120, requirements=None):
    """
    Complete detection pipeline from image to compliance results
    
    Pipeline:
    1. Detect all persons (YOLOv8n)
    2. Detect all PPE items (Custom YOLOv8)
    3. Extract pose landmarks for each person (MediaPipe)
    4. Match PPE to persons based on bbox overlap
    5. Validate PPE locations using pose landmarks
    6. Evaluate compliance against requirements
    7. Generate annotated visualization
    8. Return structured results
    
    Args:
        image: numpy array (BGR format)
        threshold: int - Pixel threshold for distance validation
        requirements: Dict - Dynamic compliance rules
    
    Returns:
        results: Dict with complete analysis
    """
    if requirements is None:
        requirements = {'helmet': True, 'vest': True, 'gloves': False}
    
    # Step 1: Detect persons
    persons = detect_persons(image)
    
    # Step 2: Detect PPE
    ppe_items = detect_ppe(image)
    
    person_analyses = []
    
    for person in persons:
        person_box = person['bbox']
        person_height = person_box[3] - person_box[1]
        
        # Step 3: Extract pose landmarks
        landmarks = extract_pose_landmarks(image, person_box)
        
        if landmarks is None:
            # Skip person if pose estimation fails
            continue
        
        # Step 4: Match PPE items to this person
        person_ppe = []
        for ppe in ppe_items:
            if has_bbox_overlap(person_box, ppe['bbox']):
                person_ppe.append(ppe)
        
        # Step 5: Validate each PPE item
        validated_ppe = []
        for ppe in person_ppe:
            is_valid, reason = validate_ppe_location(
                ppe_box=ppe['bbox'],
                ppe_class=ppe['class_name'],
                landmarks=landmarks,
                person_height=person_height
            )
            
            validated_ppe.append({
                **ppe,
                'is_valid': is_valid,
                'location_reason': reason
            })
        
        # Determine what PPE this person has (and is properly worn)
        has_helmet = any(
            p['class_name'] == 'Hardhat' and p['is_valid']
            for p in validated_ppe
        )
        has_vest = any(
            p['class_name'] == 'Safety Vest' and p['is_valid']
            for p in validated_ppe
        )
        has_gloves = any(
            p['class_name'] == 'Gloves' and p['is_valid']
            for p in validated_ppe
        )
        
        # Step 6: Evaluate compliance
        person_status = {
            'has_helmet': has_helmet,
            'has_vest': has_vest,
            'has_gloves': has_gloves
        }
        
        is_compliant, reason = evaluate_compliance(person_status, requirements)
        
        person_analyses.append({
            'person_box': person_box,
            'landmarks': landmarks,
            'ppe_items': validated_ppe,
            'overall_compliant': is_compliant,
            'reason': reason,
            'status': 'COMPLIANT' if is_compliant else 'NON-COMPLIANT'
        })
    
    # Step 7: Calculate aggregate statistics
    total_people = len(person_analyses)
    compliant_people = sum(1 for p in person_analyses if p['overall_compliant'])
    non_compliant = total_people - compliant_people
    compliance_rate = compliant_people / total_people if total_people > 0 else 0
    
    return {
        'total_people': total_people,
        'compliant_people': compliant_people,
        'non_compliant_people': non_compliant,
        'compliance_rate': compliance_rate,
        'threshold_used': threshold,
        'person_analyses': person_analyses
    }
```

---

## Performance Optimization

### 1. Model Loading Strategy

```python
# Load models once at startup (singleton pattern)
class DetectorManager:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize_models()
        return cls._instance
    
    def _initialize_models(self):
        # Load person detector (lightweight)
        self.person_detector = YOLO('yolov8n.pt')
        
        # Load PPE detector (custom, heavier)
        if USE_HUGGINGFACE:
            model_path = hf_hub_download(
                repo_id=HUGGINGFACE_REPO,
                filename=HUGGINGFACE_FILENAME
            )
        else:
            model_path = LOCAL_MODEL_PATH
        
        self.ppe_detector = YOLO(model_path)
        
        # Initialize MediaPipe (lazy loading)
        self.pose = mp.solutions.pose.Pose(
            static_image_mode=True,
            model_complexity=1,  # Balance speed/accuracy
            enable_segmentation=False,  # Disable for speed
            min_detection_confidence=0.5
        )
```

### 2. Image Processing Optimization

```python
def optimize_image_for_inference(image, max_size=640):
    """
    Resize image to optimal size for YOLOv8
    
    - Maintains aspect ratio
    - Pads to square if needed
    - Reduces processing time for large images
    """
    h, w = image.shape[:2]
    
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        image = cv2.resize(image, (new_w, new_h))
    
    return image
```

### 3. Batch Processing for Video

```python
def process_video_frame(frame, previous_results=None):
    """
    Optimize video processing with temporal coherence
    
    - Skip pose estimation if person bbox similar to previous frame
    - Use tracking instead of detection for stable persons
    - Reduce detection frequency (every N frames)
    """
    # Detect persons
    persons = detect_persons(frame)
    
    # Match with previous frame (if available)
    if previous_results:
        persons = match_persons_across_frames(persons, previous_results)
    
    # Only run pose estimation on new/moved persons
    for person in persons:
        if person.get('moved', True):
            person['landmarks'] = extract_pose_landmarks(frame, person['bbox'])
        else:
            person['landmarks'] = person.get('previous_landmarks')
    
    return persons
```

---

## Technical Specifications

### Model Details

**Person Detector (YOLOv8n):**
- Parameters: 3.2M
- Model size: 6.2 MB
- Pretrained: COCO dataset
- Input: 640×640 (auto-resize)
- Output: Person bounding boxes

**PPE Detector (Custom YOLOv8s):**
- Parameters: 11.2M
- Model size: 22.5 MB
- Training: 6,000 construction images
- Classes: 6 (Hardhat, Vest, Mask, Cone, Machinery, Vehicle)
- mAP50: 0.62-0.64
- mAP50-95: 0.35-0.36

**Pose Estimator (MediaPipe):**
- Framework: TensorFlow Lite
- Model: BlazePose
- Landmarks: 33 points
- Inference: ~10ms per person

### Performance Benchmarks

**Hardware: CPU (Intel i7-12700K)**
- Single image (640×480): ~1.2s
- Live camera (30 FPS input): 1-2 FPS output
- Memory usage: ~2GB

**Hardware: GPU (NVIDIA RTX 3070)**
- Single image: ~150ms
- Live camera: 15-20 FPS output
- Memory usage: ~3GB VRAM

---

<p align="center">
  <strong>⚙️ Technical implementation designed for accuracy and real-world deployment ⚙️</strong>
</p>

<p align="center">
  <sub>Technical Documentation | Last Updated: November 29, 2025</sub>
</p>
