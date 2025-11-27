"""
Simplified Helmet Compliance Detection
Focus: Helmet on head = COMPLIANT, anywhere else = NON-COMPLIANT
Update: Detects ALL PPE items for visualization, but enforces Helmet for compliance.

Author: Zulfaqar
Project: INF3001 Deep Learning - PPE Detection (Simplified)
"""

import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO
from typing import Dict, List, Tuple, Optional
import os


def download_model_from_huggingface(
    repo_id: str = "iMaximusiV/yolo-ppe-detector",
    filename: str = "yolov8s.pt",
    cache_dir: str = None
) -> str:
    """
    Download YOLO model from Hugging Face Hub.
    
    Args:
        repo_id: Hugging Face repository ID (e.g., "iMaximusiV/yolo-ppe-detector")
        filename: Model filename in the repository (e.g., "yolov8s.pt")
        cache_dir: Optional directory to cache the model
        
    Returns:
        Local path to the downloaded model file
    """
    try:
        from huggingface_hub import hf_hub_download
        
        print(f"\n[HuggingFace] Downloading model from: {repo_id}/{filename}")
        
        model_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            cache_dir=cache_dir
        )
        
        print(f"[HuggingFace] ✓ Model downloaded to: {model_path}")
        return model_path
        
    except ImportError:
        print("[HuggingFace] ✗ huggingface_hub not installed!")
        print("[HuggingFace] Install it with: pip install huggingface_hub")
        raise ImportError("Please install huggingface_hub: pip install huggingface_hub")
    except Exception as e:
        print(f"[HuggingFace] ✗ Error downloading model: {e}")
        raise


class HelmetComplianceDetector:
    """
    Helmet compliance detector with full PPE auditing.
    COMPLIANT: Helmet on head (Strict)
    AUDIT: Lists all other gear found (Vest, Gloves, etc.)
    
    Supports loading PPE model from:
    - Local file path
    - Hugging Face Hub (automatic download)
    """
    
    def __init__(
        self, 
        ppe_model_path: str = None,
        huggingface_repo: str = None,
        huggingface_filename: str = "yolov8s.pt",
        confidence_threshold: float = 0.5,
        use_huggingface: bool = False
    ):
        """
        Initialize the Helmet Compliance Detector.
        """
        print("="*70)
        print("HELMET COMPLIANCE DETECTOR - MULTI-CLASS AUDIT")
        print("="*70)
        
        # Load pretrained YOLOv8n for person detection
        print(f"\n[1/3] Loading YOLOv8n (pretrained) for person detection...")
        self.person_model = YOLO('yolov8n.pt')  # Auto-downloads if needed
        print(f"      ✓ Person detection ready!")
        
        # Load PPE model - either from local path or Hugging Face
        print(f"\n[2/3] Loading PPE model for helmet detection...")
        
        if use_huggingface or huggingface_repo:
            # Download from Hugging Face
            repo = huggingface_repo or "iMaximusiV/yolo-ppe-detector"
            ppe_model_path = download_model_from_huggingface(
                repo_id=repo,
                filename=huggingface_filename
            )
        elif ppe_model_path is None:
            raise ValueError(
                "Must provide either 'ppe_model_path' for local model or "
                "'huggingface_repo' / 'use_huggingface=True' for HuggingFace model"
            )
        
        print(f"      Loading from: {ppe_model_path}")
        self.ppe_model = YOLO(ppe_model_path)
        self.confidence_threshold = confidence_threshold
        print(f"      ✓ PPE model loaded!")
        
        # Print model classes
        print(f"      Model classes: {self.ppe_model.names}")
        
        # Initialize MediaPipe Pose for body keypoints
        print(f"\n[3/3] Initializing MediaPipe Pose...")
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5
        )
        print(f"      ✓ Pose estimation ready!")
        
        # Simple threshold: helmet must be within this distance of head
        self.HELMET_ON_HEAD_THRESHOLD = 400  # pixels
        
        print("\n" + "="*70)
        print("READY TO DETECT!")
        print("="*70)
        print(f"Helmet threshold: {self.HELMET_ON_HEAD_THRESHOLD}px from head")
        print("="*70 + "\n")
    
    def euclidean_distance(self, point1: Tuple[int, int], point2: Tuple[int, int]) -> float:
        """Calculate distance between two points."""
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def get_box_center(self, box: List[int]) -> Tuple[int, int]:
        """Get center of bounding box."""
        x1, y1, x2, y2 = box
        return (int((x1 + x2) / 2), int((y1 + y2) / 2))
    
    def detect_people(self, frame: np.ndarray) -> List[Dict]:
        """Detect people using YOLOv8n."""
        results = self.person_model(frame, conf=self.confidence_threshold, verbose=False)[0]
        
        people = []
        for box in results.boxes:
            cls = int(box.cls[0])
            if cls == 0:  # Person class in COCO
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                people.append({
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'confidence': float(box.conf[0])
                })
        
        return people
    
    def detect_all_ppe(self, frame: np.ndarray) -> List[Dict]:
        """Detect ALL PPE items (Helmets, Vests, Gloves, etc)."""
        results = self.ppe_model(frame, conf=self.confidence_threshold, verbose=False)[0]
        
        ppe_items = []
        for box in results.boxes:
            cls = int(box.cls[0])
            class_name = self.ppe_model.names[cls]
            
            # Skip 'person' if the PPE model detects it (we use YOLOv8n for people)
            if class_name.lower() == 'person':
                continue
                
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            ppe_items.append({
                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                'confidence': float(box.conf[0]),
                'class_name': class_name,  # e.g., 'Hardhat', 'Safety Vest', 'Gloves'
                'label': class_name
            })
        
        return ppe_items
    
    def get_head_position(self, frame: np.ndarray, person_box: List[int]) -> Optional[Tuple[int, int]]:
        """
        Get head position (nose keypoint) for a person.
        
        Returns:
            (x, y) coordinates of nose/head, or None if not detected
        """
        x1, y1, x2, y2 = person_box
        
        # Expand box slightly
        h, w = frame.shape[:2]
        expand = 30
        x1 = max(0, x1 - expand)
        y1 = max(0, y1 - expand)
        x2 = min(w, x2 + expand)
        y2 = min(h, y2 + expand)
        
        # Crop person
        person_img = frame[y1:y2, x1:x2]
        if person_img.size == 0:
            return None
        
        # Get pose
        rgb_img = cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_img)
        
        if not results.pose_landmarks:
            return None
        
        # Get nose position (head)
        nose = results.pose_landmarks.landmark[0]
        person_w = x2 - x1
        person_h = y2 - y1
        
        head_x = int(x1 + nose.x * person_w)
        head_y = int(y1 + nose.y * person_h)
        
        return (head_x, head_y)
    
    def analyze_person_ppe(self, person: Dict, all_ppe_items: List[Dict], head_pos: Tuple[int, int]) -> Dict:
        """
        Associate PPE with the person. 
        - Helmet triggers COMPLIANCE check.
        - Other items are just listed as 'detected_gear'.
        
        Returns:
            Dictionary with compliance status and list of all gear found.
        """
        px1, py1, px2, py2 = person['bbox']
        
        # Initialize result structure
        result = {
            'has_helmet': False,
            'compliant': False,  # Default to False until proven otherwise
            'status': 'NO_HELMET',
            'reason': 'No helmet detected',
            'distance_to_head': None,
            'detected_gear': [],  # List of strings e.g. ["Vest", "Gloves"]
            'ppe_items': []       # List of full object dicts for visualization
        }

        # 1. Associate items with this person
        # Item center must be roughly inside the person's bounding box (with margin)
        margin = 50
        
        for item in all_ppe_items:
            item_center = self.get_box_center(item['bbox'])
            ix, iy = item_center
            
            # Check if item is near person
            is_near = (px1 - margin < ix < px2 + margin) and \
                      (py1 - margin < iy < py2 + margin)
            
            if is_near:
                # Add to lists
                if item['class_name'] not in result['detected_gear']:
                    result['detected_gear'].append(item['class_name'])
                result['ppe_items'].append(item)

                # 2. Specific Logic for Helmet Compliance
                # Check if this item is a helmet
                class_lower = item['class_name'].lower()
                is_helmet = ('hardhat' in class_lower or 'helmet' in class_lower) and 'no' not in class_lower

                if is_helmet:
                    dist_to_head = self.euclidean_distance(item_center, head_pos)
                    
                    # If on head -> COMPLIANT
                    if dist_to_head < self.HELMET_ON_HEAD_THRESHOLD:
                        result['has_helmet'] = True
                        result['compliant'] = True
                        result['status'] = 'WEARING'
                        result['reason'] = f"Helmet on head ({dist_to_head:.0f}px)"
                        result['distance_to_head'] = dist_to_head
                        result['helmet'] = item # Store best helmet
                    
                    # If not on head, but found -> store it if we haven't found a better one yet
                    elif not result['has_helmet']: 
                        result['status'] = 'NOT_WEARING'
                        result['reason'] = f"Helmet detected off-head ({dist_to_head:.0f}px)"
                        result['distance_to_head'] = dist_to_head
                        result['helmet'] = item

        return result
    
    def process_frame(self, frame: np.ndarray, visualize: bool = True) -> Tuple[np.ndarray, Dict]:
        """
        Process frame for helmet compliance + full PPE audit.
        
        Returns:
            (annotated_frame, results_dict)
        """
        output_frame = frame.copy()
        
        # Detect everything
        people = self.detect_people(frame)
        all_ppe = self.detect_all_ppe(frame)
        
        print(f"\n[Detection] Found {len(people)} people, {len(all_ppe)} PPE items")
        
        # Analyze each person
        analyses = []
        
        for person in people:
            # Get head position
            head_pos = self.get_head_position(frame, person['bbox'])
            
            if head_pos is None:
                analyses.append({
                    'person_box': person['bbox'],
                    'head_detected': False,
                    'compliant': False,
                    'status': 'NO_POSE',
                    'reason': 'Could not detect head position',
                    'detected_gear': [],
                    'ppe_items': []
                })
                continue
            
            # Analyze gear and compliance
            result = self.analyze_person_ppe(person, all_ppe, head_pos)
            result['person_box'] = person['bbox']
            result['head_detected'] = True
            result['head_position'] = head_pos
            
            analyses.append(result)
        
        # Visualize locally (optional, mainly for debugging as API handles frontend viz)
        if visualize:
            output_frame = self.visualize(output_frame, analyses, all_ppe)
        
        # Compile results
        results = {
            'people': people,
            'all_ppe_detected': all_ppe, # Send all raw items
            'analyses': analyses,
            'total_people': len(people),
            'compliant': sum(1 for a in analyses if a.get('compliant', False)),
            'non_compliant': sum(1 for a in analyses if not a.get('compliant', False)),
            'compliance_rate': (sum(1 for a in analyses if a.get('compliant', False)) / len(analyses) 
                              if analyses else 0.0)
        }
        
        return output_frame, results
    
    def visualize(self, frame: np.ndarray, analyses: List[Dict], all_ppe: List[Dict]) -> np.ndarray:
        """Draw visualizations on frame for local debugging/output."""
        output = frame.copy()
        
        # Draw all PPE items first (thin lines)
        for item in all_ppe:
            x1, y1, x2, y2 = item['bbox']
            cv2.rectangle(output, (x1, y1), (x2, y2), (255, 200, 0), 1)
            cv2.putText(output, item['class_name'], (x1, y1-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)
        
        # Draw people and compliance status
        for analysis in analyses:
            x1, y1, x2, y2 = analysis['person_box']
            
            # Color based on compliance
            if analysis.get('compliant', False):
                color = (0, 255, 0)  # Green
                label = 'COMPLIANT'
            else:
                color = (0, 0, 255)  # Red
                label = 'NON-COMPLIANT'
            
            # Draw person box
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 3)
            
            # Draw status
            cv2.putText(output, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Draw head position (cyan dot)
            if analysis.get('head_detected'):
                head_pos = analysis['head_position']
                cv2.circle(output, head_pos, 8, (255, 255, 0), -1)
            
            # Draw reason text
            y_offset = y2 + 25
            cv2.putText(output, analysis.get('reason', ''), (x1, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return output


# Simple test function
def test_helmet_detector(ppe_model_path: str = None, image_path: str = None, use_huggingface: bool = False):
    """Quick test"""
    
    if use_huggingface:
        detector = HelmetComplianceDetector(
            huggingface_repo="iMaximusiV/yolo-ppe-detector",
            use_huggingface=True,
            confidence_threshold=0.50
        )
    else:
        detector = HelmetComplianceDetector(
            ppe_model_path=ppe_model_path,
            confidence_threshold=0.50
        )
    
    if image_path:
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Error: Could not load {image_path}")
            return
        
        output_frame, results = detector.process_frame(frame, visualize=True)
        
        print("\n" + "="*70)
        print("HELMET COMPLIANCE RESULTS (AUDIT MODE)")
        print("="*70)
        print(f"Total People: {results['total_people']}")
        print(f"Compliant: {results['compliant']}")
        print(f"Non-Compliant: {results['non_compliant']}")
        
        print("\nDetailed breakdown:")
        for i, analysis in enumerate(results['analyses'], 1):
            status_icon = "✓" if analysis.get('compliant', False) else "✗"
            gear_str = ", ".join(analysis.get('detected_gear', []))
            print(f"  Person {i}: {status_icon} {analysis.get('status')} - {analysis.get('reason')}")
            print(f"            Gear found: {gear_str if gear_str else 'None'}")
        
        print("="*70)
        
        # Save
        cv2.imwrite('helmet_compliance_audit.jpg', output_frame)
        print(f"\n✓ Output saved: helmet_compliance_audit.jpg")
    else:
        print("Model loaded successfully! Ready for detection.")


if __name__ == "__main__":
    import sys
    
    print("\nUsage options:")
    print("  1. Local model:     python helmet_compliance_detector.py <model_path> <image_path>")
    print("  2. HuggingFace:     python helmet_compliance_detector.py --huggingface <image_path>")
    
    if len(sys.argv) >= 2:
        if sys.argv[1] == '--huggingface':
            image_path = sys.argv[2] if len(sys.argv) > 2 else None
            test_helmet_detector(use_huggingface=True, image_path=image_path)
        elif len(sys.argv) >= 3:
            test_helmet_detector(ppe_model_path=sys.argv[1], image_path=sys.argv[2])
        else:
            print("Please provide arguments.")
    else:
        print("Example: python helmet_compliance_detector.py --huggingface test.jpg")