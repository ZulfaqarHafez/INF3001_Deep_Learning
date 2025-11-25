"""
Simplified Helmet Compliance Detection
Focus: Helmet on head = COMPLIANT, anywhere else = NON-COMPLIANT

Author: Zulfaqar
Project: INF3001 Deep Learning - PPE Detection (Simplified)
"""

import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO
from typing import Dict, List, Tuple, Optional


class HelmetComplianceDetector:
    """
    Simple helmet compliance detector.
    COMPLIANT: Helmet on head
    NON-COMPLIANT: Helmet anywhere else (in hand, on ground, nearby)
    """
    
    def __init__(self, ppe_model_path: str, confidence_threshold: float = 0.5):
        """
        Initialize the Helmet Compliance Detector.
        
        Args:
            ppe_model_path: Path to YOUR trained PPE model
            confidence_threshold: Minimum confidence for detections (0-1)
        """
        print("="*70)
        print("HELMET COMPLIANCE DETECTOR - SIMPLIFIED")
        print("="*70)
        
        # Load pretrained YOLOv8n for person detection
        print(f"\n[1/3] Loading YOLOv8n (pretrained) for person detection...")
        self.person_model = YOLO('yolov8n.pt')  # Auto-downloads if needed
        print(f"      ✓ Person detection ready!")
        
        # Load your custom PPE model for helmet detection
        print(f"\n[2/3] Loading YOUR PPE model from: {ppe_model_path}")
        self.ppe_model = YOLO(ppe_model_path)
        self.confidence_threshold = confidence_threshold
        print(f"      ✓ PPE model loaded!")
        
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
        self.HELMET_ON_HEAD_THRESHOLD = 120  # pixels
        
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
    
    def detect_helmets(self, frame: np.ndarray) -> List[Dict]:
        """Detect helmets using your custom model."""
        results = self.ppe_model(frame, conf=self.confidence_threshold, verbose=False)[0]
        
        helmets = []
        for box in results.boxes:
            cls = int(box.cls[0])
            class_name = self.ppe_model.names[cls].lower()
            
            # Only get helmets/hardhats (not "no-helmet")
            if 'hardhat' in class_name or 'helmet' in class_name:
                if 'no' not in class_name:  # Skip "no-hardhat"
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    helmets.append({
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'confidence': float(box.conf[0]),
                        'class_name': self.ppe_model.names[cls]
                    })
        
        return helmets
    
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
    
    def check_helmet_on_person(self, person: Dict, helmets: List[Dict], head_pos: Tuple[int, int]) -> Dict:
        """
        Check if a helmet is on this person's head.
        
        Returns:
            Dictionary with compliance status
        """
        person_center = self.get_box_center(person['bbox'])
        
        # Find closest helmet to this person
        closest_helmet = None
        min_dist_to_person = float('inf')
        
        for helmet in helmets:
            helmet_center = self.get_box_center(helmet['bbox'])
            dist = self.euclidean_distance(person_center, helmet_center)
            if dist < min_dist_to_person:
                min_dist_to_person = dist
                closest_helmet = helmet
        
        # No helmet found near this person
        if closest_helmet is None or min_dist_to_person > 400:
            return {
                'has_helmet': False,
                'compliant': False,
                'status': 'NO_HELMET',
                'reason': 'No helmet detected',
                'distance_to_head': None
            }
        
        # Check if helmet is on head
        helmet_center = self.get_box_center(closest_helmet['bbox'])
        dist_to_head = self.euclidean_distance(helmet_center, head_pos)
        
        if dist_to_head < self.HELMET_ON_HEAD_THRESHOLD:
            return {
                'has_helmet': True,
                'compliant': True,
                'status': 'WEARING',
                'reason': f'Helmet on head ({dist_to_head:.0f}px from head)',
                'distance_to_head': dist_to_head,
                'helmet': closest_helmet
            }
        else:
            return {
                'has_helmet': True,
                'compliant': False,
                'status': 'NOT_WEARING',
                'reason': f'Helmet detected but not on head ({dist_to_head:.0f}px away)',
                'distance_to_head': dist_to_head,
                'helmet': closest_helmet
            }
    
    def process_frame(self, frame: np.ndarray, visualize: bool = True) -> Tuple[np.ndarray, Dict]:
        """
        Process frame for helmet compliance.
        
        Returns:
            (annotated_frame, results_dict)
        """
        output_frame = frame.copy()
        
        # Detect people and helmets
        people = self.detect_people(frame)
        helmets = self.detect_helmets(frame)
        
        print(f"\n[Detection] Found {len(people)} people, {len(helmets)} helmets")
        
        # Analyze each person
        analyses = []
        
        for person in people:
            # Get head position
            head_pos = self.get_head_position(frame, person['bbox'])
            
            if head_pos is None:
                analyses.append({
                    'person_box': person['bbox'],
                    'head_detected': False,
                    'has_helmet': False,
                    'compliant': False,
                    'status': 'NO_POSE',
                    'reason': 'Could not detect head position'
                })
                continue
            
            # Check helmet compliance
            result = self.check_helmet_on_person(person, helmets, head_pos)
            result['person_box'] = person['bbox']
            result['head_detected'] = True
            result['head_position'] = head_pos
            
            analyses.append(result)
        
        # Visualize
        if visualize:
            output_frame = self.visualize(output_frame, analyses, helmets)
        
        # Compile results
        results = {
            'people': people,
            'helmets': helmets,
            'analyses': analyses,
            'total_people': len(people),
            'compliant': sum(1 for a in analyses if a['compliant']),
            'non_compliant': sum(1 for a in analyses if not a['compliant']),
            'compliance_rate': (sum(1 for a in analyses if a['compliant']) / len(analyses) 
                              if analyses else 0.0)
        }
        
        return output_frame, results
    
    def visualize(self, frame: np.ndarray, analyses: List[Dict], helmets: List[Dict]) -> np.ndarray:
        """Draw visualizations on frame."""
        output = frame.copy()
        
        # Draw all detected helmets (yellow boxes)
        for helmet in helmets:
            x1, y1, x2, y2 = helmet['bbox']
            cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(output, f"Helmet {helmet['confidence']:.2f}", 
                       (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        # Draw people with compliance status
        for analysis in analyses:
            x1, y1, x2, y2 = analysis['person_box']
            
            # Color based on compliance
            if analysis['compliant']:
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
            if analysis['head_detected']:
                head_pos = analysis['head_position']
                cv2.circle(output, head_pos, 8, (255, 255, 0), -1)
                cv2.circle(output, head_pos, 10, (255, 255, 255), 2)
            
            # Draw reason text
            y_offset = y2 + 25
            cv2.putText(output, analysis['reason'], (x1, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Draw summary
        total = len(analyses)
        compliant = sum(1 for a in analyses if a['compliant'])
        
        summary = f"People: {total} | Compliant: {compliant} | Non-compliant: {total - compliant}"
        cv2.rectangle(output, (10, 10), (600, 45), (0, 0, 0), -1)
        cv2.putText(output, summary, (15, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return output


# Simple test function
def test_helmet_detector(ppe_model_path: str, image_path: str):
    """Quick test"""
    detector = HelmetComplianceDetector(ppe_model_path, confidence_threshold=0.5)
    
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Error: Could not load {image_path}")
        return
    
    output_frame, results = detector.process_frame(frame, visualize=True)
    
    print("\n" + "="*70)
    print("HELMET COMPLIANCE RESULTS")
    print("="*70)
    print(f"Total People: {results['total_people']}")
    print(f"Compliant (helmet on head): {results['compliant']}")
    print(f"Non-Compliant: {results['non_compliant']}")
    print(f"Compliance Rate: {results['compliance_rate']*100:.1f}%")
    
    print("\nDetailed breakdown:")
    for i, analysis in enumerate(results['analyses'], 1):
        status_icon = "✓" if analysis['compliant'] else "✗"
        print(f"  Person {i}: {status_icon} {analysis['status']} - {analysis['reason']}")
    
    print("="*70)
    
    # Save
    cv2.imwrite('helmet_compliance_output.jpg', output_frame)
    print(f"\n✓ Output saved: helmet_compliance_output.jpg")
    
    # Display
    cv2.imshow('Helmet Compliance Detection', output_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python helmet_compliance_detector.py <ppe_model_path> <image_path>")
        print("Example: python helmet_compliance_detector.py ppe-4080-v12/weights/best.pt test.jpg")
    else:
        test_helmet_detector(sys.argv[1], sys.argv[2])
