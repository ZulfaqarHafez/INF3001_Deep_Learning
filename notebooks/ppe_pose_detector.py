"""
Hybrid PPE Pose Detection System
Combines pretrained YOLOv8 for person detection with your custom PPE model.

Author: Zulfaqar
Project: INF3001 Deep Learning - PPE Detection
Version: 3.0 - Hybrid dual-model approach
"""

import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO
from typing import Dict, List, Tuple, Optional
import json


class HybridPPEPoseDetector:
    """
    Hybrid detector using:
    1. YOLOv8n (pretrained) for person detection
    2. Your custom model for PPE detection
    3. MediaPipe Pose for compliance verification
    """
    
    def __init__(self, ppe_model_path: str, confidence_threshold: float = 0.5):
        """
        Initialize the Hybrid PPE Pose Detector.
        
        Args:
            ppe_model_path: Path to YOUR trained PPE model
            confidence_threshold: Minimum confidence for detections (0-1)
        """
        print("="*70)
        print("HYBRID PPE POSE DETECTOR - DUAL MODEL SYSTEM")
        print("="*70)
        
        # Load pretrained YOLOv8n for person detection
        print(f"\n[1/3] Loading YOLOv8n (pretrained) for person detection...")
        self.person_model = YOLO('yolov8n.pt')  # Will auto-download if not present
        print(f"      ✓ Person detection model ready!")
        
        # Load your custom PPE model
        print(f"\n[2/3] Loading YOUR PPE model from: {ppe_model_path}")
        self.ppe_model = YOLO(ppe_model_path)
        self.confidence_threshold = confidence_threshold
        print(f"      ✓ PPE detection model loaded!")
        
        # Get PPE model classes
        self.ppe_classes = self.ppe_model.names
        print(f"      PPE Model classes: {list(self.ppe_classes.values())}")
        
        # Initialize MediaPipe Pose
        print(f"\n[3/3] Initializing MediaPipe Pose...")
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        print(f"      ✓ MediaPipe Pose initialized!")
        
        # Distance thresholds (in pixels)
        self.HELMET_HEAD_THRESHOLD = 180
        self.HELMET_HAND_THRESHOLD = 200
        self.VEST_TORSO_THRESHOLD = 300
        
        print("\n" + "="*70)
        print("INITIALIZATION COMPLETE!")
        print("="*70 + "\n")
    
    def detect_people(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect people using pretrained YOLOv8n.
        
        Returns:
            List of person detections
        """
        results = self.person_model(frame, conf=self.confidence_threshold, verbose=False)[0]
        
        people = []
        for box in results.boxes:
            cls = int(box.cls[0])
            # Class 0 in COCO dataset is 'person'
            if cls == 0:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                
                people.append({
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'confidence': conf,
                    'class_id': cls,
                    'class_name': 'person'
                })
        
        return people
    
    def detect_ppe(self, frame: np.ndarray) -> Dict:
        """
        Detect PPE items using your custom model.
        
        Returns:
            Dictionary of PPE detections by category
        """
        results = self.ppe_model(frame, conf=self.confidence_threshold, verbose=False)[0]
        
        detections = {
            'hardhats': [],
            'vests': [],
            'no_hardhat': [],
            'no_vest': [],
            'other': []
        }
        
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            
            bbox = [int(x1), int(y1), int(x2), int(y2)]
            detection = {
                'bbox': bbox,
                'confidence': conf,
                'class_id': cls,
                'class_name': self.ppe_classes.get(cls, 'unknown')
            }
            
            # Categorize by class name
            class_name_lower = detection['class_name'].lower()
            
            if 'hardhat' in class_name_lower or 'helmet' in class_name_lower:
                if 'no' in class_name_lower:
                    detections['no_hardhat'].append(detection)
                else:
                    detections['hardhats'].append(detection)
            elif 'vest' in class_name_lower:
                if 'no' in class_name_lower:
                    detections['no_vest'].append(detection)
                else:
                    detections['vests'].append(detection)
            else:
                detections['other'].append(detection)
        
        return detections
    
    def euclidean_distance(self, point1: Tuple[int, int], point2: Tuple[int, int]) -> float:
        """Calculate Euclidean distance between two points."""
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def get_box_center(self, box: List[int]) -> Tuple[int, int]:
        """Get center point of bounding box."""
        x1, y1, x2, y2 = box
        return (int((x1 + x2) / 2), int((y1 + y2) / 2))
    
    def get_pose_landmarks(self, frame: np.ndarray, person_box: List[int]) -> Optional[Dict]:
        """
        Extract pose landmarks for a person.
        
        Args:
            frame: Full frame
            person_box: Person bounding box [x1, y1, x2, y2]
            
        Returns:
            Dictionary with pose landmarks or None
        """
        x1, y1, x2, y2 = person_box
        
        # Expand box for better pose detection
        h, w = frame.shape[:2]
        expand = 30
        x1 = max(0, x1 - expand)
        y1 = max(0, y1 - expand)
        x2 = min(w, x2 + expand)
        y2 = min(h, y2 + expand)
        
        # Crop person region
        person_img = frame[y1:y2, x1:x2]
        
        if person_img.size == 0:
            return None
        
        # Convert to RGB for MediaPipe
        rgb_img = cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB)
        
        # Process pose
        results = self.pose.process(rgb_img)
        
        if not results.pose_landmarks:
            return None
        
        landmarks = results.pose_landmarks.landmark
        
        # Convert to pixel coordinates in full frame
        person_w = x2 - x1
        person_h = y2 - y1
        
        pose_data = {
            'nose': (int(x1 + landmarks[0].x * person_w), 
                    int(y1 + landmarks[0].y * person_h)),
            'left_shoulder': (int(x1 + landmarks[11].x * person_w), 
                            int(y1 + landmarks[11].y * person_h)),
            'right_shoulder': (int(x1 + landmarks[12].x * person_w), 
                             int(y1 + landmarks[12].y * person_h)),
            'left_wrist': (int(x1 + landmarks[15].x * person_w), 
                          int(y1 + landmarks[15].y * person_h)),
            'right_wrist': (int(x1 + landmarks[16].x * person_w), 
                           int(y1 + landmarks[16].y * person_h)),
            'left_hip': (int(x1 + landmarks[23].x * person_w), 
                        int(y1 + landmarks[23].y * person_h)),
            'right_hip': (int(x1 + landmarks[24].x * person_w), 
                         int(y1 + landmarks[24].y * person_h)),
        }
        
        return pose_data
    
    def check_helmet_status(self, helmet_box: List[int], pose_data: Dict) -> Dict:
        """Determine if helmet is worn or held."""
        helmet_center = self.get_box_center(helmet_box)
        
        head_pos = pose_data['nose']
        left_hand = pose_data['left_wrist']
        right_hand = pose_data['right_wrist']
        
        dist_to_head = self.euclidean_distance(helmet_center, head_pos)
        dist_to_left_hand = self.euclidean_distance(helmet_center, left_hand)
        dist_to_right_hand = self.euclidean_distance(helmet_center, right_hand)
        
        if dist_to_head < self.HELMET_HEAD_THRESHOLD:
            status = 'WEARING'
            compliant = True
            reason = f'Helmet on head (distance: {dist_to_head:.1f}px)'
        elif dist_to_left_hand < self.HELMET_HAND_THRESHOLD:
            status = 'HOLDING_LEFT_HAND'
            compliant = False
            reason = f'Helmet in left hand (distance: {dist_to_left_hand:.1f}px)'
        elif dist_to_right_hand < self.HELMET_HAND_THRESHOLD:
            status = 'HOLDING_RIGHT_HAND'
            compliant = False
            reason = f'Helmet in right hand (distance: {dist_to_right_hand:.1f}px)'
        else:
            status = 'NEARBY'
            compliant = False
            reason = 'Helmet detected but not worn or held'
        
        return {
            'status': status,
            'compliant': compliant,
            'reason': reason,
            'distances': {
                'to_head': dist_to_head,
                'to_left_hand': dist_to_left_hand,
                'to_right_hand': dist_to_right_hand
            }
        }
    
    def check_vest_status(self, vest_box: List[int], pose_data: Dict) -> Dict:
        """Determine if vest is worn on torso."""
        vest_center = self.get_box_center(vest_box)
        
        left_shoulder = pose_data['left_shoulder']
        right_shoulder = pose_data['right_shoulder']
        left_hip = pose_data['left_hip']
        right_hip = pose_data['right_hip']
        
        torso_center = (
            int((left_shoulder[0] + right_shoulder[0] + left_hip[0] + right_hip[0]) / 4),
            int((left_shoulder[1] + right_shoulder[1] + left_hip[1] + right_hip[1]) / 4)
        )
        
        dist_to_torso = self.euclidean_distance(vest_center, torso_center)
        
        vest_y_center = vest_center[1]
        torso_y_min = min(left_shoulder[1], right_shoulder[1])
        torso_y_max = max(left_hip[1], right_hip[1])
        
        is_in_torso_region = torso_y_min <= vest_y_center <= torso_y_max
        
        if dist_to_torso < self.VEST_TORSO_THRESHOLD and is_in_torso_region:
            status = 'WEARING'
            compliant = True
            reason = f'Vest on torso (distance: {dist_to_torso:.1f}px)'
        else:
            status = 'NOT_WEARING'
            compliant = False
            reason = f'Vest not on torso (distance: {dist_to_torso:.1f}px)'
        
        return {
            'status': status,
            'compliant': compliant,
            'reason': reason,
            'distance_to_torso': dist_to_torso
        }
    
    def analyze_person(self, frame: np.ndarray, person: Dict, 
                      ppe_detections: Dict) -> Dict:
        """
        Analyze a person for PPE compliance.
        
        Args:
            frame: Input frame
            person: Person detection dict
            ppe_detections: PPE detections dict
            
        Returns:
            Analysis results
        """
        person_box = person['bbox']
        person_center = self.get_box_center(person_box)
        
        # Get pose
        pose_data = self.get_pose_landmarks(frame, person_box)
        
        if pose_data is None:
            return {
                'person_box': person_box,
                'pose_detected': False,
                'helmet_status': {'status': 'NO_POSE', 'compliant': False, 'reason': 'Could not detect pose'},
                'vest_status': {'status': 'NO_POSE', 'compliant': False, 'reason': 'Could not detect pose'},
                'overall_compliant': False,
                'warnings': ['Could not detect pose']
            }
        
        # Find closest helmet
        closest_helmet = None
        min_helmet_dist = float('inf')
        
        for helmet in ppe_detections['hardhats']:
            helmet_center = self.get_box_center(helmet['bbox'])
            dist = self.euclidean_distance(person_center, helmet_center)
            if dist < min_helmet_dist:
                min_helmet_dist = dist
                closest_helmet = helmet
        
        # Find closest vest
        closest_vest = None
        min_vest_dist = float('inf')
        
        for vest in ppe_detections['vests']:
            vest_center = self.get_box_center(vest['bbox'])
            dist = self.euclidean_distance(person_center, vest_center)
            if dist < min_vest_dist:
                min_vest_dist = dist
                closest_vest = vest
        
        # Analyze helmet
        if closest_helmet and min_helmet_dist < 400:
            helmet_status = self.check_helmet_status(closest_helmet['bbox'], pose_data)
        else:
            helmet_status = {
                'status': 'NOT_DETECTED',
                'compliant': False,
                'reason': 'No helmet detected near person'
            }
        
        # Analyze vest
        if closest_vest and min_vest_dist < 400:
            vest_status = self.check_vest_status(closest_vest['bbox'], pose_data)
        else:
            vest_status = {
                'status': 'NOT_DETECTED',
                'compliant': False,
                'reason': 'No vest detected near person'
            }
        
        # Overall compliance
        overall_compliant = (
            helmet_status.get('compliant', False) and 
            vest_status.get('compliant', False)
        )
        
        # Warnings
        warnings = []
        if not helmet_status.get('compliant', False):
            warnings.append(f"Helmet: {helmet_status.get('reason', 'Not compliant')}")
        if not vest_status.get('compliant', False):
            warnings.append(f"Vest: {vest_status.get('reason', 'Not compliant')}")
        
        return {
            'person_box': person_box,
            'pose_detected': True,
            'pose_data': pose_data,
            'helmet_status': helmet_status,
            'vest_status': vest_status,
            'overall_compliant': overall_compliant,
            'warnings': warnings
        }
    
    def process_frame(self, frame: np.ndarray, visualize: bool = True) -> Tuple[np.ndarray, Dict]:
        """
        Process frame using dual-model approach.
        
        Args:
            frame: Input frame (BGR)
            visualize: Whether to draw visualizations
            
        Returns:
            (annotated_frame, results_dict)
        """
        output_frame = frame.copy()
        
        # Step 1: Detect people (using pretrained YOLOv8n)
        people = self.detect_people(frame)
        print(f"[Detection] Found {len(people)} people")
        
        # Step 2: Detect PPE (using your custom model)
        ppe_detections = self.detect_ppe(frame)
        print(f"[Detection] Found {len(ppe_detections['hardhats'])} helmets, "
              f"{len(ppe_detections['vests'])} vests")
        
        # Step 3: Analyze each person
        person_analyses = []
        for person in people:
            analysis = self.analyze_person(frame, person, ppe_detections)
            person_analyses.append(analysis)
        
        # Step 4: Visualize
        if visualize:
            output_frame = self.visualize_results(output_frame, person_analyses, ppe_detections)
        
        # Compile results
        results = {
            'people': people,
            'ppe_detections': ppe_detections,
            'person_analyses': person_analyses,
            'total_people': len(people),
            'compliant_people': sum(1 for p in person_analyses if p['overall_compliant']),
            'non_compliant_people': sum(1 for p in person_analyses if not p['overall_compliant']),
            'compliance_rate': (sum(1 for p in person_analyses if p['overall_compliant']) / 
                              len(person_analyses) if person_analyses else 0.0)
        }
        
        return output_frame, results
    
    def visualize_results(self, frame: np.ndarray, person_analyses: List[Dict], 
                         ppe_detections: Dict) -> np.ndarray:
        """Draw visualizations on frame."""
        output = frame.copy()
        
        # Draw PPE detections
        for helmet in ppe_detections['hardhats']:
            x1, y1, x2, y2 = helmet['bbox']
            cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 255), 2)
            label = f"Helmet {helmet['confidence']:.2f}"
            cv2.putText(output, label, (x1, y1-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        for vest in ppe_detections['vests']:
            x1, y1, x2, y2 = vest['bbox']
            cv2.rectangle(output, (x1, y1), (x2, y2), (255, 0, 255), 2)
            label = f"Vest {vest['confidence']:.2f}"
            cv2.putText(output, label, (x1, y1-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
        
        # Draw person analyses
        for analysis in person_analyses:
            x1, y1, x2, y2 = analysis['person_box']
            
            if analysis['overall_compliant']:
                color = (0, 255, 0)
                status = 'COMPLIANT'
            else:
                color = (0, 0, 255)
                status = 'NON-COMPLIANT'
            
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 3)
            cv2.putText(output, status, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Draw pose keypoints
            if analysis['pose_detected'] and analysis.get('pose_data'):
                pose_data = analysis['pose_data']
                
                for kp in ['nose', 'left_wrist', 'right_wrist', 'left_shoulder', 'right_shoulder']:
                    if kp in pose_data:
                        pt = pose_data[kp]
                        cv2.circle(output, pt, 4, (255, 255, 0), -1)
            
            # Draw warnings
            y_offset = y2 + 20
            for warning in analysis['warnings']:
                cv2.putText(output, warning, (x1, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                y_offset += 15
        
        # Summary
        total = len(person_analyses)
        compliant = sum(1 for p in person_analyses if p['overall_compliant'])
        
        summary = f"People: {total} | Compliant: {compliant} | Non-compliant: {total - compliant}"
        cv2.rectangle(output, (10, 10), (580, 40), (0, 0, 0), -1)
        cv2.putText(output, summary, (15, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return output


def test_hybrid_detector(ppe_model_path: str, image_path: str):
    """Quick test function"""
    detector = HybridPPEPoseDetector(ppe_model_path, confidence_threshold=0.5)
    
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Error: Could not load image from {image_path}")
        return
    
    output_frame, results = detector.process_frame(frame, visualize=True)
    
    print("\n" + "="*70)
    print("DETECTION RESULTS")
    print("="*70)
    print(f"Total People: {results['total_people']}")
    print(f"Compliant: {results['compliant_people']}")
    print(f"Non-Compliant: {results['non_compliant_people']}")
    print(f"Compliance Rate: {results['compliance_rate']*100:.1f}%")
    print("="*70)
    
    # Save output
    cv2.imwrite('hybrid_output.jpg', output_frame)
    print(f"\n✓ Output saved to: hybrid_output.jpg")
    
    # Display
    cv2.imshow('Hybrid PPE Detection', output_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python hybrid_ppe_detector.py <ppe_model_path> <image_path>")
        print("Example: python hybrid_ppe_detector.py ppe-4080-v12/weights/best.pt test.jpg")
    else:
        test_hybrid_detector(sys.argv[1], sys.argv[2])