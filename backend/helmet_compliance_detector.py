"""
Simplified Helmet Compliance Detection with Pose Estimation
Focus: Dynamic Compliance (Gear must be worn at appropriate locations)
- Helmet -> Head (Nose)
- Vest -> Torso (Midpoint of Shoulders/Hips)
- Gloves -> Hands (Wrists)

Author: Zulfaqar
Project: INF3001 Deep Learning - PPE Detection (Advanced)
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
    """Download YOLO model from Hugging Face Hub."""
    try:
        from huggingface_hub import hf_hub_download
        print(f"\n[HuggingFace] Downloading model from: {repo_id}/{filename}")
        model_path = hf_hub_download(repo_id=repo_id, filename=filename, cache_dir=cache_dir)
        print(f"[HuggingFace] ✓ Model downloaded to: {model_path}")
        return model_path
    except ImportError:
        raise ImportError("Please install huggingface_hub: pip install huggingface_hub")
    except Exception as e:
        print(f"[HuggingFace] ✗ Error downloading model: {e}")
        raise


class HelmetComplianceDetector:
    """
    Helmet compliance detector with Pose Estimation.
    Verifies that detected gear is actually worn on the correct body part.
    """
    
    def __init__(
        self, 
        ppe_model_path: str = None,
        huggingface_repo: str = None,
        huggingface_filename: str = "yolov8s.pt",
        confidence_threshold: float = 0.5,
        use_huggingface: bool = False
    ):
        print("="*70)
        print("HELMET COMPLIANCE DETECTOR - POSE AWARE")
        print("="*70)
        
        # 1. Load YOLOv8n for Person Detection
        print(f"\n[1/3] Loading YOLOv8n (pretrained) for person detection...")
        self.person_model = YOLO('yolov8n.pt')
        print(f"      ✓ Person detection ready!")
        
        # 2. Load PPE Model
        print(f"\n[2/3] Loading PPE model...")
        if use_huggingface or huggingface_repo:
            repo = huggingface_repo or "iMaximusiV/yolo-ppe-detector"
            ppe_model_path = download_model_from_huggingface(repo_id=repo, filename=huggingface_filename)
        elif ppe_model_path is None:
            raise ValueError("Must provide model path or use Hugging Face")
        
        self.ppe_model = YOLO(ppe_model_path)
        self.confidence_threshold = confidence_threshold
        print(f"      ✓ PPE model loaded! Classes: {self.ppe_model.names}")
        
        # 3. Initialize MediaPipe Pose
        print(f"\n[3/3] Initializing MediaPipe Pose...")
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5
        )
        print(f"      ✓ Pose estimation ready!")
        
        # Thresholds (Dynamic scaling will be applied based on person size)
        self.HELMET_ON_HEAD_THRESHOLD = 400 
        
        print("\n" + "="*70)
        print("READY TO DETECT!")
        print("="*70)
    
    def euclidean_distance(self, point1: Tuple[int, int], point2: Tuple[int, int]) -> float:
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def get_box_center(self, box: List[int]) -> Tuple[int, int]:
        x1, y1, x2, y2 = box
        return (int((x1 + x2) / 2), int((y1 + y2) / 2))
    
    def detect_people(self, frame: np.ndarray) -> List[Dict]:
        results = self.person_model(frame, conf=self.confidence_threshold, verbose=False)[0]
        people = []
        for box in results.boxes:
            cls = int(box.cls[0])
            if cls == 0:  # Person class
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                people.append({
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'confidence': float(box.conf[0])
                })
        return people
    
    def detect_all_ppe(self, frame: np.ndarray) -> List[Dict]:
        results = self.ppe_model(frame, conf=self.confidence_threshold, verbose=False)[0]
        ppe_items = []
        for box in results.boxes:
            cls = int(box.cls[0])
            class_name = self.ppe_model.names[cls]
            if class_name.lower() == 'person': continue
            
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            ppe_items.append({
                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                'confidence': float(box.conf[0]),
                'class_name': class_name,
                'label': class_name
            })
        return ppe_items
    
    def get_body_landmarks(self, frame: np.ndarray, person_box: List[int]) -> Optional[Dict]:
        """
        Extracts key body landmarks (Nose, Torso Center, Wrists, etc.) for pose estimation.
        """
        x1, y1, x2, y2 = person_box
        h, w = frame.shape[:2]
        
        # Add slight margin to crop
        expand = 20
        px1, py1 = max(0, x1 - expand), max(0, y1 - expand)
        px2, py2 = min(w, x2 + expand), min(h, y2 + expand)
        
        person_img = frame[py1:py2, px1:px2]
        if person_img.size == 0: return None
        
        # MediaPipe processing
        rgb_img = cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_img)
        
        if not results.pose_landmarks: return None
        
        landmarks = results.pose_landmarks.landmark
        p_w, p_h = px2 - px1, py2 - py1
        
        # Helper to convert local MP coords to global image coords
        def to_global(lm):
            return (int(px1 + lm.x * p_w), int(py1 + lm.y * p_h))
        
        # Extract specific points
        nose = to_global(landmarks[0])
        l_shoulder = to_global(landmarks[11])
        r_shoulder = to_global(landmarks[12])
        l_elbow = to_global(landmarks[13])
        r_elbow = to_global(landmarks[14])
        l_wrist = to_global(landmarks[15])
        r_wrist = to_global(landmarks[16])
        l_hip = to_global(landmarks[23])
        r_hip = to_global(landmarks[24])
        
        # Calculate Torso Center (Midpoint of shoulders and hips)
        torso_x = int((l_shoulder[0] + r_shoulder[0] + l_hip[0] + r_hip[0]) / 4)
        torso_y = int((l_shoulder[1] + r_shoulder[1] + l_hip[1] + r_hip[1]) / 4)
        
        return {
            'nose': nose,
            'torso_center': (torso_x, torso_y),
            'shoulders': (l_shoulder, r_shoulder),
            'elbows': (l_elbow, r_elbow),
            'wrists': (l_wrist, r_wrist),
            'hips': (l_hip, r_hip)
        }

    def is_location_valid(self, item_box: List[int], item_type: str, landmarks: Dict, person_height: float) -> Tuple[bool, str]:
        """
        Validates if the PPE item is in the correct anatomical location using relative body size.
        """
        item_center = self.get_box_center(item_box)
        
        # Helmet -> Nose check
        if 'helmet' in item_type or 'hardhat' in item_type:
            threshold = person_height * 0.15 # ~15% of height
            dist = self.euclidean_distance(item_center, landmarks['nose'])
            return (True, "On Head") if dist < threshold else (False, "Off Head")
            
        # Vest -> Torso Center check
        elif 'vest' in item_type:
            threshold = person_height * 0.30 # Torso is larger
            dist = self.euclidean_distance(item_center, landmarks['torso_center'])
            return (True, "On Body") if dist < threshold else (False, "Off Body")
            
        # Glove -> Wrists check (nearest)
        elif 'glove' in item_type:
            threshold = person_height * 0.20
            dists = [self.euclidean_distance(item_center, w) for w in landmarks['wrists']]
            min_dist = min(dists)
            return (True, "On Hand") if min_dist < threshold else (False, "Off Hand")
            
        return True, "Detected"

    def analyze_person_ppe(self, person: Dict, all_ppe_items: List[Dict], landmarks: Dict) -> Dict:
        """
        Analyze all PPE gear associated with this person and verify compliance locations.
        """
        px1, py1, px2, py2 = person['bbox']
        person_height = py2 - py1
        
        # Track status of required items
        gear_status = {
            'helmet': {'detected': False, 'valid': False},
            'vest': {'detected': False, 'valid': False},
            'gloves': {'detected': False, 'valid': False}
        }
        
        detected_gear_list = []
        ppe_items_with_status = []
        
        margin = 50 
        
        for item in all_ppe_items:
            item_center = self.get_box_center(item['bbox'])
            ix, iy = item_center
            
            # 1. Spatial Association
            is_near = (px1 - margin < ix < px2 + margin) and \
                      (py1 - margin < iy < py2 + margin)
            
            if is_near:
                class_lower = item['class_name'].lower()
                
                # 2. Location Validation
                is_valid, loc_reason = self.is_location_valid(item['bbox'], class_lower, landmarks, person_height)
                
                # Update tracking
                if 'helmet' in class_lower or 'hardhat' in class_lower:
                    gear_status['helmet']['detected'] = True
                    if is_valid: gear_status['helmet']['valid'] = True
                elif 'vest' in class_lower:
                    gear_status['vest']['detected'] = True
                    if is_valid: gear_status['vest']['valid'] = True
                elif 'glove' in class_lower:
                    gear_status['gloves']['detected'] = True
                    if is_valid: gear_status['gloves']['valid'] = True
                
                # Prepare Output
                conf_pct = int(item['confidence'] * 100)
                status_tag = " [OK]" if is_valid else " [BAD POS]"
                gear_label = f"{item['class_name']} {conf_pct}%{status_tag}"
                
                if gear_label not in detected_gear_list:
                    detected_gear_list.append(gear_label)
                
                item_copy = item.copy()
                item_copy['is_valid_location'] = is_valid
                item_copy['location_reason'] = loc_reason
                ppe_items_with_status.append(item_copy)

        # 3. Compliance Logic (Helmet Mandatory + Valid Vest if detected)
        reasons = []
        compliant = True
        status_label = "COMPLIANT"
        
        # Helmet is mandatory
        if not gear_status['helmet']['detected']:
            compliant = False
            reasons.append("Missing Helmet")
            status_label = "NO HELMET"
        elif not gear_status['helmet']['valid']:
            compliant = False
            reasons.append("Helmet not on head")
            status_label = "NOT WORN"
        else:
            reasons.append("Helmet OK")
            
        # Vest contributes to compliance if detected
        if gear_status['vest']['detected']:
            if gear_status['vest']['valid']:
                reasons.append("Vest OK")
            else:
                compliant = False
                reasons.append("Vest not on body")
                status_label = "VEST OFF"
                
        # Gloves check (optional enforcement, but reporting status)
        if gear_status['gloves']['detected']:
            if gear_status['gloves']['valid']:
                reasons.append("Gloves OK")
            else:
                # Uncomment next line to make gloves mandatory for compliance if detected
                # compliant = False 
                reasons.append("Gloves not on hands")

        if not compliant and status_label == "COMPLIANT":
            status_label = "NON-COMPLIANT"

        return {
            'has_helmet': gear_status['helmet']['valid'],
            'compliant': compliant,
            'status': status_label,
            'reason': ", ".join(reasons),
            'detected_gear': detected_gear_list,
            'ppe_items': ppe_items_with_status
        }
    
    def process_frame(self, frame: np.ndarray, visualize: bool = True) -> Tuple[np.ndarray, Dict]:
        output_frame = frame.copy()
        people = self.detect_people(frame)
        all_ppe = self.detect_all_ppe(frame)
        
        print(f"\n[Detection] Found {len(people)} people, {len(all_ppe)} PPE items")
        
        analyses = []
        for person in people:
            landmarks = self.get_body_landmarks(frame, person['bbox'])
            
            if landmarks is None:
                analyses.append({
                    'person_box': person['bbox'],
                    'head_detected': False,
                    'compliant': False,
                    'status': 'NO_POSE',
                    'reason': 'Pose estimation failed',
                    'detected_gear': [],
                    'ppe_items': []
                })
                continue
            
            result = self.analyze_person_ppe(person, all_ppe, landmarks)
            result['person_box'] = person['bbox']
            result['head_detected'] = True
            result['head_position'] = landmarks['nose']
            result['landmarks'] = landmarks
            analyses.append(result)
        
        if visualize:
            output_frame = self.visualize(output_frame, analyses, all_ppe)
        
        results = {
            'people': people,
            'all_ppe_detected': all_ppe,
            'analyses': analyses,
            'total_people': len(people),
            'compliant': sum(1 for a in analyses if a.get('compliant', False)),
            'non_compliant': sum(1 for a in analyses if not a.get('compliant', False)),
            'compliance_rate': (sum(1 for a in analyses if a.get('compliant', False)) / len(analyses) 
                              if analyses else 0.0)
        }
        
        return output_frame, results
    
    def visualize(self, frame: np.ndarray, analyses: List[Dict], all_ppe: List[Dict]) -> np.ndarray:
        output = frame.copy()
        
        # 1. Draw Skeleton/Pose first (so it's behind boxes)
        for analysis in analyses:
            if 'landmarks' in analysis:
                lms = analysis['landmarks']
                
                # Colors
                bone_color = (0, 255, 255) # Yellow
                joint_color = (0, 0, 255)  # Red
                
                # Points dict
                pts = {
                    'N': lms['nose'],
                    'LS': lms['shoulders'][0], 'RS': lms['shoulders'][1],
                    'LE': lms['elbows'][0],    'RE': lms['elbows'][1],
                    'LW': lms['wrists'][0],    'RW': lms['wrists'][1],
                    'LH': lms['hips'][0],      'RH': lms['hips'][1],
                    'TC': lms['torso_center']
                }
                
                # Draw Bones
                connections = [
                    ('LS', 'RS'), ('LS', 'LE'), ('LE', 'LW'), # Left Arm
                    ('RS', 'RE'), ('RE', 'RW'),               # Right Arm
                    ('LS', 'LH'), ('RS', 'RH'),               # Torso Sides
                    ('LH', 'RH'), ('LS', 'TC'), ('RS', 'TC')  # Hips & Neck
                ]
                
                for p1, p2 in connections:
                    cv2.line(output, pts[p1], pts[p2], bone_color, 2)
                
                # Draw Line from Nose to Torso Center (Neck-ish)
                cv2.line(output, pts['N'], pts['TC'], bone_color, 2)
                
                # Draw Joints
                for pt in pts.values():
                    cv2.circle(output, pt, 4, joint_color, -1)

        # 2. Draw PPE items
        for item in all_ppe:
            x1, y1, x2, y2 = item['bbox']
            conf_str = f"{int(item['confidence']*100)}%"
            label = f"{item['class_name']} {conf_str}"
            
            # Color based on validity if available
            color = (255, 200, 0) # Default cyan/yellow
            if item.get('is_valid_location', False):
                color = (0, 255, 0) # Green if valid
            elif 'is_valid_location' in item: # Only red if explicitly invalid
                color = (0, 0, 255) 

            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
            cv2.putText(output, label, (x1, y1-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # 3. Draw Person Status
        for analysis in analyses:
            x1, y1, x2, y2 = analysis['person_box']
            color = (0, 255, 0) if analysis.get('compliant', False) else (0, 0, 255)
            label = analysis.get('status', 'UNKNOWN')
            
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 3)
            cv2.putText(output, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            y_offset = y2 + 25
            cv2.putText(output, analysis.get('reason', ''), (x1, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return output

# ... Test function kept same logic
def test_helmet_detector(ppe_model_path: str = None, image_path: str = None, use_huggingface: bool = False):
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
        if frame is None: return
        output_frame, results = detector.process_frame(frame, visualize=True)
        cv2.imwrite('helmet_compliance_audit.jpg', output_frame)

if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 2:
        if sys.argv[1] == '--huggingface':
            image_path = sys.argv[2] if len(sys.argv) > 2 else None
            test_helmet_detector(use_huggingface=True, image_path=image_path)
        elif len(sys.argv) >= 3:
            test_helmet_detector(ppe_model_path=sys.argv[1], image_path=sys.argv[2])