"""
Live Camera Helmet Compliance Detection
Real-time detection using webcam

Author: Zulfaqar
Project: INF3001 Deep Learning - PPE Detection (Live)
"""

import cv2
import time
from backend.helmet_compliance_detector import HelmetComplianceDetector


def live_helmet_detection(ppe_model_path: str, camera_id: int = 0):
    """
    Run live helmet detection from camera.
    
    Args:
        ppe_model_path: Path to your PPE model
        camera_id: Camera index (0 for default webcam)
    """
    print("="*70)
    print("LIVE HELMET COMPLIANCE DETECTION")
    print("="*70)
    print("\nInitializing detector...")
    
    # Initialize detector
    detector = HelmetComplianceDetector(
        ppe_model_path=ppe_model_path,
        confidence_threshold=0.5
    )
    
    # Optional: Adjust threshold for your setup
    detector.HELMET_ON_HEAD_THRESHOLD = 150  # Adjust if needed
    
    print(f"\n✓ Detector ready!")
    print(f"✓ Helmet threshold: {detector.HELMET_ON_HEAD_THRESHOLD}px")
    print(f"\nOpening camera {camera_id}...")
    
    # Open camera
    cap = cv2.VideoCapture(camera_id)
    
    if not cap.isOpened():
        print(f"❌ Error: Could not open camera {camera_id}")
        print("\nTroubleshooting:")
        print("- Try camera_id=1 if you have multiple cameras")
        print("- Check if another program is using the camera")
        print("- Make sure your camera is connected")
        return
    
    # Set camera resolution (optional)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    print("✓ Camera opened successfully!")
    print("\n" + "="*70)
    print("CONTROLS:")
    print("="*70)
    print("  [SPACE] - Pause/Resume")
    print("  [S]     - Save current frame")
    print("  [Q]     - Quit")
    print("  [+]     - Increase threshold (+10px)")
    print("  [-]     - Decrease threshold (-10px)")
    print("="*70)
    print("\nStarting detection...\n")
    
    # FPS tracking
    fps = 0
    frame_count = 0
    start_time = time.time()
    
    paused = False
    saved_count = 0
    
    while True:
        if not paused:
            ret, frame = cap.read()
            
            if not ret:
                print("❌ Error: Could not read frame from camera")
                break
            
            # Process frame
            output_frame, results = detector.process_frame(frame, visualize=True)
            
            # Calculate FPS
            frame_count += 1
            if frame_count % 30 == 0:
                end_time = time.time()
                fps = 30 / (end_time - start_time)
                start_time = time.time()
            
            # Add FPS and threshold info
            info_text = f"FPS: {fps:.1f} | Threshold: {detector.HELMET_ON_HEAD_THRESHOLD}px"
            cv2.putText(output_frame, info_text, (10, output_frame.shape[0] - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Print results to console
            if results['total_people'] > 0:
                print(f"\rPeople: {results['total_people']} | "
                      f"Compliant: {results['compliant']} | "
                      f"Non-Compliant: {results['non_compliant']} | "
                      f"FPS: {fps:.1f}", end='', flush=True)
        else:
            # Show paused message
            paused_frame = output_frame.copy()
            cv2.putText(paused_frame, "PAUSED - Press SPACE to resume", 
                       (int(paused_frame.shape[1]/2 - 250), int(paused_frame.shape[0]/2)),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
            output_frame = paused_frame
        
        # Display
        cv2.imshow('Live Helmet Compliance Detection', output_frame)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q') or key == ord('Q'):
            print("\n\n✓ Quitting...")
            break
        
        elif key == ord(' '):  # Space bar
            paused = not paused
            if paused:
                print("\n\n⏸ PAUSED")
            else:
                print("\n▶ RESUMED")
        
        elif key == ord('s') or key == ord('S'):
            # Save current frame
            filename = f'helmet_detection_save_{saved_count}.jpg'
            cv2.imwrite(filename, output_frame)
            saved_count += 1
            print(f"\n📸 Saved: {filename}")
        
        elif key == ord('+') or key == ord('='):
            # Increase threshold
            detector.HELMET_ON_HEAD_THRESHOLD += 10
            print(f"\n⬆ Threshold increased to {detector.HELMET_ON_HEAD_THRESHOLD}px")
        
        elif key == ord('-') or key == ord('_'):
            # Decrease threshold
            detector.HELMET_ON_HEAD_THRESHOLD = max(20, detector.HELMET_ON_HEAD_THRESHOLD - 10)
            print(f"\n⬇ Threshold decreased to {detector.HELMET_ON_HEAD_THRESHOLD}px")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("\n" + "="*70)
    print("SESSION SUMMARY")
    print("="*70)
    print(f"Total frames processed: {frame_count}")
    print(f"Images saved: {saved_count}")
    print(f"Final threshold: {detector.HELMET_ON_HEAD_THRESHOLD}px")
    print("="*70)
    print("\n✓ Camera closed. Goodbye!")


def test_camera_availability():
    """
    Test which cameras are available on your system.
    """
    print("Testing camera availability...\n")
    
    available_cameras = []
    
    for i in range(5):  # Test first 5 camera indices
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f"✓ Camera {i}: Available ({frame.shape[1]}x{frame.shape[0]})")
                available_cameras.append(i)
            cap.release()
        else:
            print(f"✗ Camera {i}: Not available")
    
    if not available_cameras:
        print("\n❌ No cameras detected!")
        print("\nTroubleshooting:")
        print("- Check if camera is connected")
        print("- Try closing other programs using the camera")
        print("- Check camera permissions")
    else:
        print(f"\n✓ Found {len(available_cameras)} camera(s)")
        print(f"Available camera IDs: {available_cameras}")
    
    return available_cameras


if __name__ == "__main__":
    import sys
    
    print("\n" + "="*70)
    print("LIVE HELMET COMPLIANCE DETECTION")
    print("="*70 + "\n")
    
    if len(sys.argv) < 2:
        print("Usage: python live_helmet_detection.py <ppe_model_path> [camera_id]")
        print("\nExample:")
        print("  python live_helmet_detection.py ppe-4080-v12/weights/best.pt")
        print("  python live_helmet_detection.py ppe-4080-v12/weights/best.pt 1")
        print("\nFirst, let's test which cameras are available:")
        print("-"*70)
        
        available = test_camera_availability()
        
        if available:
            print("\n💡 To start live detection, run:")
            print(f"   python live_helmet_detection.py <model_path> {available[0]}")
    else:
        model_path = sys.argv[1]
        camera_id = int(sys.argv[2]) if len(sys.argv) > 2 else 0
        
        live_helmet_detection(model_path, camera_id)
