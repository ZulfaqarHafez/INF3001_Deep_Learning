#!/usr/bin/env python3
"""
Simple usage script for PPE Pose Detection
Quick start guide for testing the enhanced PPE detection system.

Usage:
    python demo_usage.py
"""

from ppe_pose_detector import PPEPoseDetector
import cv2

def main():
    print("="*70)
    print("PPE POSE DETECTION - QUICK START DEMO")
    print("="*70)
    
    # Step 1: Initialize detector
    print("\n[1/4] Initializing detector...")
    print("      Loading YOLOv8 model and MediaPipe Pose...")
    
    model_path = 'runs/detect/ppe-4080-v12/weights/best.pt'
    
    try:
        detector = PPEPoseDetector(
            yolo_model_path=model_path,
            confidence_threshold=0.5
        )
        print("      ✓ Detector initialized successfully!")
    except Exception as e:
        print(f"      ✗ Error: {e}")
        print("\n      Please ensure:")
        print(f"      1. YOLOv8 model exists at: {model_path}")
        print("      2. Required packages are installed:")
        print("         pip install ultralytics mediapipe opencv-python")
        return
    
    # Step 2: Choose input mode
    print("\n[2/4] Select input mode:")
    print("      1. Test image")
    print("      2. Webcam (real-time)")
    print("      3. Video file")
    
    choice = input("\n      Enter choice (1/2/3): ").strip()
    
    # Step 3: Process based on choice
    print("\n[3/4] Processing...")
    
    if choice == '1':
        # Image mode
        image_path = input("      Enter image path: ").strip()
        
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"      ✗ Could not load image: {image_path}")
            return
        
        output_frame, results = detector.process_frame(frame, visualize=True)
        
        # Display results
        print("\n" + "="*70)
        print("RESULTS")
        print("="*70)
        print(f"Total People: {results['total_people']}")
        print(f"Compliant: {results['compliant_people']}")
        print(f"Non-Compliant: {results['non_compliant_people']}")
        print(f"Compliance Rate: {results['compliance_rate']*100:.1f}%")
        
        # Show detailed analysis
        for i, person in enumerate(results['person_analyses'], 1):
            print(f"\nPerson {i}:")
            print(f"  - Overall: {'✓ COMPLIANT' if person['overall_compliant'] else '✗ NON-COMPLIANT'}")
            if person['helmet_status']:
                print(f"  - Helmet: {person['helmet_status']['status']}")
            if person['vest_status']:
                print(f"  - Vest: {person['vest_status']['status']}")
        
        # Save output
        output_path = 'output_detected.jpg'
        cv2.imwrite(output_path, output_frame)
        print(f"\n✓ Output saved to: {output_path}")
        
        # Display
        cv2.imshow('PPE Detection Results', output_frame)
        print("\nPress any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    elif choice == '2':
        # Webcam mode
        print("      Starting webcam... (Press 'q' to quit)")
        detector.process_webcam()
    
    elif choice == '3':
        # Video mode
        video_path = input("      Enter video path: ").strip()
        output_path = input("      Enter output path (e.g., output.mp4): ").strip()
        
        detector.process_video(
            video_path=video_path,
            output_path=output_path,
            display=True
        )
        
        print(f"\n✓ Video processed and saved to: {output_path}")
    
    else:
        print("      Invalid choice!")
        return
    
    # Step 4: Done
    print("\n[4/4] Complete!")
    print("\n" + "="*70)
    print("For more advanced usage, see PPE_Pose_Detection_Demo.ipynb")
    print("="*70)


if __name__ == "__main__":
    main()
