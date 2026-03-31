import cv2
import sys
from simple_detector import SimpleDrowsinessDetector

def main():
    """Main application function using simplified detector"""
    print("=" * 50)
    print("DROWSINESS DETECTION SYSTEM - SIMPLIFIED")
    print("=" * 50)
    print("Using OpenCV built-in detectors (no external model needed)")
    print()
    
    # Initialize the drowsiness detector
    try:
        detector = SimpleDrowsinessDetector()
        print("System initialized successfully!")
    except Exception as e:
        print(f"Error initializing detector: {e}")
        return
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera.")
        print("Please check if your camera is connected and not being used by another application.")
        return
    
    print("Camera initialized. Starting detection...")
    print("Press 'q' to quit, 'r' to reset counters")
    print("-" * 50)
    print()
    print("Detection Info:")
    print("- Eyes closed for 60+ frames triggers alert")
    print("- Yawning detected for 15+ frames triggers alert")
    print("- Alert stops when you become alert again")
    print()
    
    # Main detection loop
    frame_count = 0
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Could not read from camera.")
            break
        
        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Perform drowsiness detection
        processed_frame, drowsy_detected = detector.detect_drowsiness(frame)
        
        # Display the frame
        cv2.imshow('Drowsiness Detection - Simplified', processed_frame)
        
        # Log status every 30 frames
        frame_count += 1
        if frame_count % 30 == 0:
            if drowsy_detected:
                print(f"⚠️  DROWSINESS DETECTED - Alert active")
            else:
                print(f"✅ Status: Alert")
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            print("Quitting application...")
            break
        elif key == ord('r'):
            detector.reset()
    
    cap.release()
    cv2.destroyAllWindows()
    detector.stop_alert()
    print("Application closed successfully.")

if __name__ == "__main__":
    main()
