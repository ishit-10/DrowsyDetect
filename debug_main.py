import cv2
import sys
from simple_detector import SimpleDrowsinessDetector

def main():
    """Debug version with detailed logging"""
    print("=" * 50)
    print("DROWSINESS DETECTION SYSTEM - DEBUG MODE")
    print("=" * 50)
    
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
        return
    
    print("Camera initialized. Starting detection...")
    print("DEBUG: Watch the console for detailed counter information")
    print("Press 'q' to quit, 'r' to reset counters")
    print("-" * 50)
    
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
        
        # Debug logging every 10 frames
        frame_count += 1
        if frame_count % 10 == 0:
            eye_seconds = detector.eye_closed_counter / 30.0
            mouth_seconds = detector.mouth_open_counter / 30.0
            print(f"Frame {frame_count:4d}: Eyes={eye_seconds:.1f}s, Mouth={mouth_seconds:.1f}s, "
                  f"Drowsy={drowsy_detected}, Alert={detector.alert_active}")
        
        # Display the frame
        cv2.imshow('Drowsiness Detection - Debug Mode', processed_frame)
        
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
