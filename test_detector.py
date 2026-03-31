import cv2
import numpy as np
import time
from simple_detector import SimpleDrowsinessDetector

def create_test_frame(frame_num, width=640, height=480):
    """Create a simulated test frame"""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Create a simple background
    frame[:] = (50, 50, 50)  # Dark gray background
    
    # Simulate a face region
    face_x, face_y = width//2 - 100, height//2 - 120
    face_w, face_h = 200, 240
    
    # Draw face rectangle
    cv2.rectangle(frame, (face_x, face_y), (face_x + face_w, face_y + face_h), (100, 100, 200), -1)
    
    # Simulate eyes (open/closed based on frame number)
    eye_state = "open"
    if frame_num % 100 < 30:  # Eyes closed for 30 frames every 100 frames
        eye_state = "closed"
        # Closed eyes
        cv2.line(frame, (face_x + 50, face_y + 80), (face_x + 90, face_y + 80), (0, 0, 0), 3)
        cv2.line(frame, (face_x + 110, face_y + 80), (face_x + 150, face_y + 80), (0, 0, 0), 3)
    else:
        # Open eyes
        cv2.circle(frame, (face_x + 70, face_y + 80), 15, (0, 0, 0), -1)
        cv2.circle(frame, (face_x + 130, face_y + 80), 15, (0, 0, 0), -1)
        cv2.circle(frame, (face_x + 70, face_y + 80), 8, (255, 255, 255), -1)
        cv2.circle(frame, (face_x + 130, face_y + 80), 8, (255, 255, 255), -1)
    
    # Simulate mouth (open/closed based on frame number)
    if frame_num % 150 < 45:  # Mouth open (yawning) for 45 frames every 150 frames
        # Open mouth
        cv2.ellipse(frame, (face_x + 100, face_y + 180), (30, 20), 0, 0, 360, (0, 0, 0), -1)
    else:
        # Closed mouth
        cv2.line(frame, (face_x + 70, face_y + 180), (face_x + 130, face_y + 180), (0, 0, 0), 3)
    
    # Add frame counter
    cv2.putText(frame, f"Frame: {frame_num}", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    return frame, eye_state

def test_detector():
    """Test the drowsiness detector with simulated frames"""
    print("=" * 50)
    print("DROWSINESS DETECTOR TEST MODE")
    print("=" * 50)
    print("Testing with simulated video frames...")
    print()
    print("Test Pattern:")
    print("- Eyes closed: Frames 0-29, 100-129, 200-229...")
    print("- Mouth open (yawn): Frames 0-44, 150-194, 300-344...")
    print("- Normal state: All other frames")
    print()
    print("Press any key to exit")
    print("-" * 50)
    
    # Initialize detector
    detector = SimpleDrowsinessDetector()
    
    # Test with simulated frames
    frame_num = 0
    drowsy_periods = []
    
    while True:
        # Create test frame
        frame, eye_state = create_test_frame(frame_num)
        
        # Run detection
        processed_frame, drowsy_detected = detector.detect_drowsiness(frame)
        
        # Track drowsy periods
        if drowsy_detected:
            if not drowsy_periods or drowsy_periods[-1][1] != frame_num - 1:
                drowsy_periods.append([frame_num, frame_num])
            else:
                drowsy_periods[-1][1] = frame_num
        
        # Display the frame
        cv2.imshow('Drowsiness Detection - Test Mode', processed_frame)
        
        # Print status every 10 frames
        if frame_num % 10 == 0:
            status = "DROWSY" if drowsy_detected else "ALERT"
            print(f"Frame {frame_num:3d}: {status} (Eyes: {eye_state})")
        
        # Handle keyboard input
        key = cv2.waitKey(50) & 0xFF  # 50ms delay = ~20 FPS
        
        if key != 255:  # Any key pressed
            break
        
        frame_num += 1
        
        # Reset after 500 frames to continue testing
        if frame_num >= 500:
            print("\n" + "=" * 50)
            print("TEST CYCLE COMPLETE - Starting new cycle...")
            print("=" * 50)
            detector.reset()
            frame_num = 0
            drowsy_periods = []
    
    # Cleanup
    cv2.destroyAllWindows()
    detector.stop_alert()
    
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    print(f"Total frames processed: {frame_num}")
    print(f"Drowsy periods detected: {len(drowsy_periods)}")
    for i, (start, end) in enumerate(drowsy_periods):
        duration = end - start + 1
        print(f"  Period {i+1}: Frames {start}-{end} ({duration} frames)")
    print("\nTest completed successfully!")

if __name__ == "__main__":
    test_detector()
