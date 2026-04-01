import cv2
import numpy as np
import pygame
import time
import threading
from scipy.spatial import distance as dist

class SimpleDrowsinessDetector:
    def __init__(self):
        # Initialize OpenCV face detector
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

        # Initialize pygame for sound (with fallback for headless environments like Streamlit Cloud)
        self.audio_available = False
        try:
            pygame.mixer.init()
            self.create_alert_sound()
            self.audio_available = True
        except pygame.error:
            # Audio not available (e.g., on Streamlit Cloud)
            print("⚠️ Audio not available - running in silent mode")
            self.alert_sound = None
        
        # Detection parameters (assuming ~30 FPS)
        self.eye_closed_counter = 0
        self.eye_closed_threshold = 60  # 2 seconds at 30 FPS (reduced from 3s)
        self.mouth_open_counter = 0
        self.mouth_open_threshold = 15  # 0.5 seconds at 30 FPS (reduced from 1s)
        
        # Alert state
        self.alert_active = False
        self.is_drowsy = False
        
        # Frame timing for accurate FPS calculation
        self.last_frame_time = time.time()
        self.fps = 30  # Estimated FPS
        
    def create_alert_sound(self):
        """Create a simple alert sound programmatically"""
        sample_rate = 22050
        duration = 0.5
        frequency = 880  # A5 note
        
        # Generate sine wave
        frames = int(duration * sample_rate)
        arr = np.zeros(frames)
        for i in range(frames):
            arr[i] = np.sin(2 * np.pi * frequency * i / sample_rate)
            
        # Convert to 16-bit integers
        arr = (arr * 32767).astype(np.int16)
        
        # Create stereo sound
        stereo_arr = np.zeros((frames, 2), dtype=np.int16)
        stereo_arr[:, 0] = arr
        stereo_arr[:, 1] = arr
        
        self.alert_sound = pygame.sndarray.make_sound(stereo_arr)
        
    def start_alert(self):
        """Start playing alert sound in a loop"""
        if not self.alert_active:
            print("🚨 ALERT STARTED - Drowsiness detected!")
            self.alert_active = True
            if self.audio_available:
                self.alert_thread = threading.Thread(target=self._play_alert_loop)
                self.alert_thread.daemon = True
                self.alert_thread.start()
            
    def _play_alert_loop(self):
        """Play alert sound in a loop"""
        while self.alert_active:
            try:
                self.alert_sound.play()
                time.sleep(0.6)
            except Exception as e:
                print(f"Sound error: {e}")
                break
            
    def stop_alert(self):
        """Stop playing alert sound"""
        if self.alert_active:
            print("🔇 ALERT STOPPED - Driver is alert now")
        self.alert_active = False
        if self.audio_available:
            pygame.mixer.stop()
        
    def detect_eyes(self, face_roi):
        """Detect eyes in face region with improved sensitivity"""
        # Use more lenient parameters for better detection
        eyes = self.eye_cascade.detectMultiScale(
            face_roi, 
            scaleFactor=1.05,  # Even smaller scale factor
            minNeighbors=2,   # Even more lenient
            minSize=(15, 15), # Even smaller minimum size
            maxSize=(100, 100) # Larger maximum size
        )
        return eyes
        
    def estimate_mouth_state(self, face_roi):
        """Improved mouth state estimation"""
        # Check if face_roi is already grayscale
        if len(face_roi.shape) == 3:
            gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        else:
            gray_face = face_roi
        
        # Look for dark regions that might indicate open mouth
        height, width = gray_face.shape
        mouth_region = gray_face[int(height*0.65):int(height*0.9), int(width*0.25):int(width*0.75)]
        
        if mouth_region.size > 0:
            # Calculate multiple features for better yawn detection
            variance = np.var(mouth_region)
            mean_intensity = np.mean(mouth_region)
            
            # Thresholds for yawn detection
            variance_threshold = 400  
            intensity_threshold = 120 
            
            # Both conditions must be met for yawn detection
            is_yawning = (variance > variance_threshold and mean_intensity < intensity_threshold)
            
            return is_yawning
        return False
        
    def detect_drowsiness(self, frame):
        """Main detection function with improved accuracy"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        
        drowsy_detected = False
        status_text = "Alert"
        status_color = (0, 255, 0)  # Green
        
        # Calculate FPS for accurate timing
        current_time = time.time()
        time_delta = current_time - self.last_frame_time
        self.last_frame_time = current_time
        
        for (x, y, w, h) in faces:
            # Draw face rectangle
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # Extract face ROI
            face_roi = gray[y:y+h, x:x+w]
            face_roi_color = frame[y:y+h, x:x+w]
            
            eyes = self.detect_eyes(face_roi)
            
            eyes_closed = len(eyes) == 0  # Only consider closed if NO eyes detected
            
            if eyes_closed:
                self.eye_closed_counter += 1
                print(f"👁️  Eyes closed - Counter: {self.eye_closed_counter}")
            else:
                if self.eye_closed_counter > 0:
                    print(f"👁️  Eyes reopened - Was closed for {self.eye_closed_counter} frames")
                self.eye_closed_counter = 0
                
                # Draw eye rectangles
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(face_roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
            
            # Check mouth state with improved detection
            mouth_open = self.estimate_mouth_state(face_roi)
            
            if mouth_open:
                self.mouth_open_counter += 1
                print(f"👄 Mouth open - Counter: {self.mouth_open_counter}")
            else:
                if self.mouth_open_counter > 0:
                    print(f"👄 Mouth closed - Was open for {self.mouth_open_counter} frames")
                self.mouth_open_counter = 0
            
            # Determine drowsiness with time-based thresholds
            eye_closed_seconds = self.eye_closed_counter / 30.0  # Convert to seconds
            mouth_open_seconds = self.mouth_open_counter / 30.0   # Convert to seconds
            
            if self.eye_closed_counter >= self.eye_closed_threshold:
                drowsy_detected = True
                status_text = f"DROWSY - Eyes closed {eye_closed_seconds:.1f}s!"
                status_color = (0, 0, 255)  # Red
            elif self.mouth_open_counter >= self.mouth_open_threshold:
                drowsy_detected = True
                status_text = f"DROWSY - Yawning {mouth_open_seconds:.1f}s!"
                status_color = (0, 0, 255)  # Red
            elif self.eye_closed_counter > 10:
                status_text = f"Blinking ({eye_closed_seconds:.1f}s)"
                status_color = (0, 255, 255)  # Yellow
            elif self.mouth_open_counter > 5:
                status_text = f"Mouth open ({mouth_open_seconds:.1f}s)"
                status_color = (255, 255, 0)  # Cyan
                
            # Draw status on face
            cv2.putText(frame, status_text, (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 2)
        
        # Handle alerts
        if drowsy_detected and not self.is_drowsy:
            print(f"🚨 DROWSINESS DETECTED - Starting alert system")
            self.is_drowsy = True
            self.start_alert()
        elif not drowsy_detected and self.is_drowsy:
            print(f"✅ DRIVER ALERT - Stopping alert system")
            self.is_drowsy = False
            self.stop_alert()
            self.eye_closed_counter = 0
            self.mouth_open_counter = 0
            
        # Draw overall status
        self.draw_status(frame, status_text, status_color)
        
        return frame, drowsy_detected
    
    def draw_status(self, frame, status_text, status_color):
        """Draw status information on the frame"""
        height, width = frame.shape[:2]
        
        # Draw main status
        cv2.putText(frame, f"Status: {status_text}", (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
        
        # Draw counters with time display
        eye_closed_seconds = self.eye_closed_counter / 30.0
        mouth_open_seconds = self.mouth_open_counter / 30.0
        
        cv2.putText(frame, f"Eyes Closed: {eye_closed_seconds:.1f}s / 2.0s", 
                   (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, f"Mouth Open: {mouth_open_seconds:.1f}s / 0.5s", 
                   (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Draw detection info
        cv2.putText(frame, f"Eyes detected: {'Yes' if self.eye_closed_counter == 0 else 'No'}", 
                   (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Draw alert status with flashing effect
        if self.alert_active:
            # Flashing red border for alert
            if int(time.time() * 2) % 2 == 0:  # Flash every 0.5 seconds
                cv2.rectangle(frame, (0, 0), (width-1, height-1), (0, 0, 255), 10)
            
            cv2.putText(frame, "🚨 ALERT ACTIVE 🚨", (20, height - 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(frame, "SOUND PLAYING", (20, height - 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
        # Draw instructions
        instructions = "Press 'q' to quit | Press 'r' to reset"
        cv2.putText(frame, instructions, (20, height - 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    def reset(self):
        """Reset all counters"""
        self.eye_closed_counter = 0
        self.mouth_open_counter = 0
        self.is_drowsy = False
        self.stop_alert()
        print("System reset - All counters cleared")


# Initialize pygame at module level for headless support
try:
    pygame.init()
except pygame.error:
    pass  # Ignore initialization errors in headless environments
