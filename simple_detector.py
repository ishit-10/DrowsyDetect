import cv2
import time

class SimpleDrowsinessDetector:
    def __init__(self):
        # Initialize OpenCV face detector
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        self.mouth_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_mouth.xml')

        # Detection parameters (at 30 FPS)
        self.eye_closed_counter = 0
        self.eye_closed_threshold = 45  # 1.5 seconds at 30 FPS
        self.mouth_open_counter = 0
        self.mouth_open_threshold = 15  # 0.5 seconds at 30 FPS

        # Alert state
        self.alert_active = False
        self.is_drowsy = False

        # Frame timing
        self.last_frame_time = time.time()
        self.consecutive_no_face = 0

    def detect_eyes(self, face_roi):
        """Detect eyes in face region with optimized parameters"""
        eyes = self.eye_cascade.detectMultiScale(
            face_roi,
            scaleFactor=1.1,
            minNeighbors=3,
            minSize=(15, 15),
            maxSize=(100, 100)
        )
        return eyes

    def detect_mouth_open(self, face_roi, face_rect):
        """
        Detect if mouth is open using multiple methods:
        1. Try haarcascade mouth detector first
        2. Fall back to color/texture analysis
        """
        _, _, w, h = face_rect

        # Method 1: Try haarcascade mouth detector
        mouth_roi = face_roi[int(h*0.5):int(h*0.9), int(w*0.25):int(w*0.75)]

        if mouth_roi.size > 0:
            mouths = self.mouth_cascade.detectMultiScale(
                mouth_roi,
                scaleFactor=1.1,
                minNeighbors=2,
                minSize=(20, 20)
            )

            # If mouth detected and it's relatively tall, it's open (yawning)
            if len(mouths) > 0:
                for (_, _, mw, mh) in mouths:
                    aspect_ratio = mh / max(mw, 1)
                    if aspect_ratio > 0.4:  # Open mouth is more square/tall
                        return True

        # Method 2: Color/texture analysis for open mouth detection
        # Open mouth appears as a dark region with specific characteristics
        if len(face_roi.shape) == 3:
            gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        else:
            gray_face = face_roi

        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray_face, (5, 5), 0)

        # Define mouth region (lower part of face)
        mouth_y_start = int(h * 0.55)
        mouth_y_end = int(h * 0.85)
        mouth_x_start = int(w * 0.3)
        mouth_x_end = int(w * 0.7)

        mouth_region = blurred[mouth_y_start:mouth_y_end, mouth_x_start:mouth_x_end]

        if mouth_region.size == 0:
            return False

        # Threshold to find dark regions
        _, thresh = cv2.threshold(mouth_region, 60, 255, cv2.THRESH_BINARY_INV)

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return False

        # Find largest contour in center region
        max_area = 0
        center_x = (mouth_x_end - mouth_x_start) // 2

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 150:  # Minimum area threshold
                cnt_x, _, cnt_w, _ = cv2.boundingRect(contour)
                contour_center = cnt_x + cnt_w // 2

                # Check if contour is near center
                if abs(contour_center - center_x) < (mouth_x_end - mouth_x_start) * 0.3:
                    if area > max_area:
                        max_area = area

        # Open mouth typically has area > 400 pixels
        # Aspect ratio check: open mouth is taller than wide
        if max_area > 400:
            return True

        return False

    def detect_drowsiness(self, frame):
        """
        Main detection function with robust eye and mouth detection.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)

        drowsy_detected = False
        status_text = "Alert"
        status_color = (0, 255, 0)  # Green
        face_detected = len(faces) > 0

        # Update timing
        current_time = time.time()
        self.last_frame_time = current_time

        for (x, y, w, h) in faces:
            # Draw face rectangle
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            # Extract face ROI
            face_roi = gray[y:y+h, x:x+w]
            face_roi_color = frame[y:y+h, x:x+w]

            # ===== EYE DETECTION =====
            eyes = self.detect_eyes(face_roi)
            eyes_detected = len(eyes) >= 2  # Need both eyes for reliable detection

            if eyes_detected:
                # Both eyes detected - they are OPEN
                if self.eye_closed_counter > 0:
                    print(f"  Eyes reopened - Was closed for {self.eye_closed_counter} frames")
                self.eye_closed_counter = 0

                # Draw eye rectangles
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(face_roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
            else:
                # Eyes not detected (could be closed or face angle issue)
                # Only count if we had eyes before (continuous tracking)
                if self.eye_closed_counter >= 0:
                    self.eye_closed_counter += 1
                    if self.eye_closed_counter > 5:  # Only log after initial frames
                        print(f"  Eyes closed - Counter: {self.eye_closed_counter}")

            # ===== MOUTH DETECTION =====
            mouth_open = self.detect_mouth_open(face_roi, (x, y, w, h))

            if mouth_open:
                self.mouth_open_counter += 1
                if self.mouth_open_counter > 5:
                    print(f"  Mouth open (yawning) - Counter: {self.mouth_open_counter}")
            else:
                if self.mouth_open_counter > 0:
                    print(f"  Mouth closed - Was open for {self.mouth_open_counter} frames")
                self.mouth_open_counter = 0

            # ===== DROWSINESS DECISION =====
            eye_closed_seconds = self.eye_closed_counter / 30.0
            mouth_open_seconds = self.mouth_open_counter / 30.0

            # Check for drowsiness conditions
            if self.eye_closed_counter >= self.eye_closed_threshold:
                drowsy_detected = True
                status_text = "DROWSY"
                status_color = (0, 0, 255)  # Red
            elif self.mouth_open_counter >= self.mouth_open_threshold:
                drowsy_detected = True
                status_text = "DROWSY"
                status_color = (0, 0, 255)  # Red
            elif not face_detected:
                status_text = "No face detected"
                status_color = (128, 128, 128)
                self.consecutive_no_face += 1
                # Reset after 2 seconds of no face
                if self.consecutive_no_face > 60:
                    self.eye_closed_counter = 0
                    self.mouth_open_counter = 0
            elif not eyes_detected:
                status_text = "Position face properly"
                status_color = (0, 255, 255)  # Yellow
            elif self.eye_closed_counter > 15:
                status_text = f"Blinking ({eye_closed_seconds:.1f}s)"
                status_color = (0, 255, 255)
            elif self.mouth_open_counter > 5:
                status_text = f"Mouth opening ({mouth_open_seconds:.1f}s)"
                status_color = (255, 255, 0)
            else:
                status_text = "Alert"
                status_color = (0, 255, 0)
                self.consecutive_no_face = 0

            # Draw status on face
            cv2.putText(frame, status_text, (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 2)

        # ===== ALERT HANDLING =====
        if drowsy_detected and not self.is_drowsy:
            print(f"DROWSINESS DETECTED - Starting alert")
            self.is_drowsy = True
            self.start_alert()
        elif not drowsy_detected and self.is_drowsy:
            print(f"Driver alert - Stopping alert")
            self.is_drowsy = False
            self.stop_alert()
            self.eye_closed_counter = 0
            self.mouth_open_counter = 0

        # Draw overall status
        self.draw_status(frame, status_text, status_color, face_detected)

        return frame, drowsy_detected

    def start_alert(self):
        """Start visual alert indicator"""
        if not self.alert_active:
            print("ALERT STARTED - Drowsiness detected!")
            self.alert_active = True

    def stop_alert(self):
        """Stop alert indicator"""
        if self.alert_active:
            print("ALERT STOPPED - Driver is alert now")
        self.alert_active = False

    def draw_status(self, frame, status_text, status_color, face_detected):
        """Draw status information on the frame"""
        height, width = frame.shape[:2]

        # Draw main status
        cv2.putText(frame, f"Status: {status_text}", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)

        # Draw counters with time display
        eye_closed_seconds = self.eye_closed_counter / 30.0
        mouth_open_seconds = self.mouth_open_counter / 30.0

        cv2.putText(frame, f"Eyes Closed: {eye_closed_seconds:.1f}s / 1.5s",
                   (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, f"Mouth Open: {mouth_open_seconds:.1f}s / 0.5s",
                   (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Draw detection info
        cv2.putText(frame, f"Face: {'Yes' if face_detected else 'No'}",
                   (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Draw alert status with flashing effect
        if self.alert_active:
            # Flashing red border for alert
            if int(time.time() * 2) % 2 == 0:
                cv2.rectangle(frame, (0, 0), (width-1, height-1), (0, 0, 255), 10)

            cv2.putText(frame, "ALERT ACTIVE", (20, height - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    def reset(self):
        """Reset all counters"""
        self.eye_closed_counter = 0
        self.mouth_open_counter = 0
        self.is_drowsy = False
        self.stop_alert()
        self.consecutive_no_face = 0
        print("System reset - All counters cleared")
