import cv2
import time
from scipy.spatial import distance as dist

class SimpleDrowsinessDetector:
    def __init__(self):
        # Initialize OpenCV face detector
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

        # Detection parameters (assuming ~30 FPS)
        self.eye_closed_counter = 0
        self.eye_closed_threshold = 60  # 2 seconds at 30 FPS
        self.mouth_open_counter = 0
        self.mouth_open_threshold = 15  # 0.5 seconds at 30 FPS

        # Alert state
        self.alert_active = False
        self.is_drowsy = False

        # Frame timing for accurate FPS calculation
        self.last_frame_time = time.time()
        self.fps = 30  # Estimated FPS

        # Eye aspect ratio threshold - below this means eyes are closed
        self.EYE_AR_THRESH = 0.25

        # Mouth aspect ratio threshold - above this means yawning
        self.MOUTH_AR_THRESH = 0.7

    def eye_aspect_ratio(self, eye_points):
        """Calculate the eye aspect ratio for eye closure detection"""
        # Vertical eye landmarks
        v1 = dist.euclidean(eye_points[1], eye_points[5])
        v2 = dist.euclidean(eye_points[2], eye_points[4])

        # Horizontal eye landmarks
        h = dist.euclidean(eye_points[0], eye_points[3])

        # EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)
        ear = (v1 + v2) / (2.0 * h)
        return ear

    def detect_eyes(self, face_roi):
        """Detect eyes in face region"""
        eyes = self.eye_cascade.detectMultiScale(
            face_roi,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(20, 20)
        )
        return eyes

    def estimate_mouth_state(self, face_roi, face_rect):
        """
        Detect yawn by analyzing the mouth region geometry.
        Uses contour analysis on the lower part of the face.
        """
        x, y, w, h = face_rect

        # Convert to grayscale if needed
        if len(face_roi.shape) == 3:
            gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        else:
            gray_face = face_roi

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray_face, (5, 5), 0)

        # Apply adaptive threshold to find dark regions (like open mouth)
        thresh = cv2.adaptiveThreshold(
            blurred, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            11, 2
        )

        # Define mouth region (lower half of face, centered horizontally)
        mouth_y_start = int(h * 0.5)
        mouth_y_end = int(h * 0.85)
        mouth_x_start = int(w * 0.2)
        mouth_x_end = int(w * 0.8)

        mouth_thresh = thresh[mouth_y_start:mouth_y_end, mouth_x_start:mouth_x_end]

        if mouth_thresh.size == 0:
            return False

        # Find contours in mouth region
        contours, _ = cv2.findContours(mouth_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return False

        # Find the largest contour that's in the center (likely the mouth)
        mouth_contour = None
        max_area = 0
        center_x = (mouth_x_end - mouth_x_start) // 2

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 200:  # Filter small noise
                # Get bounding box
                cnt_x, _, cnt_w, _ = cv2.boundingRect(contour)
                contour_center = cnt_x + cnt_w // 2

                # Prefer contours near the center
                if abs(contour_center - center_x) < w * 0.3:
                    if area > max_area:
                        max_area = area
                        mouth_contour = contour

        if mouth_contour is None:
            return False

        # Calculate aspect ratio of the mouth contour
        _, _, mw, mh = cv2.boundingRect(mouth_contour)

        # A yawn typically has a larger vertical opening relative to width
        # Normal mouth: wide and short (low aspect ratio)
        # Yawning: more square or tall (higher aspect ratio)
        aspect_ratio = mh / max(mw, 1)

        # Yawn threshold: aspect ratio > 0.5 and sufficient area
        # This means the mouth opening is significantly vertical
        is_yawning = aspect_ratio > 0.5 and max_area > 800

        return is_yawning

    def detect_drowsiness(self, frame):
        """
        Main detection function.
        Only detects drowsiness when faces AND eyes are properly detected.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)

        drowsy_detected = False
        status_text = "Alert"
        status_color = (0, 255, 0)  # Green
        face_detected = len(faces) > 0

        # Calculate FPS for accurate timing
        current_time = time.time()
        self.last_frame_time = current_time

        for (x, y, w, h) in faces:
            # Draw face rectangle
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            # Extract face ROI
            face_roi = gray[y:y+h, x:x+w]
            face_roi_color = frame[y:y+h, x:x+w]

            # Detect eyes - this is the key fix
            # We need to SEE eyes to determine if they're open or closed
            eyes = self.detect_eyes(face_roi)
            eyes_detected = len(eyes) > 0

            if eyes_detected:
                # Eyes are visible - check if they're open or closed
                # For now, if we detect eyes, we assume they're open
                # (Haarcascade eye detector typically detects open eyes)
                if self.eye_closed_counter > 0:
                    print(f"  Eyes reopened - Was closed for {self.eye_closed_counter} frames")
                self.eye_closed_counter = 0

                # Draw eye rectangles
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(face_roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
            else:
                # Eyes not detected - could be closed OR face at wrong angle
                # Only increment counter if we previously had eyes detected
                # This prevents false positives from poor camera angles
                if eyes_detected or self.eye_closed_counter > 0:
                    self.eye_closed_counter += 1
                    print(f"  Eyes not visible - Counter: {self.eye_closed_counter}")

            # Check mouth state with improved detection
            mouth_open = self.estimate_mouth_state(face_roi, (x, y, w, h))

            if mouth_open:
                self.mouth_open_counter += 1
                print(f"  Mouth open (yawning) - Counter: {self.mouth_open_counter}")
            else:
                if self.mouth_open_counter > 0:
                    print(f"  Mouth closed - Was open for {self.mouth_open_counter} frames")
                self.mouth_open_counter = 0

            # Determine drowsiness with time-based thresholds
            eye_closed_seconds = self.eye_closed_counter / 30.0
            mouth_open_seconds = self.mouth_open_counter / 30.0

            # Only trigger drowsiness if we have enough evidence
            if self.eye_closed_counter >= self.eye_closed_threshold:
                drowsy_detected = True
                status_text = f"DROWSY - Eyes closed {eye_closed_seconds:.1f}s!"
                status_color = (0, 0, 255)  # Red
            elif self.mouth_open_counter >= self.mouth_open_threshold:
                drowsy_detected = True
                status_text = f"DROWSY - Yawning {mouth_open_seconds:.1f}s!"
                status_color = (0, 0, 255)  # Red
            elif not face_detected:
                status_text = "No face detected"
                status_color = (128, 128, 128)  # Gray
                # Reset counters when no face
                self.eye_closed_counter = 0
                self.mouth_open_counter = 0
            elif not eyes_detected:
                status_text = "Position face properly"
                status_color = (0, 255, 255)  # Yellow
            elif self.eye_closed_counter > 10:
                status_text = f"Blinking ({eye_closed_seconds:.1f}s)"
                status_color = (0, 255, 255)  # Yellow
            elif self.mouth_open_counter > 5:
                status_text = f"Mouth opening ({mouth_open_seconds:.1f}s)"
                status_color = (255, 255, 0)  # Cyan

            # Draw status on face
            cv2.putText(frame, status_text, (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 2)

        # Handle alerts
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

        cv2.putText(frame, f"Eyes Closed: {eye_closed_seconds:.1f}s / 2.0s",
                   (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, f"Mouth Open: {mouth_open_seconds:.1f}s / 0.5s",
                   (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Draw detection info
        cv2.putText(frame, f"Face: {'Yes' if face_detected else 'No'}",
                   (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Draw alert status with flashing effect
        if self.alert_active:
            # Flashing red border for alert
            if int(time.time() * 2) % 2 == 0:  # Flash every 0.5 seconds
                cv2.rectangle(frame, (0, 0), (width-1, height-1), (0, 0, 255), 10)

            cv2.putText(frame, "ALERT ACTIVE", (20, height - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    def reset(self):
        """Reset all counters"""
        self.eye_closed_counter = 0
        self.mouth_open_counter = 0
        self.is_drowsy = False
        self.stop_alert()
        print("System reset - All counters cleared")
