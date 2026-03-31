import pygame
import time
import threading

class AlertSystem:
    def __init__(self):
        pygame.mixer.init()
        self.is_playing = False
        self.alert_thread = None
        
        # Create a simple beep sound programmatically
        self.create_alert_sound()
        
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
        if not self.is_playing:
            self.is_playing = True
            self.alert_thread = threading.Thread(target=self._play_alert_loop)
            self.alert_thread.daemon = True
            self.alert_thread.start()
            
    def _play_alert_loop(self):
        """Play alert sound in a loop"""
        while self.is_playing:
            self.alert_sound.play()
            time.sleep(0.6)  # Wait for sound to finish + small gap
            
    def stop_alert(self):
        """Stop playing alert sound"""
        self.is_playing = False
        pygame.mixer.stop()
        
    def play_single_alert(self):
        """Play alert sound once"""
        self.alert_sound.play()

# Import numpy for sound generation
import numpy as np
