import cv2 as cv
import numpy as np
import pygame
import struct

class DJVisualizer:
    def __init__(self, frame_width, frame_height):
        self.width = frame_width
        self.height = frame_height
        self.volume = 0.5
        
        # Separate waveforms for left and right decks
        self.left_waveform = []
        self.right_waveform = []
        self.line_speed = 3
        
    def get_audio_level(self):
        """Get current audio amplitude (0-100)"""
        # This is a simplified approach - pygame doesn't give us direct access to waveform data
        # Returns random for now, but we can enhance this
        if pygame.mixer.music.get_busy():
            return np.random.randint(20, 80)
        return 0
    
    def draw_waveform(self, frame, waveform_data, center_y, color_base, label):
        """Draw a single waveform line"""
        
        if pygame.mixer.music.get_busy():
            # Get audio level (we'll enhance this later with real analysis)
            amplitude = self.get_audio_level() // 2
            new_point = {
                'x': self.width,
                'y': center_y + np.random.choice([-amplitude, amplitude])
            }
            waveform_data.append(new_point)
        
        # Move all points left
        for point in waveform_data:
            point['x'] -= self.line_speed
        
        # Remove off-screen points
        waveform_data[:] = [p for p in waveform_data if p['x'] > 0]
        
        # Draw the waveform line
        if len(waveform_data) > 1:
            for i in range(len(waveform_data) - 1):
                pt1 = (waveform_data[i]['x'], waveform_data[i]['y'])
                pt2 = (waveform_data[i + 1]['x'], waveform_data[i + 1]['y'])
                
                # Gradient color based on position
                color_intensity = int(255 * (waveform_data[i]['x'] / self.width))
                color = (color_base[0], color_base[1], color_intensity)
                
                cv.line(frame, pt1, pt2, color, 3)
        
        # Center reference line
        cv.line(frame, (0, center_y), (self.width, center_y), (50, 50, 50), 1)
        
        # Label
        cv.putText(frame, label, (10, center_y - 50), 
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    def draw_dual_waveforms(self, frame, left_playing=False, right_playing=False):
        """Draw two separate waveforms for left and right decks"""
        # Top third for left deck
        left_y = int(self.height * 0.15)
        # Bottom of top third for right deck  
        right_y = int(self.height * 0.30)
        
        # Only update waveform if that deck is playing
        if left_playing:
            self.draw_waveform(frame, self.left_waveform, left_y, 
                             (0, 100), "LEFT DECK")
        else:
            self.left_waveform = []
            
        if right_playing:
            self.draw_waveform(frame, self.right_waveform, right_y, 
                             (100, 0), "RIGHT DECK")
        else:
            self.right_waveform = []
    
    def draw_volume_knob(self, frame, x, y, radius=80):
        """Draw a circular volume knob"""
        cv.circle(frame, (x, y), radius, (100, 100, 100), 3)
        
        angle = (self.volume * 270) - 135
        angle_rad = np.radians(angle)
        end_x = int(x + (radius - 10) * np.cos(angle_rad))
        end_y = int(y + (radius - 10) * np.sin(angle_rad))
        cv.line(frame, (x, y), (end_x, end_y), (0, 255, 0), 4)
        
        cv.circle(frame, (x, y), 10, (255, 255, 255), -1)
        
        volume_pct = int(self.volume * 100)
        cv.putText(frame, f"VOL: {volume_pct}%", (x - 40, y + radius + 35),
                   cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    def update_volume(self, new_volume):
        """Update volume (0.0 to 1.0)"""
        self.volume = max(0.0, min(1.0, new_volume))
        pygame.mixer.music.set_volume(self.volume)
    
    def draw_all(self, frame, left_playing=False, right_playing=False):
        """Draw all UI elements"""
        self.draw_dual_waveforms(frame, left_playing, right_playing)
        self.draw_volume_knob(frame, self.width - 120, self.height - 120)