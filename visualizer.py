import cv2 as cv
import numpy as np
import pygame

class DJVisualizer:
    def __init__(self, frame_width, frame_height):
        self.width = frame_width
        self.height = frame_height
        self.volume = 0.5  # default vol (0.0 to 1.0)
        self.waveform_lines = []
        self.line_speed = 3  # pxls per frame
        
    def draw_waveform(self, frame):
        """Draw scrolling waveform lines at the top"""
        waveform_height = 120
        center_y = 60  # center of waveform area
        
        if pygame.mixer.music.get_busy():
            # create new wavy line segment at the right edge
            amplitude = np.random.randint(15, 40) 
            new_point = {
                'x': self.width,
                'y': center_y + np.random.choice([-amplitude, amplitude])
            }
            self.waveform_lines.append(new_point)
        
        for point in self.waveform_lines:
            point['x'] -= self.line_speed
        
        self.waveform_lines = [p for p in self.waveform_lines if p['x'] > 0]
        
        if len(self.waveform_lines) > 1:
            for i in range(len(self.waveform_lines) - 1):
                pt1 = (self.waveform_lines[i]['x'], self.waveform_lines[i]['y'])
                pt2 = (self.waveform_lines[i + 1]['x'], self.waveform_lines[i + 1]['y'])
                
                color_intensity = int(255 * (self.waveform_lines[i]['x'] / self.width))
                color = (0, color_intensity, 255)
                
                cv.line(frame, pt1, pt2, color, 3)
        
        cv.line(frame, (0, center_y), (self.width, center_y), (50, 50, 50), 1)
    
    def draw_volume_knob(self, frame, x, y, radius=50):
        """Draw a circular volume knob"""
        cv.circle(frame, (x, y), radius, (100, 100, 100), 3)
        
        angle = (self.volume * 270) - 135  #  -135 to 135 degrees
        angle_rad = np.radians(angle)
        end_x = int(x + (radius - 10) * np.cos(angle_rad))
        end_y = int(y + (radius - 10) * np.sin(angle_rad))
        cv.line(frame, (x, y), (end_x, end_y), (0, 255, 0), 4)
        
        cv.circle(frame, (x, y), 8, (255, 255, 255), -1)
        
        volume_pct = int(self.volume * 100)
        cv.putText(frame, f"{volume_pct}%", (x - 25, y + radius + 30),
                   cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    def update_volume(self, new_volume):
        """Update volume (0.0 to 1.0)"""
        self.volume = max(0.0, min(1.0, new_volume))
        pygame.mixer.music.set_volume(self.volume)
    
    def draw_all(self, frame):
        """Draw all UI elements"""
        self.draw_waveform(frame)
        self.draw_volume_knob(frame, self.width - 100, self.height - 100)