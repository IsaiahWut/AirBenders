import cv2 as cv
import numpy as np
import pygame

class DJVisualizer:
    def __init__(self, frame_width, frame_height):
        self.width = frame_width
        self.height = frame_height
        
        #  waveforms for left and right decks
        self.left_waveform = []
        self.right_waveform = []
        self.line_speed = 3
        
    def get_audio_level(self):
        """Get current audio amplitude (0-100)"""
        if pygame.mixer.music.get_busy():
            return np.random.randint(20, 80)
        return 0
    
    def draw_waveform(self, frame, waveform_data, center_y, color_base, label):
        """Draw a single waveform line"""
        
        if pygame.mixer.music.get_busy():
            amplitude = self.get_audio_level() // 2
            new_point = {
                'x': self.width,
                'y': center_y + np.random.choice([-amplitude, amplitude])
            }
            waveform_data.append(new_point)
        
        for point in waveform_data:
            point['x'] -= self.line_speed
        
        waveform_data[:] = [p for p in waveform_data if p['x'] > 0]
        
        # waveform line
        if len(waveform_data) > 1:
            for i in range(len(waveform_data) - 1):
                pt1 = (waveform_data[i]['x'], waveform_data[i]['y'])
                pt2 = (waveform_data[i + 1]['x'], waveform_data[i + 1]['y'])
                
                color_intensity = int(255 * (waveform_data[i]['x'] / self.width))
                color = (color_base[0], color_base[1], color_intensity)
                
                cv.line(frame, pt1, pt2, color, 3)
        
        # border lines
        border_offset = 60
        cv.line(frame, (0, center_y - border_offset), (self.width, center_y - border_offset), (100, 100, 100), 2)
        cv.line(frame, (0, center_y + border_offset), (self.width, center_y + border_offset), (100, 100, 100), 2)
        
        cv.putText(frame, label, (10, center_y - border_offset - 10), 
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    def draw_dual_waveforms(self, frame, left_playing=False, right_playing=False):
        """Draw two separate waveforms for left and right decks"""
        left_y = int(self.height * 0.12)   
        right_y = int(self.height * 0.27)   
        
        if left_playing:
            self.draw_waveform(frame, self.left_waveform, left_y, (0, 100), "LEFT DECK")
        else:
            self.left_waveform = []
            
        if right_playing:
            self.draw_waveform(frame, self.right_waveform, right_y, (100, 0), "RIGHT DECK")
        else:
            self.right_waveform = []
    
    def draw_all(self, frame, left_playing=False, right_playing=False):
        """Draw all UI elements"""
        self.draw_dual_waveforms(frame, left_playing, right_playing)