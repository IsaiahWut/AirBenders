"""
DJ Session Recorder
Records video feed and audio mix simultaneously
"""

import cv2 as cv
import numpy as np
import soundfile as sf
import threading
import time
from pathlib import Path
from datetime import datetime
import subprocess
import os

class DJRecorder:
    def __init__(self, output_folder="recordings"):
        self.output_folder = Path(output_folder)
        self.output_folder.mkdir(exist_ok=True)

        # Recording state
        self.is_recording = False
        self.video_writer = None
        self.audio_buffer = []
        self.audio_lock = threading.Lock()

        # File paths
        self.session_name = None
        self.video_path = None
        self.audio_path = None
        self.final_path = None

        # Video settings
        self.fps = 30
        self.frame_size = None

        # Audio settings
        self.sample_rate = 44100
        self.audio_channels = 2

        # Recording start time
        self.start_time = None

        # Frame counter for sync
        self.frame_count = 0

    def start_recording(self, frame_width, frame_height):
        if self.is_recording:
            print("⚠️  Already recording!")
            return False

        # 🔧 FIX 1: force EVEN dimensions (mp4/h264 REQUIRE THIS)
        frame_width  -= frame_width  % 2
        frame_height -= frame_height % 2
        self.frame_size = (frame_width, frame_height)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_name = f"dj_session_{timestamp}"

        self.video_path = self.output_folder / f"{self.session_name}_video.mp4"
        self.audio_path = self.output_folder / f"{self.session_name}_audio.wav"
        self.final_path = self.output_folder / f"{self.session_name}.mp4"

        # 🔧 FIX 2: macOS-safe codec fallback
        self.video_writer = None
        for codec in ['avc1', 'mp4v']:
            fourcc = cv.VideoWriter_fourcc(*codec)
            writer = cv.VideoWriter(
                str(self.video_path),
                fourcc,
                self.fps,
                self.frame_size
            )
            if writer.isOpened():
                self.video_writer = writer
                print(f"✅ VideoWriter opened with codec: {codec}")
                break

        if self.video_writer is None:
            print("❌ Failed to initialize video writer")
            return False

        with self.audio_lock:
            self.audio_buffer = []

        self.is_recording = True
        self.start_time = time.time()
        self.frame_count = 0

        print(f"🔴 Recording started: {self.session_name}")
        return True

    def add_video_frame(self, frame):
        if not self.is_recording or self.video_writer is None:
            return

        if frame.shape[1] != self.frame_size[0] or frame.shape[0] != self.frame_size[1]:
            frame = cv.resize(frame, self.frame_size)

        # 🔧 FIX: duplicate frames to match real elapsed time
        elapsed = time.time() - self.start_time
        expected_frame_count = int(elapsed * self.fps)
        while self.frame_count < expected_frame_count:
            self.video_writer.write(frame)
            self.frame_count += 1

    def add_audio_chunk(self, audio_chunk, sample_rate):
        if not self.is_recording:
            return
        with self.audio_lock:
            self.audio_buffer.append(audio_chunk.copy())

    def stop_recording(self):
        if not self.is_recording:
            print("⚠️  Not currently recording!")
            return None

        self.is_recording = False
        duration = time.time() - self.start_time
        print(f"⏸️  Recording stopped. Duration: {duration:.1f}s")

        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None

        print("🎵 Saving audio...")
        with self.audio_lock:
            if self.audio_buffer:
                audio = np.concatenate(self.audio_buffer, axis=0)
                sf.write(str(self.audio_path), audio, self.sample_rate)
                print(f"   ✓ Audio saved: {self.audio_path}")
            else:
                print("   ⚠️  No audio recorded")
                self.audio_path = None
            self.audio_buffer = []

        print("🎥 Combining video and audio...")
        success = self._combine_video_audio()

        if success:
            print(f"✅ Recording saved: {self.final_path}")
            if self.video_path.exists():
                self.video_path.unlink()
            if self.audio_path and self.audio_path.exists():
                self.audio_path.unlink()
            return str(self.final_path)

        print("❌ Failed to combine video and audio")
        return None

    def _combine_video_audio(self):
        try:
            subprocess.run(['ffmpeg', '-version'], capture_output=True, timeout=5)
        except:
            print("   ✗ ffmpeg not found. Install with: brew install ffmpeg")
            return False

        if self.audio_path and self.audio_path.exists():
            # 🔧 FIX: Force fps during combination to preserve duration
            cmd = [
                'ffmpeg',
                '-y',
                '-r', str(self.fps),              # force video framerate
                '-i', str(self.video_path),
                '-i', str(self.audio_path),
                '-c:v', 'libx264',
                '-preset', 'medium',
                '-crf', '23',
                '-c:a', 'aac',
                '-b:a', '192k',
                '-shortest',
                str(self.final_path)
            ]
        else:
            cmd = [
                'ffmpeg',
                '-y',
                '-r', str(self.fps),              # force video framerate
                '-i', str(self.video_path),
                '-c:v', 'libx264',
                '-preset', 'medium',
                '-crf', '23',
                str(self.final_path)
            ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(result.stderr)
        return result.returncode == 0 and self.final_path.exists()

    def get_recording_duration(self):
        if not self.is_recording or not self.start_time:
            return 0
        return time.time() - self.start_time

    def is_ffmpeg_available(self):
        try:
            subprocess.run(['ffmpeg', '-version'], capture_output=True, timeout=5)
            return True
        except:
            return False


class RecordButton:
    def __init__(self, center, radius=35):
        self.cx, self.cy = center
        self.radius = radius
        self.is_recording = False
        self.was_pinching = False
        self.pulse = 0

    def contains(self, x, y):
        dx, dy = x - self.cx, y - self.cy
        return dx*dx + dy*dy <= self.radius*self.radius

    def update(self, pinch_positions, is_recording):
        self.is_recording = is_recording
        pinching = any(self.contains(x, y) for x, y in pinch_positions)
        newly_pinched = pinching and not self.was_pinching
        self.was_pinching = pinching
        if self.is_recording:
            self.pulse += 0.1
        return newly_pinched

    def draw(self, frame, duration=0):
        import math
        if self.is_recording:
            intensity = int(100 + 155 * (0.5 + 0.5 * math.sin(self.pulse)))
            color = (0, 0, intensity)
        else:
            color = (60, 60, 60)

        cv.circle(frame, (self.cx, self.cy), self.radius, color, -1)
        cv.circle(frame, (self.cx, self.cy), self.radius, (255,255,255), 2)

        if self.is_recording:
            text = f"{int(duration//60):02d}:{int(duration%60):02d}"
            scale = 0.5
        else:
            text = "REC"
            scale = 0.6

        size = cv.getTextSize(text, cv.FONT_HERSHEY_SIMPLEX, scale, 2)[0]
        cv.putText(
            frame,
            text,
            (self.cx - size[0]//2, self.cy + size[1]//2),
            cv.FONT_HERSHEY_SIMPLEX,
            scale,
            (255,255,255),
            2
        )
