import os
import time
import sounddevice as sd
import soundfile as sf
import numpy as np
import threading
import scipy.signal as signal

# Import scratch configuration
try:
    from scratch_config import *
except ImportError:
    # Default values if config file doesn't exist
    USE_TRACK_SCRATCH = True
    SCRATCH_SENSITIVITY = 0.5
    PITCH_SHIFT_MIN = 0.3
    PITCH_SHIFT_MAX = 3.0
    SCRATCH_BUFFER_DURATION = 2.0
    SCRATCH_BASE_VOLUME = 0.4
    SCRATCH_MAX_VOLUME = 0.8
    SCRATCH_SPEED_VOLUME_FACTOR = 0.4
    SPEED_MULTIPLIER = 2.0

# -----------------------------
# Track State Management
# -----------------------------
songs = []
active_track = -1  # Currently active track (for display purposes)

# Scratch sound state
scratch_audio = None
scratch_sample_rate = None
scratch_playback_position = 0
scratch_is_playing = False
scratch_speed = 0.0
scratch_direction = 1  # 1 for forward, -1 for backward
scratch_lock = threading.Lock()

# Track scratching state (using actual track audio)
scratch_track_index = -1
scratch_track_buffer = None
scratch_track_buffer_position = 0
use_track_scratch = USE_TRACK_SCRATCH  # Configurable

class TrackState:
    def __init__(self, filepath, target_sample_rate=44100, target_channels=2):
        self.filepath = filepath
        self.audio_data, self.sample_rate = sf.read(filepath, dtype='float32')
        
        # Convert to mono if single channel
        if len(self.audio_data.shape) == 1:
            self.audio_data = self.audio_data.reshape(-1, 1)
        
        # Resample if needed
        if self.sample_rate != target_sample_rate:
            print(f"  Resampling from {self.sample_rate}Hz to {target_sample_rate}Hz...")
            num_samples = int(len(self.audio_data) * target_sample_rate / self.sample_rate)
            self.audio_data = signal.resample(self.audio_data, num_samples)
            self.sample_rate = target_sample_rate
        
        # Convert to target number of channels
        current_channels = self.audio_data.shape[1]
        if current_channels != target_channels:
            if current_channels == 1 and target_channels == 2:
                # Mono to stereo: duplicate the channel
                print(f"  Converting mono to stereo...")
                self.audio_data = np.repeat(self.audio_data, 2, axis=1)
            elif current_channels == 2 and target_channels == 1:
                # Stereo to mono: average the channels
                print(f"  Converting stereo to mono...")
                self.audio_data = np.mean(self.audio_data, axis=1, keepdims=True)
        
        self.position = 0.0  # Position in seconds
        self.last_update_time = None
        self.is_scrubbing = False
        self.is_playing = False
        self.was_playing_before_scrub = False
        self.volume = 1.0  # 0.0 to 1.0
        self.stream = None
        self.duration = len(self.audio_data) / self.sample_rate
        self.playback_position = 0  # Position in samples
        self.lock = threading.Lock()
        
        print(f"  Duration: {self.duration:.2f}s, Sample rate: {self.sample_rate}Hz, Channels: {self.audio_data.shape[1]}")

track_states = {}  # index -> TrackState
mixer_streams = []  # Keep track of all active streams

# -----------------------------
# Load Scratch Sound
# -----------------------------
def load_scratch_sound(target_sample_rate=44100, target_channels=2):
    """Load the scratch sound effect"""
    global scratch_audio, scratch_sample_rate
    
    scratch_paths = [
        "scratch.wav",
        "scratch.mp3",
        "sounds/scratch.wav",
        "sounds/scratch.mp3",
        "assets/scratch.wav",
        "assets/scratch.mp3"
    ]
    
    for path in scratch_paths:
        if os.path.exists(path):
            try:
                print(f"Loading scratch sound from {path}...")
                audio_data, sample_rate = sf.read(path, dtype='float32')
                
                # Convert to proper format
                if len(audio_data.shape) == 1:
                    audio_data = audio_data.reshape(-1, 1)
                
                # Resample if needed
                if sample_rate != target_sample_rate:
                    num_samples = int(len(audio_data) * target_sample_rate / sample_rate)
                    audio_data = signal.resample(audio_data, num_samples)
                    sample_rate = target_sample_rate
                
                # Convert to target channels
                current_channels = audio_data.shape[1]
                if current_channels == 1 and target_channels == 2:
                    audio_data = np.repeat(audio_data, 2, axis=1)
                elif current_channels == 2 and target_channels == 1:
                    audio_data = np.mean(audio_data, axis=1, keepdims=True)
                
                scratch_audio = audio_data
                scratch_sample_rate = sample_rate
                print(f"  ✓ Loaded scratch sound: {len(audio_data)} samples, {sample_rate}Hz")
                return True
            except Exception as e:
                print(f"  Error loading {path}: {e}")
    
    print("⚠ Warning: No scratch sound file found. Scratching will be silent.")
    print("  To add scratch sounds, place 'scratch.wav' or 'scratch.mp3' in your project folder")
    print("  You can generate one using: python generate_scratch_sound.py")
    return False

# -----------------------------
# Audio Mixer
# -----------------------------
class AudioMixer:
    def __init__(self, sample_rate=44100):
        self.sample_rate = sample_rate
        self.streams = {}
        self.lock = threading.Lock()
        
    def callback(self, outdata, frames, time_info, status):
        """Mix all active tracks and scratch sounds"""
        global scratch_playback_position, scratch_is_playing
        global scratch_track_buffer, scratch_track_buffer_position, scratch_track_index
        
        outdata.fill(0)
        
        with self.lock:
            # Mix music tracks (SKIP if currently being scratched)
            for index, state in track_states.items():
                # CRITICAL: Don't play the track if it's being scrubbed
                if state.is_playing and not state.is_scrubbing:
                    # Get audio from this track
                    start_frame = state.playback_position
                    end_frame = start_frame + frames
                    
                    if end_frame <= len(state.audio_data):
                        chunk = state.audio_data[start_frame:end_frame]
                    else:
                        remaining = len(state.audio_data) - start_frame
                        if remaining > 0:
                            chunk = state.audio_data[start_frame:]
                            chunk = np.pad(chunk, ((0, frames - remaining), (0, 0)), mode='constant')
                        else:
                            chunk = np.zeros((frames, state.audio_data.shape[1]))
                            state.is_playing = False
                            continue
                    
                    # Apply volume and mix
                    outdata[:] += chunk * state.volume
                    state.playback_position += frames
            
            # Scratch sound - use actual track audio with speed variation
            with scratch_lock:
                if scratch_is_playing and scratch_track_index >= 0 and use_track_scratch:
                    # Scratching the actual track audio
                    state = track_states.get(scratch_track_index)
                    if state and scratch_track_buffer is not None:
                        # Calculate playback speed based on scratch speed
                        # Positive delta = forward (faster), negative = backward (slower/reverse)
                        playback_rate = 1.0 + scratch_speed * SPEED_MULTIPLIER
                        playback_rate = np.clip(playback_rate, PITCH_SHIFT_MIN, PITCH_SHIFT_MAX)
                        
                        # Calculate how many samples to read based on speed
                        samples_to_read = int(frames * playback_rate)
                        
                        # Get chunk from scratch buffer
                        start_pos = scratch_track_buffer_position
                        end_pos = start_pos + samples_to_read
                        
                        if end_pos <= len(scratch_track_buffer):
                            scratch_chunk = scratch_track_buffer[start_pos:end_pos]
                            
                            # Resample to fit output frames (this creates pitch shift)
                            if len(scratch_chunk) != frames:
                                # Simple linear interpolation for speed change
                                indices = np.linspace(0, len(scratch_chunk) - 1, frames)
                                scratch_chunk_resampled = np.zeros((frames, scratch_chunk.shape[1]))
                                for ch in range(scratch_chunk.shape[1]):
                                    scratch_chunk_resampled[:, ch] = np.interp(
                                        indices, 
                                        np.arange(len(scratch_chunk)), 
                                        scratch_chunk[:, ch]
                                    )
                                scratch_chunk = scratch_chunk_resampled
                            
                            # Apply volume (louder for faster scratches)
                            scratch_volume = min(
                                SCRATCH_MAX_VOLUME, 
                                SCRATCH_BASE_VOLUME + abs(scratch_speed) * SCRATCH_SPEED_VOLUME_FACTOR
                            )
                            outdata[:] += scratch_chunk * scratch_volume * state.volume
                            
                            scratch_track_buffer_position += samples_to_read
                        else:
                            # Loop the scratch buffer
                            scratch_track_buffer_position = 0
                
                # Fallback to scratch sample sound if not using track scratch
                elif scratch_is_playing and scratch_audio is not None and not use_track_scratch:
                    start_frame = scratch_playback_position
                    end_frame = start_frame + frames
                    
                    if end_frame <= len(scratch_audio):
                        scratch_chunk = scratch_audio[start_frame:end_frame]
                        scratch_volume = min(0.7, 0.3 + abs(scratch_speed) * 0.4)
                        outdata[:] += scratch_chunk * scratch_volume
                        scratch_playback_position += frames
                    else:
                        remaining = len(scratch_audio) - start_frame
                        if remaining > 0:
                            scratch_chunk = scratch_audio[start_frame:]
                            scratch_chunk = np.pad(scratch_chunk, ((0, frames - remaining), (0, 0)), mode='constant')
                            scratch_volume = min(0.7, 0.3 + abs(scratch_speed) * 0.4)
                            outdata[:] += scratch_chunk * scratch_volume
                        scratch_playback_position = 0

mixer = None
output_stream = None

# -----------------------------
# Load all mp3 files
# -----------------------------
def load_music_folder(folder_path):
    global songs, track_states, active_track, mixer, output_stream
    songs = [os.path.join(folder_path, f)
             for f in os.listdir(folder_path) if f.lower().endswith(".mp3")]
    songs.sort()
    track_states = {}
    active_track = -1

    # Use standard sample rate and stereo
    TARGET_SAMPLE_RATE = 44100
    TARGET_CHANNELS = 2

    print("Loading songs...")
    for i in range(len(songs)):
        try:
            print(f"Loading {os.path.basename(songs[i])}...")
            track_states[i] = TrackState(songs[i], TARGET_SAMPLE_RATE, TARGET_CHANNELS)
        except Exception as e:
            print(f"Error loading {songs[i]}: {e}")
            import traceback
            traceback.print_exc()

    if not songs:
        raise ValueError("No mp3 files found")
    
    # Load scratch sound effect
    load_scratch_sound(TARGET_SAMPLE_RATE, TARGET_CHANNELS)
    
    # Initialize the mixer with standard settings
    print(f"Initializing audio mixer at {TARGET_SAMPLE_RATE}Hz, {TARGET_CHANNELS} channels...")
    mixer = AudioMixer(TARGET_SAMPLE_RATE)
    output_stream = sd.OutputStream(
        samplerate=TARGET_SAMPLE_RATE,
        channels=TARGET_CHANNELS,
        callback=mixer.callback,
        blocksize=2048
    )
    output_stream.start()
    
    print(f"Loaded {len(songs)} songs successfully!")

# -----------------------------
# Update all playing tracks
# -----------------------------
def update_active_track_position():
    """Update position for ALL playing tracks"""
    now = time.time()
    for index, state in track_states.items():
        if state.is_playing and not state.is_scrubbing:
            if state.last_update_time is not None:
                elapsed = now - state.last_update_time
                state.position += elapsed
                
                # Check if track finished
                if state.position >= state.duration:
                    state.position = state.duration
                    state.is_playing = False
                    
            state.last_update_time = now

# -----------------------------
# Volume Control
# -----------------------------
def set_volume(index, volume):
    if index < 0 or index >= len(songs):
        return
    volume = max(0.0, min(1.0, volume))
    track_states[index].volume = volume

def get_volume(index):
    if index < 0 or index >= len(songs):
        return 0.0
    return track_states[index].volume

# -----------------------------
# Get position
# -----------------------------
def get_position(index):
    if index < 0 or index >= len(songs):
        return 0.0
    return track_states[index].position

# -----------------------------
# Play / Pause
# -----------------------------
def toggle_play(index):
    global active_track
    if index < 0 or index >= len(songs):
        return

    state = track_states[index]

    with mixer.lock:
        if state.is_playing:
            # Pause this track
            state.is_playing = False
            state.last_update_time = None
            
            if active_track == index:
                active_track = -1
                for i, s in track_states.items():
                    if s.is_playing:
                        active_track = i
                        break
        else:
            # Play/Resume this track from current position
            if state.position >= state.duration:
                state.position = 0.0
                state.playback_position = 0
            else:
                state.playback_position = int(state.position * state.sample_rate)
                
            state.is_playing = True
            state.last_update_time = time.time()
            active_track = index

def stop(index):
    global active_track
    state = track_states.get(index)
    if state:
        with mixer.lock:
            state.is_playing = False
            state.last_update_time = None
            state.position = 0.0
            state.playback_position = 0
            
            if active_track == index:
                active_track = -1
                for i, s in track_states.items():
                    if s.is_playing:
                        active_track = i
                        break

# -----------------------------
# Scratch Sound Effects
# -----------------------------
def prepare_track_scratch_buffer(index, buffer_duration=None):
    """Prepare a buffer of the track audio for scratching"""
    global scratch_track_buffer, scratch_track_index, scratch_track_buffer_position
    
    if buffer_duration is None:
        buffer_duration = SCRATCH_BUFFER_DURATION
    
    if index < 0 or index >= len(songs):
        return
    
    state = track_states[index]
    
    # Get a chunk of audio around current position for scratching
    center_position = int(state.position * state.sample_rate)
    buffer_samples = int(buffer_duration * state.sample_rate)
    
    start_pos = max(0, center_position - buffer_samples // 2)
    end_pos = min(len(state.audio_data), start_pos + buffer_samples)
    
    scratch_track_buffer = state.audio_data[start_pos:end_pos].copy()
    scratch_track_buffer_position = buffer_samples // 2  # Start in middle of buffer
    scratch_track_index = index

def play_scratch_effect(delta, track_index):
    """Play or update the scratch sound effect"""
    global scratch_is_playing, scratch_playback_position, scratch_speed, scratch_direction
    
    with scratch_lock:
        scratch_speed = delta  # Preserve sign for direction
        scratch_direction = 1 if delta >= 0 else -1
        
        if not scratch_is_playing:
            scratch_is_playing = True
            scratch_playback_position = 0
            
            # Prepare track buffer for realistic scratching
            if use_track_scratch and track_index >= 0:
                prepare_track_scratch_buffer(track_index)

def stop_scratch_effect():
    """Stop the scratch sound effect"""
    global scratch_is_playing, scratch_playback_position, scratch_track_buffer
    global scratch_track_index, scratch_track_buffer_position
    
    with scratch_lock:
        scratch_is_playing = False
        scratch_playback_position = 0
        scratch_track_buffer = None
        scratch_track_buffer_position = 0
        scratch_track_index = -1

# -----------------------------
# Jog Wheel Scrub
# -----------------------------
def scrub(delta, index):
    if index < 0 or index >= len(songs):
        return

    state = track_states[index]

    # CRITICAL: Set scrubbing flag FIRST, before anything else
    if not state.is_scrubbing:
        state.is_scrubbing = True  # Set this IMMEDIATELY
        state.was_playing_before_scrub = state.is_playing
        
        # Force stop playback immediately in mixer
        with mixer.lock:
            state.is_playing = False

    # Update position
    state.position += delta * SCRATCH_SENSITIVITY
    state.position = max(0, min(state.position, state.duration))
    state.playback_position = int(state.position * state.sample_rate)
    state.last_update_time = time.time()
    
    # Play scratch sound effect with track index
    play_scratch_effect(delta, index)

def end_scrub(index):
    global active_track
    if index < 0 or index >= len(songs):
        return

    state = track_states[index]

    if state.is_scrubbing:
        state.is_scrubbing = False
        
        # Stop scratch sound
        stop_scratch_effect()
        
        if state.was_playing_before_scrub:
            with mixer.lock:
                state.is_playing = True
                state.last_update_time = time.time()
                active_track = index
        
        state.was_playing_before_scrub = False

# -----------------------------
# Helpers
# -----------------------------
def is_playing(index=None):
    if index is None:
        index = active_track
    if index < 0 or index >= len(songs):
        return False
    return track_states[index].is_playing

def get_current_song_name(index):
    if index < 0 or index >= len(songs):
        return None
    return os.path.basename(songs[index])

def get_active_track():
    return active_track