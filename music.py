import os
import time
import sounddevice as sd
import soundfile as sf
import numpy as np
import threading
import scipy.signal as signal

# -----------------------------
# Track State Management
# -----------------------------
songs = []
active_track = -1  # Currently active track (for display purposes)

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
# Audio Mixer
# -----------------------------
class AudioMixer:
    def __init__(self, sample_rate=44100):
        self.sample_rate = sample_rate
        self.streams = {}
        self.lock = threading.Lock()
        
    def callback(self, outdata, frames, time_info, status):
        """Mix all active tracks"""
        outdata.fill(0)
        with self.lock:
            for index, state in track_states.items():
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
# Jog Wheel Scrub
# -----------------------------
def scrub(delta, index):
    if index < 0 or index >= len(songs):
        return

    state = track_states[index]

    if not state.is_scrubbing:
        state.was_playing_before_scrub = state.is_playing
        with mixer.lock:
            if state.is_playing:
                state.is_playing = False

    state.is_scrubbing = True
    state.position += delta * 0.5
    state.position = max(0, min(state.position, state.duration))
    state.playback_position = int(state.position * state.sample_rate)
    state.last_update_time = time.time()

def end_scrub(index):
    global active_track
    if index < 0 or index >= len(songs):
        return

    state = track_states[index]

    if state.is_scrubbing:
        state.is_scrubbing = False
        
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