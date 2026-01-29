import os
import time
import sounddevice as sd
import soundfile as sf
import numpy as np
import threading
import scipy.signal as signal
from pathlib import Path
from auto_stems import get_auto_stem_manager

# Import scratch configuration
try:
    from scratch_config import *
except ImportError:
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
active_track = -1

# Scratch sound state
scratch_audio = None
scratch_sample_rate = None
scratch_playback_position = 0
scratch_is_playing = False
scratch_speed = 0.0
scratch_direction = 1
scratch_lock = threading.Lock()

# Track scratching state
scratch_track_index = -1
scratch_track_buffer = None
scratch_track_buffer_position = 0
use_track_scratch = USE_TRACK_SCRATCH

# Stem manager
stem_manager = None

# Recorder reference (will be set by hands.py)
audio_recorder = None

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
                print(f"  Converting mono to stereo...")
                self.audio_data = np.repeat(self.audio_data, 2, axis=1)
            elif current_channels == 2 and target_channels == 1:
                print(f"  Converting stereo to mono...")
                self.audio_data = np.mean(self.audio_data, axis=1, keepdims=True)
        
        self.position = 0.0
        self.last_update_time = None
        self.is_scrubbing = False
        self.is_playing = False
        self.was_playing_before_scrub = False
        self.volume = 1.0
        self.stream = None
        self.duration = len(self.audio_data) / self.sample_rate
        self.playback_position = 0
        self.lock = threading.Lock()
        
        # Stem state
        self.stems = {}  # stem_type -> audio_data
        self.stem_enabled = {}  # stem_type -> bool
        self.has_stems = False
        
        print(f"  Duration: {self.duration:.2f}s, Sample rate: {self.sample_rate}Hz, Channels: {self.audio_data.shape[1]}")
    
    def load_stem(self, stem_type, stem_path, target_sample_rate=44100, target_channels=2):
        """Load a stem file for this track"""
        try:
            audio_data, sample_rate = sf.read(stem_path, dtype='float32')
            
            if len(audio_data.shape) == 1:
                audio_data = audio_data.reshape(-1, 1)
            
            if sample_rate != target_sample_rate:
                num_samples = int(len(audio_data) * target_sample_rate / sample_rate)
                audio_data = signal.resample(audio_data, num_samples)
            
            current_channels = audio_data.shape[1]
            if current_channels != target_channels:
                if current_channels == 1 and target_channels == 2:
                    audio_data = np.repeat(audio_data, 2, axis=1)
                elif current_channels == 2 and target_channels == 1:
                    audio_data = np.mean(audio_data, axis=1, keepdims=True)
            
            self.stems[stem_type] = audio_data
            self.stem_enabled[stem_type] = False  # Start disabled
            self.has_stems = True
            print(f"    ✓ Loaded {stem_type} stem")
            return True
        except Exception as e:
            print(f"    ✗ Failed to load {stem_type} stem: {e}")
            return False
    
    def get_mixed_audio(self, start_frame, num_frames):
        """Get audio with stem mixing applied"""
        if not self.has_stems or not any(self.stem_enabled.values()):
            # No stems or all disabled - use original audio
            end_frame = min(start_frame + num_frames, len(self.audio_data))
            if start_frame >= len(self.audio_data):
                return np.zeros((num_frames, self.audio_data.shape[1]))
            chunk = self.audio_data[start_frame:end_frame]
            if len(chunk) < num_frames:
                chunk = np.pad(chunk, ((0, num_frames - len(chunk)), (0, 0)), mode='constant')
            return chunk
        
        # Mix enabled stems
        mixed = np.zeros((num_frames, self.audio_data.shape[1]))
        
        for stem_type, enabled in self.stem_enabled.items():
            if enabled and stem_type in self.stems:
                stem_data = self.stems[stem_type]
                end_frame = min(start_frame + num_frames, len(stem_data))
                if start_frame < len(stem_data):
                    chunk = stem_data[start_frame:end_frame]
                    if len(chunk) < num_frames:
                        chunk = np.pad(chunk, ((0, num_frames - len(chunk)), (0, 0)), mode='constant')
                    mixed += chunk
        
        return mixed

track_states = {}
mixer_streams = []

# -----------------------------
# Load Scratch Sound
# -----------------------------
def load_scratch_sound(target_sample_rate=44100, target_channels=2):
    global scratch_audio, scratch_sample_rate
    
    scratch_paths = [
        "scratch.wav", "scratch.mp3",
        "sounds/scratch.wav", "sounds/scratch.mp3",
        "assets/scratch.wav", "assets/scratch.mp3"
    ]
    
    for path in scratch_paths:
        if os.path.exists(path):
            try:
                print(f"Loading scratch sound from {path}...")
                audio_data, sample_rate = sf.read(path, dtype='float32')
                
                if len(audio_data.shape) == 1:
                    audio_data = audio_data.reshape(-1, 1)
                
                if sample_rate != target_sample_rate:
                    num_samples = int(len(audio_data) * target_sample_rate / sample_rate)
                    audio_data = signal.resample(audio_data, num_samples)
                    sample_rate = target_sample_rate
                
                current_channels = audio_data.shape[1]
                if current_channels == 1 and target_channels == 2:
                    audio_data = np.repeat(audio_data, 2, axis=1)
                elif current_channels == 2 and target_channels == 1:
                    audio_data = np.mean(audio_data, axis=1, keepdims=True)
                
                scratch_audio = audio_data
                scratch_sample_rate = sample_rate
                print(f"  ✓ Loaded scratch sound")
                return True
            except Exception as e:
                print(f"  Error loading {path}: {e}")
    
    print("⚠ Warning: No scratch sound file found")
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
        global scratch_playback_position, scratch_is_playing
        global scratch_track_buffer, scratch_track_buffer_position, scratch_track_index
        
        outdata.fill(0)
        
        with self.lock:
            # Mix music tracks with stems
            for index, state in track_states.items():
                if state.is_playing and not state.is_scrubbing:
                    start_frame = state.playback_position
                    
                    # Get mixed audio (with stems if enabled)
                    chunk = state.get_mixed_audio(start_frame, frames)
                    
                    if len(chunk) > 0:
                        outdata[:len(chunk)] += chunk * state.volume
                        state.playback_position += len(chunk)
                        
                        if state.playback_position >= len(state.audio_data):
                            state.is_playing = False
            
            # Scratch sound handling
            with scratch_lock:
                if scratch_is_playing and scratch_track_index >= 0 and use_track_scratch:
                    state = track_states.get(scratch_track_index)
                    if state and scratch_track_buffer is not None:
                        playback_rate = 1.0 + scratch_speed * SPEED_MULTIPLIER
                        playback_rate = np.clip(playback_rate, PITCH_SHIFT_MIN, PITCH_SHIFT_MAX)
                        
                        samples_to_read = int(frames * playback_rate)
                        start_pos = scratch_track_buffer_position
                        end_pos = start_pos + samples_to_read
                        
                        if end_pos <= len(scratch_track_buffer):
                            scratch_chunk = scratch_track_buffer[start_pos:end_pos]
                            
                            if len(scratch_chunk) != frames:
                                indices = np.linspace(0, len(scratch_chunk) - 1, frames)
                                scratch_chunk_resampled = np.zeros((frames, scratch_chunk.shape[1]))
                                for ch in range(scratch_chunk.shape[1]):
                                    scratch_chunk_resampled[:, ch] = np.interp(
                                        indices, 
                                        np.arange(len(scratch_chunk)), 
                                        scratch_chunk[:, ch]
                                    )
                                scratch_chunk = scratch_chunk_resampled
                            
                            scratch_volume = min(
                                SCRATCH_MAX_VOLUME, 
                                SCRATCH_BASE_VOLUME + abs(scratch_speed) * SCRATCH_SPEED_VOLUME_FACTOR
                            )
                            outdata[:] += scratch_chunk * scratch_volume * state.volume
                            scratch_track_buffer_position += samples_to_read
                        else:
                            scratch_track_buffer_position = 0
                
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
            
            # Send audio to recorder if recording
            global audio_recorder
            if audio_recorder is not None:
                audio_recorder.add_audio_chunk(outdata, self.sample_rate)

mixer = None
output_stream = None

# -----------------------------
# Load Music with Stems
# -----------------------------
def load_music_folder(folder_path):
    global songs, track_states, active_track, mixer, output_stream, stem_manager
    
    folder_path = Path(folder_path)
    
    # Find all MP3 files (excluding stem files)
    all_files = list(folder_path.glob("*.mp3"))
    songs = [str(f) for f in all_files if not any(
        stem_marker in f.stem.lower() 
        for stem_marker in ['_vocals', '_instrumental', '_drums', '_bass', '_other']
    )]
    songs.sort()
    
    track_states = {}
    active_track = -1

    TARGET_SAMPLE_RATE = 44100
    TARGET_CHANNELS = 2

    # Initialize stem manager and generate stems
    stem_manager = get_auto_stem_manager(str(folder_path))
    
    print("\n🎵 Generating stems (this may take a few minutes)...")
    stem_manager.process_all_songs(songs, full_separation=True)
    
    print("\n🎵 Loading songs...")
    for i, song_path in enumerate(songs):
        try:
            print(f"Loading {os.path.basename(song_path)}...")
            track_states[i] = TrackState(song_path, TARGET_SAMPLE_RATE, TARGET_CHANNELS)
            
            # Load stems if available
            song_stem = Path(song_path).stem
            stem_files = {
                'vocals': folder_path / f"{song_stem}_vocals.mp3",
                'instrumental': folder_path / f"{song_stem}_instrumental.mp3",
                'drums': folder_path / f"{song_stem}_drums.mp3",
                'bass': folder_path / f"{song_stem}_bass.mp3",
                'other': folder_path / f"{song_stem}_other.mp3"
            }
            
            for stem_type, stem_path in stem_files.items():
                if stem_path.exists():
                    track_states[i].load_stem(stem_type, str(stem_path), TARGET_SAMPLE_RATE, TARGET_CHANNELS)
                    
        except Exception as e:
            print(f"Error loading {song_path}: {e}")
            import traceback
            traceback.print_exc()

    if not songs:
        raise ValueError("No mp3 files found")
    
    load_scratch_sound(TARGET_SAMPLE_RATE, TARGET_CHANNELS)
    
    print(f"\n🎵 Initializing audio mixer...")
    mixer = AudioMixer(TARGET_SAMPLE_RATE)
    output_stream = sd.OutputStream(
        samplerate=TARGET_SAMPLE_RATE,
        channels=TARGET_CHANNELS,
        callback=mixer.callback,
        blocksize=2048
    )
    output_stream.start()
    
    print(f"✅ Loaded {len(songs)} songs successfully!\n")

# -----------------------------
# Stem Control Functions
# -----------------------------
def toggle_stem(index, stem_type):
    """Toggle a stem on/off"""
    if index < 0 or index >= len(songs):
        return False
    
    state = track_states[index]
    if stem_type not in state.stems:
        return False
    
    state.stem_enabled[stem_type] = not state.stem_enabled[stem_type]
    return state.stem_enabled[stem_type]

def get_stem_state(index, stem_type):
    """Check if a stem is enabled"""
    if index < 0 or index >= len(songs):
        return False
    
    state = track_states[index]
    return state.stem_enabled.get(stem_type, False)

def get_available_stems(index):
    """Get list of available stems for a track"""
    if index < 0 or index >= len(songs):
        return []
    
    state = track_states[index]
    return list(state.stems.keys())

def has_stems(index):
    """Check if track has any stems loaded"""
    if index < 0 or index >= len(songs):
        return False
    
    return track_states[index].has_stems

# -----------------------------
# Existing Functions (unchanged)
# -----------------------------
def update_active_track_position():
    now = time.time()
    for index, state in track_states.items():
        if state.is_playing and not state.is_scrubbing:
            if state.last_update_time is not None:
                elapsed = now - state.last_update_time
                state.position += elapsed
                
                if state.position >= state.duration:
                    state.position = state.duration
                    state.is_playing = False
                    
            state.last_update_time = now

def set_volume(index, volume):
    if index < 0 or index >= len(songs):
        return
    volume = max(0.0, min(1.0, volume))
    track_states[index].volume = volume

def get_volume(index):
    if index < 0 or index >= len(songs):
        return 0.0
    return track_states[index].volume

def get_position(index):
    if index < 0 or index >= len(songs):
        return 0.0
    return track_states[index].position

def toggle_play(index):
    global active_track
    if index < 0 or index >= len(songs):
        return

    state = track_states[index]

    with mixer.lock:
        if state.is_playing:
            state.is_playing = False
            state.last_update_time = None
            
            if active_track == index:
                active_track = -1
                for i, s in track_states.items():
                    if s.is_playing:
                        active_track = i
                        break
        else:
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

def prepare_track_scratch_buffer(index, buffer_duration=None):
    global scratch_track_buffer, scratch_track_index, scratch_track_buffer_position
    
    if buffer_duration is None:
        buffer_duration = SCRATCH_BUFFER_DURATION
    
    if index < 0 or index >= len(songs):
        return
    
    state = track_states[index]
    center_position = int(state.position * state.sample_rate)
    buffer_samples = int(buffer_duration * state.sample_rate)
    
    start_pos = max(0, center_position - buffer_samples // 2)
    end_pos = min(len(state.audio_data), start_pos + buffer_samples)
    
    scratch_track_buffer = state.audio_data[start_pos:end_pos].copy()
    scratch_track_buffer_position = buffer_samples // 2
    scratch_track_index = index

def play_scratch_effect(delta, track_index):
    global scratch_is_playing, scratch_playback_position, scratch_speed, scratch_direction
    
    with scratch_lock:
        scratch_speed = delta
        scratch_direction = 1 if delta >= 0 else -1
        
        if not scratch_is_playing:
            scratch_is_playing = True
            scratch_playback_position = 0
            
            if use_track_scratch and track_index >= 0:
                prepare_track_scratch_buffer(track_index)

def stop_scratch_effect():
    global scratch_is_playing, scratch_playback_position, scratch_track_buffer
    global scratch_track_index, scratch_track_buffer_position
    
    with scratch_lock:
        scratch_is_playing = False
        scratch_playback_position = 0
        scratch_track_buffer = None
        scratch_track_buffer_position = 0
        scratch_track_index = -1

def scrub(delta, index):
    if index < 0 or index >= len(songs):
        return

    state = track_states[index]

    if not state.is_scrubbing:
        state.is_scrubbing = True
        state.was_playing_before_scrub = state.is_playing
        
        with mixer.lock:
            state.is_playing = False

    state.position += delta * SCRATCH_SENSITIVITY
    state.position = max(0, min(state.position, state.duration))
    state.playback_position = int(state.position * state.sample_rate)
    state.last_update_time = time.time()
    
    play_scratch_effect(delta, index)

def end_scrub(index):
    global active_track
    if index < 0 or index >= len(songs):
        return

    state = track_states[index]

    if state.is_scrubbing:
        state.is_scrubbing = False
        stop_scratch_effect()
        
        if state.was_playing_before_scrub:
            with mixer.lock:
                state.is_playing = True
                state.last_update_time = time.time()
                active_track = index
        
        state.was_playing_before_scrub = False

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