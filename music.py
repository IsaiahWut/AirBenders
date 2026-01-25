import pygame
import os
import time

# -----------------------------
# Audio Init
# -----------------------------
pygame.mixer.init()

songs = []
active_track = -1  # Currently active track

# Track state for each song independently
class TrackState:
    def __init__(self):
        self.position = 0.0
        self.last_update_time = None
        self.is_scrubbing = False
        self.is_playing = False  # True if track is currently playing

track_states = {}  # index -> TrackState

# -----------------------------
# Load all mp3 files
# -----------------------------
def load_music_folder(folder_path):
    global songs, track_states, active_track
    songs = [os.path.join(folder_path, f)
             for f in os.listdir(folder_path) if f.lower().endswith(".mp3")]
    songs.sort()
    track_states = {}
    active_track = -1
    for i in range(len(songs)):
        track_states[i] = TrackState()
    if not songs:
        raise ValueError("No mp3 files found")

# -----------------------------
# Update active track time
# -----------------------------
def update_active_track_position():
    """Call every frame to update currently playing track's position"""
    global active_track
    if active_track < 0:
        return
    state = track_states[active_track]
    if state.is_playing and not state.is_scrubbing:
        now = time.time()
        if state.last_update_time is not None:
            state.position += now - state.last_update_time
        state.last_update_time = now

# -----------------------------
# Get position (READ ONLY)
# -----------------------------
def get_position(index):
    if index < 0 or index >= len(songs):
        return 0.0
    return track_states[index].position

# -----------------------------
# Play/Pause controls (PlayButton)
# -----------------------------
def toggle_play(index):
    """Toggle play/pause for the given track"""
    global active_track
    if index < 0 or index >= len(songs):
        return
    state = track_states[index]

    if state.is_playing:
        # Pause
        pygame.mixer.music.stop()
        state.is_playing = False
        state.last_update_time = None
    else:
        # Play from saved position
        pygame.mixer.music.load(songs[index])
        pygame.mixer.music.play(start=state.position)
        state.is_playing = True
        state.last_update_time = time.time()
        active_track = index

def stop(index):
    """Stop a track completely"""
    state = track_states.get(index)
    if state:
        pygame.mixer.music.stop()
        state.is_playing = False
        state.last_update_time = None
        state.position = 0.0

# -----------------------------
# Jog Wheel Scrub (fast forward / rewind)
# -----------------------------
def scrub(delta, index):
    if index < 0 or index >= len(songs):
        return
    state = track_states[index]

    # Pause the track while scrubbing
    if state.is_playing and not state.is_scrubbing:
        pygame.mixer.music.stop()

    state.is_scrubbing = True
    state.position += delta * 0.5  # sensitivity
    state.position = max(0, state.position)
    state.last_update_time = time.time()

def end_scrub(index):
    """End scrubbing and optionally resume playback"""
    if index < 0 or index >= len(songs):
        return
    state = track_states[index]
    
    if state.is_scrubbing:
        state.is_scrubbing = False
        # If the track was playing before scrubbing, resume it
        if active_track == index:
            pygame.mixer.music.load(songs[index])
            pygame.mixer.music.play(start=state.position)
            state.is_playing = True
            state.last_update_time = time.time()

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