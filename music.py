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
        self.is_playing = False
        self.was_playing_before_scrub = False
        self.volume = 1.0  # 🔊 DEFAULT MAX VOLUME

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
# Volume Control (NEW)
# -----------------------------
def set_volume(index, volume):
    if index < 0 or index >= len(songs):
        return

    volume = max(0.0, min(1.0, volume))
    track_states[index].volume = volume

    # Apply immediately if this track is active
    if active_track == index:
        pygame.mixer.music.set_volume(volume)

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

    if state.is_playing:
        pygame.mixer.music.stop()
        state.is_playing = False
        state.last_update_time = None
        if active_track == index:
            active_track = -1
    else:
        # Stop other track
        if active_track >= 0 and active_track != index:
            other = track_states[active_track]
            pygame.mixer.music.stop()
            other.is_playing = False
            other.last_update_time = None

        pygame.mixer.music.load(songs[index])
        pygame.mixer.music.set_volume(state.volume)  # 🔊 APPLY STORED VOLUME
        pygame.mixer.music.play(start=state.position)

        state.is_playing = True
        state.last_update_time = time.time()
        active_track = index

def stop(index):
    global active_track
    state = track_states.get(index)
    if state:
        pygame.mixer.music.stop()
        state.is_playing = False
        state.last_update_time = None
        state.position = 0.0
        if active_track == index:
            active_track = -1

# -----------------------------
# Jog Wheel Scrub
# -----------------------------
def scrub(delta, index):
    if index < 0 or index >= len(songs):
        return

    state = track_states[index]

    if not state.is_scrubbing:
        state.was_playing_before_scrub = state.is_playing
        if state.is_playing:
            pygame.mixer.music.stop()
            state.is_playing = False

    state.is_scrubbing = True
    state.position += delta * 0.5
    state.position = max(0, state.position)
    state.last_update_time = time.time()

def end_scrub(index):
    global active_track
    if index < 0 or index >= len(songs):
        return

    state = track_states[index]

    if state.is_scrubbing:
        state.is_scrubbing = False
        if state.was_playing_before_scrub:
            pygame.mixer.music.load(songs[index])
            pygame.mixer.music.set_volume(state.volume)  # 🔊 APPLY VOLUME
            pygame.mixer.music.play(start=state.position)
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
