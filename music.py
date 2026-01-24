import pygame
import os
import random

# -----------------------------
# Audio Init
# -----------------------------
pygame.mixer.init()

songs = []
current_index = -1

# -----------------------------
# Load all mp3 files from folder
# -----------------------------
def load_music_folder(folder_path):
    global songs, current_index

    songs = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.lower().endswith(".mp3")
    ]

    songs.sort()
    current_index = -1

    if not songs:
        raise ValueError("No mp3 files found in music folder")

# -----------------------------
# Play controls
# -----------------------------
def play_index(index):
    global current_index

    if index < 0 or index >= len(songs):
        return

    if current_index != index:
        pygame.mixer.music.load(songs[index])
        pygame.mixer.music.play()
        current_index = index

def play_random():
    if not songs:
        return

    index = random.randint(0, len(songs) - 1)
    play_index(index)

def play_next():
    global current_index
    if not songs:
        return

    next_index = (current_index + 1) % len(songs)
    play_index(next_index)

def play_previous():
    global current_index
    if not songs:
        return

    prev_index = (current_index - 1) % len(songs)
    play_index(prev_index)

def stop():
    pygame.mixer.music.stop()

def is_playing():
    return pygame.mixer.music.get_busy()

def get_current_song_name():
    """Returns the name of the currently playing song"""
    if current_index >= 0 and current_index < len(songs):
        return os.path.basename(songs[current_index])
    return None