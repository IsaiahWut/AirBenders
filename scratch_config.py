# DJ Scratch Configuration
# Edit these settings to customize your scratch behavior

# SCRATCH MODE
# True = Scratch the actual track audio (more realistic, pitch shifts with speed)
# False = Use pre-recorded scratch sound samples
USE_TRACK_SCRATCH = True

# SCRATCH SENSITIVITY
# How much the jog wheel movement affects position
# Lower = more control, Higher = faster seeking
SCRATCH_SENSITIVITY = 0.5

# PITCH SHIFT RANGE
# How much the pitch changes when scratching
# Format: (min_rate, max_rate)
# 1.0 = normal speed, 0.5 = half speed, 2.0 = double speed
PITCH_SHIFT_MIN = 0.3
PITCH_SHIFT_MAX = 3.0

# SCRATCH BUFFER DURATION
# How much audio to buffer for scratching (in seconds)
# Longer = can scratch further, but uses more memory
SCRATCH_BUFFER_DURATION = 2.0

# VOLUME SETTINGS
# Base volume for scratch (0.0 to 1.0)
SCRATCH_BASE_VOLUME = 0.4
# Maximum scratch volume (0.0 to 1.0)
SCRATCH_MAX_VOLUME = 0.8
# How much scratch speed affects volume (0.0 to 1.0)
SCRATCH_SPEED_VOLUME_FACTOR = 0.4

# SPEED MULTIPLIER
# How much delta (jog movement) affects playback speed
# Higher = more dramatic pitch changes
SPEED_MULTIPLIER = 2.0