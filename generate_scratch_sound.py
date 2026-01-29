"""
Generate realistic vinyl scratch sound effects for DJ scratching
Creates multiple scratch samples with different characteristics
Requires: pip install numpy scipy
"""

import numpy as np
from scipy.io import wavfile
import os

def generate_vinyl_noise(duration, sample_rate=44100):
    """Generate realistic vinyl surface noise"""
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Multiple layers of noise for texture
    # High frequency crackle
    crackle = np.random.normal(0, 0.03, len(t))
    crackle = np.convolve(crackle, np.ones(3)/3, mode='same')  # Smooth slightly
    
    # Mid frequency texture
    texture = np.random.normal(0, 0.02, len(t))
    
    # Low frequency rumble
    rumble_freq = 30 + np.random.randn(len(t)) * 5
    rumble = 0.01 * np.sin(2 * np.pi * np.cumsum(rumble_freq) / sample_rate)
    
    return crackle + texture + rumble

def generate_chirp_scratch(duration=0.3, sample_rate=44100, direction=1):
    """
    Generate a realistic chirp-style scratch
    direction: 1 for forward, -1 for backward
    """
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Frequency sweep - this is the core scratch sound
    if direction > 0:
        # Forward scratch: low to high
        f_start = 150
        f_end = 2500
    else:
        # Backward scratch: high to low
        f_start = 2500
        f_end = 150
    
    # Exponential frequency sweep for more natural sound
    frequency = np.logspace(np.log10(f_start), np.log10(f_end), len(t))
    
    # Add some wobble to the frequency (hand movement isn't perfectly smooth)
    wobble = np.sin(2 * np.pi * 8 * t) * 50 + np.sin(2 * np.pi * 15 * t) * 30
    frequency = frequency + wobble
    
    # Generate the main tone
    phase = 2 * np.pi * np.cumsum(frequency) / sample_rate
    scratch = np.sin(phase)
    
    # Add harmonics for richer sound
    scratch += 0.3 * np.sin(2 * phase)  # 2nd harmonic
    scratch += 0.15 * np.sin(3 * phase)  # 3rd harmonic
    scratch += 0.08 * np.sin(4 * phase)  # 4th harmonic
    
    # Add vinyl surface noise
    vinyl_noise = generate_vinyl_noise(duration, sample_rate)
    scratch = scratch + vinyl_noise
    
    # Create realistic envelope
    # Quick attack, sustained, then quick release
    envelope = np.ones_like(t)
    
    # Attack (first 10%)
    attack_samples = int(0.1 * len(t))
    envelope[:attack_samples] = np.linspace(0, 1, attack_samples) ** 0.5
    
    # Release (last 15%)
    release_samples = int(0.15 * len(t))
    envelope[-release_samples:] = np.linspace(1, 0, release_samples) ** 2
    
    scratch = scratch * envelope
    
    # Add some saturation/distortion for warmth
    scratch = np.tanh(scratch * 1.8) * 0.7
    
    # Normalize
    scratch = scratch / (np.max(np.abs(scratch)) + 1e-6)
    
    return scratch

def generate_baby_scratch(duration=0.2, sample_rate=44100):
    """Generate a quick baby scratch (forward-back motion)"""
    # Short forward and backward scratch combined
    half_duration = duration / 2
    
    forward = generate_chirp_scratch(half_duration, sample_rate, direction=1)
    backward = generate_chirp_scratch(half_duration, sample_rate, direction=-1)
    
    # Combine with slight overlap
    scratch = np.concatenate([forward, backward])
    
    return scratch

def generate_transformer_scratch(duration=0.25, sample_rate=44100):
    """Generate a transformer/cutting scratch effect"""
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Rapid on-off pattern
    num_cuts = 6
    scratch = np.zeros(len(t))
    
    cut_length = len(t) // num_cuts
    for i in range(num_cuts):
        if i % 2 == 0:  # On beats
            start = i * cut_length
            end = min(start + cut_length, len(t))
            chunk_t = t[start:end] - t[start]
            
            # Quick frequency chirp for each cut
            freq = 400 + 600 * (i / num_cuts)
            chunk = np.sin(2 * np.pi * freq * chunk_t)
            
            # Sharp envelope
            env = np.ones_like(chunk)
            fade = min(20, len(chunk) // 4)
            env[:fade] = np.linspace(0, 1, fade)
            env[-fade:] = np.linspace(1, 0, fade)
            
            scratch[start:end] = chunk * env * 0.8
    
    # Add vinyl noise
    scratch = scratch + generate_vinyl_noise(duration, sample_rate) * 0.5
    
    # Normalize
    scratch = scratch / (np.max(np.abs(scratch)) + 1e-6)
    
    return scratch

def create_stereo(mono_scratch):
    """Convert mono to stereo with slight width"""
    # Add tiny delay between channels for stereo width
    stereo_left = mono_scratch
    stereo_right = np.roll(mono_scratch, 2)  # 2 sample delay
    stereo = np.column_stack((stereo_left, stereo_right))
    return stereo

def normalize_audio(audio, target_level=-3):
    """Normalize audio to target dB level"""
    current_peak = np.max(np.abs(audio))
    if current_peak > 0:
        target_amplitude = 10 ** (target_level / 20)
        audio = audio * (target_amplitude / current_peak)
    return audio

if __name__ == "__main__":
    print("Generating realistic vinyl scratch sound effects...")
    print("=" * 60)
    
    sample_rate = 44100
    
    # Generate main scratch sound (chirp)
    print("\n1. Creating chirp scratch (forward)...")
    scratch_forward = generate_chirp_scratch(duration=0.3, sample_rate=sample_rate, direction=1)
    scratch_forward = normalize_audio(scratch_forward, -3)
    scratch_forward_stereo = create_stereo(scratch_forward)
    scratch_forward_int = np.int16(scratch_forward_stereo * 32767)
    wavfile.write("scratch.wav", sample_rate, scratch_forward_int)
    print(f"   ✓ Created scratch.wav ({len(scratch_forward)} samples)")
    
    # Generate backward scratch
    print("\n2. Creating chirp scratch (backward)...")
    scratch_backward = generate_chirp_scratch(duration=0.3, sample_rate=sample_rate, direction=-1)
    scratch_backward = normalize_audio(scratch_backward, -3)
    scratch_backward_stereo = create_stereo(scratch_backward)
    scratch_backward_int = np.int16(scratch_backward_stereo * 32767)
    wavfile.write("scratch_back.wav", sample_rate, scratch_backward_int)
    print(f"   ✓ Created scratch_back.wav ({len(scratch_backward)} samples)")
    
    # Generate baby scratch
    print("\n3. Creating baby scratch...")
    baby = generate_baby_scratch(duration=0.25, sample_rate=sample_rate)
    baby = normalize_audio(baby, -3)
    baby_stereo = create_stereo(baby)
    baby_int = np.int16(baby_stereo * 32767)
    wavfile.write("scratch_baby.wav", sample_rate, baby_int)
    print(f"   ✓ Created scratch_baby.wav ({len(baby)} samples)")
    
    # Generate transformer scratch
    print("\n4. Creating transformer scratch...")
    transformer = generate_transformer_scratch(duration=0.25, sample_rate=sample_rate)
    transformer = normalize_audio(transformer, -3)
    transformer_stereo = create_stereo(transformer)
    transformer_int = np.int16(transformer_stereo * 32767)
    wavfile.write("scratch_transformer.wav", sample_rate, transformer_int)
    print(f"   ✓ Created scratch_transformer.wav ({len(transformer)} samples)")
    
    print("\n" + "=" * 60)
    print("SUMMARY:")
    print("  ✓ scratch.wav - Main forward chirp scratch")
    print("  ✓ scratch_back.wav - Backward chirp scratch")
    print("  ✓ scratch_baby.wav - Quick baby scratch")
    print("  ✓ scratch_transformer.wav - Cutting/transformer scratch")
    print("\nAll files created at 44100 Hz, 16-bit stereo")
    print("\nTo use: The system will automatically use scratch.wav")
    print("You can swap different scratch types by renaming them to scratch.wav")
    
    total_size = sum([
        os.path.getsize(f) for f in [
            "scratch.wav", 
            "scratch_back.wav", 
            "scratch_baby.wav", 
            "scratch_transformer.wav"
        ]
    ])
    print(f"\nTotal size: {total_size / 1024:.1f} KB")