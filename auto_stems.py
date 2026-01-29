"""
Auto Stem Manager - PERSISTENT VERSION
Automatically generates stems on startup ONLY if they don't exist
Keeps stems on disk permanently to avoid 20-minute regeneration
"""

import os
import subprocess
import shutil
from pathlib import Path

class AutoStemManager:
    def __init__(self, music_folder):
        self.music_folder = Path(music_folder)
        self.stems_folder = self.music_folder / "_temp_stems"
        self.generated_stems = []
        self.demucs_available = self._check_demucs()
        
        # NO cleanup on exit - keep stems permanently!
        # atexit.register(self.cleanup)  # REMOVED
    
    def _check_demucs(self):
        """Check if Demucs is installed"""
        try:
            result = subprocess.run(['demucs', '--help'], capture_output=True, timeout=5)
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False
    
    def _stems_exist(self, mp3_path, full_separation=False):
        """Check if stems already exist for this song"""
        mp3_path = Path(mp3_path)
        song_stem = mp3_path.stem
        
        if full_separation:
            # Check for all 4-stem files
            required_stems = [
                self.music_folder / f"{song_stem}_vocals.mp3",
                self.music_folder / f"{song_stem}_drums.mp3",
                self.music_folder / f"{song_stem}_bass.mp3",
                self.music_folder / f"{song_stem}_other.mp3"
            ]
        else:
            # Check for 2-stem files
            required_stems = [
                self.music_folder / f"{song_stem}_vocals.mp3",
                self.music_folder / f"{song_stem}_instrumental.mp3"
            ]
        
        return all(stem.exists() for stem in required_stems)

    # -----------------------------
    # Stem Separation
    # -----------------------------
    def separate_song(self, mp3_path):
        """Fast 2-stem separation (vocals + instrumental)"""
        if not self.demucs_available:
            print(f"  ⚠ Demucs not available, skipping stem separation")
            return None
        
        mp3_path = Path(mp3_path)
        song_name = mp3_path.stem
        
        # Check if stems already exist
        if self._stems_exist(mp3_path, full_separation=False):
            print(f"  ✓ Stems already exist for {mp3_path.name}, skipping...")
            stems = {
                'vocals': str(self.music_folder / f"{song_name}_vocals.mp3"),
                'instrumental': str(self.music_folder / f"{song_name}_instrumental.mp3")
            }
            return stems
        
        print(f"  Separating stems for {mp3_path.name}...")
        
        try:
            cmd = [
                'demucs',
                '--two-stems', 'vocals',
                '-o', str(self.stems_folder),
                '--mp3',
                '--mp3-bitrate', '192',
                str(mp3_path)
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            
            if result.returncode != 0:
                print(f"  ❌ Failed to separate {mp3_path.name}")
                if result.stderr:
                    print(f"     Error: {result.stderr[:200]}")
                return None
            
            demucs_output = self.stems_folder / 'htdemucs' / song_name
            if not demucs_output.exists():
                print(f"  ❌ Demucs output not found at {demucs_output}")
                return None
            
            stems = {}
            vocals_file = demucs_output / 'vocals.mp3'
            if vocals_file.exists():
                dest_vocals = self.music_folder / f"{song_name}_vocals.mp3"
                shutil.copy2(vocals_file, dest_vocals)
                stems['vocals'] = str(dest_vocals)
                self.generated_stems.append(dest_vocals)

            inst_file = demucs_output / 'no_vocals.mp3'
            if inst_file.exists():
                dest_inst = self.music_folder / f"{song_name}_instrumental.mp3"
                shutil.copy2(inst_file, dest_inst)
                stems['instrumental'] = str(dest_inst)
                self.generated_stems.append(dest_inst)
            
            # Clean up temp folder after copying
            if self.stems_folder.exists():
                shutil.rmtree(self.stems_folder, ignore_errors=True)
            
            print(f"  ✓ Generated {len(stems)} stems: {', '.join(stems.keys())}")
            return stems
        
        except subprocess.TimeoutExpired:
            print(f"  ❌ Error separating {mp3_path.name}: Timeout after 600 seconds")
            return None
        except Exception as e:
            print(f"  ❌ Error separating {mp3_path.name}: {e}")
            return None

    def separate_song_full(self, mp3_path):
        """Full 4-stem separation (drums, bass, other, vocals)"""
        if not self.demucs_available:
            return None
        
        mp3_path = Path(mp3_path)
        song_name = mp3_path.stem
        
        # Check if stems already exist
        if self._stems_exist(mp3_path, full_separation=True):
            print(f"  ✓ Stems already exist for {mp3_path.name}, skipping...")
            stems = {
                'vocals': str(self.music_folder / f"{song_name}_vocals.mp3"),
                'drums': str(self.music_folder / f"{song_name}_drums.mp3"),
                'bass': str(self.music_folder / f"{song_name}_bass.mp3"),
                'other': str(self.music_folder / f"{song_name}_other.mp3")
            }
            # Add instrumental if it exists
            inst_path = self.music_folder / f"{song_name}_instrumental.mp3"
            if inst_path.exists():
                stems['instrumental'] = str(inst_path)
            return stems
        
        print(f"  Full stem separation for {mp3_path.name}...")
        
        try:
            cmd = [
                'demucs',
                '-o', str(self.stems_folder),
                '--mp3',
                '--mp3-bitrate', '192',
                str(mp3_path)
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
            
            if result.returncode != 0:
                print(f"  ❌ Failed full separation for {mp3_path.name}")
                return None
            
            demucs_output = self.stems_folder / 'htdemucs' / song_name
            if not demucs_output.exists():
                return None
            
            stems = {}
            for stem_type, filename in {
                'drums': 'drums.mp3',
                'vocals': 'vocals.mp3',
                'bass': 'bass.mp3',
                'other': 'other.mp3'
            }.items():
                stem_file = demucs_output / filename
                if stem_file.exists():
                    dest_file = self.music_folder / f"{song_name}_{stem_type}.mp3"
                    shutil.copy2(stem_file, dest_file)
                    stems[stem_type] = str(dest_file)
                    self.generated_stems.append(dest_file)

            # Also copy no_vocals as instrumental
            no_vocals_file = demucs_output / 'no_vocals.mp3'
            if no_vocals_file.exists():
                dest_inst = self.music_folder / f"{song_name}_instrumental.mp3"
                shutil.copy2(no_vocals_file, dest_inst)
                stems['instrumental'] = str(dest_inst)
                self.generated_stems.append(dest_inst)
            
            # Clean up temp folder after copying
            if self.stems_folder.exists():
                shutil.rmtree(self.stems_folder, ignore_errors=True)
            
            print(f"  ✓ Generated {len(stems)} stems: {', '.join(stems.keys())}")
            return stems
        
        except subprocess.TimeoutExpired:
            print(f"  ❌ Error in full separation: Timeout after 900 seconds")
            return None
        except Exception as e:
            print(f"  ❌ Error in full separation: {e}")
            return None

    def process_all_songs(self, mp3_files, full_separation=False):
        """Process all songs in folder (skip if stems already exist)"""
        if not self.demucs_available:
            print("⚠ Demucs not installed, skipping stem generation")
            return
        
        self.stems_folder.mkdir(exist_ok=True)
        
        # Count how many songs need processing
        songs_to_process = [f for f in mp3_files if not self._stems_exist(f, full_separation)]
        
        if not songs_to_process:
            print("✓ All stems already exist! Skipping generation (fast startup!)")
            return
        
        print(f"Found {len(songs_to_process)} songs needing stem generation...")
        print(f"({len(mp3_files) - len(songs_to_process)} already have stems)")
        
        # First song will download the model if needed
        if songs_to_process:
            print("Note: First-time model download (~320MB) happens once only!")
            print()
        
        for mp3_file in mp3_files:
            if full_separation:
                self.separate_song_full(mp3_file)
            else:
                self.separate_song(mp3_file)
        
        print(f"✓ Stem generation complete!")
        print(f"💾 Stems saved permanently - next startup will be instant!")

    # -----------------------------
    # Cleanup (DISABLED - keep stems permanently)
    # -----------------------------
    def cleanup(self):
        """Cleanup disabled - stems are kept permanently on disk"""
        # Only clean up the temp processing folder
        if self.stems_folder.exists():
            shutil.rmtree(self.stems_folder, ignore_errors=True)
        # Do NOT delete the generated stem files
        print("💾 Stems kept on disk for next session")

# -----------------------------
# Global instance
# -----------------------------
_auto_stem_manager = None

def get_auto_stem_manager(music_folder=None):
    global _auto_stem_manager
    if _auto_stem_manager is None and music_folder is not None:
        _auto_stem_manager = AutoStemManager(music_folder)
    return _auto_stem_manager