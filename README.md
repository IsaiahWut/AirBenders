# AirBenders — RoseHacks 2026

**A gesture-controlled DJ applications**

A gesture-controlled DJ application that simulates core DJ controller functionality without any physical hardware. This project uses hand tracking to allow users to mix, play, and manipulate music tracks using natural hand gestures.

---

## Features

### Deck Controls
* **Play / Pause:** Simple "Pinch" gesture to start or stop the music.
* **Load Tracks:** Select songs from a visual song list using gestures.
* **Jog Wheels:** Scrub, scratch, and nudge tracks like a professional DJ.

### Volume Controls
* **Independent Adjustment:** Adjust each deck independently with a **five-finger claw gesture**.
* **Dual Hand Support:** Left hand controls Deck 1, right hand controls Deck 2.
* **Visual Feedback:** UI tracks finger positions for real-time volume precision.

### Visualizer & UI
* **Frequency Visualizer:** Dynamic real-time visualization of audio tracks.
* **Active Deck Logic:** Highlights the active deck with a color-coded UI.
* **Song List:** Browse, select, and load music tracks with gesture-based toggling.

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3.11%2B-blue?style=for-the-badge&logo=python)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green?style=for-the-badge&logo=opencv)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Hand%20Tracking-orange?style=for-the-badge&logo=google)
![Pygame](https://img.shields.io/badge/Pygame-Audio%20Engine-yellow?style=for-the-badge&logo=pygame)

* **OpenCV:** Video capture & UI overlay.
* **MediaPipe:** High-fidelity hand and finger tracking.
* **pygame:** Music playback engine.
* **Custom Modules:** Built from scratch for UI components, song management, and gesture control.

---

## Getting Started

### Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YOUR_USERNAME/AirBenders.git](https://github.com/YOUR_USERNAME/AirBenders.git)
    cd AirBenders
    ```

2.  **Add Music:**
    Place your `.mp3` files into the `MP3/` folder.

3.  **Run the App:**
    ```bash
    python hands.py
    ```

---

## How to Use

| Gesture | Action | Description |
| :--- | :--- | :--- |
| **Pinch** (Thumb + Index) | **Play / Pause / Load** | Used to start/stop tracks or select a song from the list. |
| **Claw** (5 Fingers) | **Volume Control** | Open/Close your hand (claw shape) to raise or lower volume. |
| **Jog Wheel** | **Scrub / Scratch** | Interact with the virtual wheel to seek through the track. |
| **Swipe/Point** | **Song Selection** | Highlight and load songs from the library. |


---

## Future Plans
[ ] AI-assisted auto-mixing and beat matching.

[ ] Recording and exporting DJ sets.

[ ] Crossfader implementation for smooth transitions.

[ ] Custom user sound effects (SFX) pads.

---

## 📂 Project Structure

```text
AirBenders/
│
├── hands.py             # Main app entry point & gesture logic
├── volumeSlider.py      # Claw-based volume control logic
├── playbutton.py        # Play/Pause button UI class
├── jogwheel.py          # Jog wheel UI and scrubbing logic
├── music.py             # Music playback and mixer management
├── songlist.py          # Song list panel UI
├── load.py              # Load button functionality
├── visualizer.py        # Audio frequency visualizer
├── MP3/                 # Directory for user music files
└── README.md
```

👥 Contributors

Isaiah Alcayde

Paolo Uytiepo

Joshua Yu
