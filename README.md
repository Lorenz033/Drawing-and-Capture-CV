# Drawing-and-Capture-CV

# Finger Drawing App

A webcam-based drawing app that uses hand-tracking (MediaPipe) so you can draw on screen with your finger, erase, take photos of your canvas, and browse those photos — all with hand gestures. No mouse or touchscreen needed.

## How it works

Your webcam feed is analyzed frame-by-frame to detect your hand and fingertip positions. The position of your index fingertip becomes the "cursor," and which fingers you're holding up decides what that cursor does (draw, hover, erase, capture a photo, or open the photo viewer).

## Requirements

- Python 3
- A webcam
- The following Python packages:
  - `opencv-python`
  - `mediapipe`
  - `numpy`

Install them with:

```bash
pip install opencv-python mediapipe numpy
```

## Running the app

```bash
python your_script_name.py
```

A window titled **"Finger Drawing - press q to quit"** will open showing your webcam feed. Press **`q`** at any time to quit.

> **Note:** The script opens webcam index `1` (`cv2.VideoCapture(1)`), not the default `0`. If you get a black screen or "could not open webcam" error, try changing this to `0` in the code (or whichever index matches your webcam).

## Hand gestures

The app reads which fingers are extended to decide your current gesture. Only **one hand** is tracked by default (`MAX_HANDS = 1`).

| Gesture | Fingers held up | What it does |
|---|---|---|
| ✏️ **Draw** | Index finger only | Draws a line on the canvas, following your fingertip |
| 👆 **Hover** | Index + middle finger | Moves the cursor without drawing (lets you reposition without leaving a mark) |
| 🧹 **Erase** | Index + middle + ring + pinky (everything but thumb) | Erases a large area around your fingertip |
| 🤙 **Capture photo** | Pinky finger only | Starts a 5-second countdown, then saves a snapshot of your current canvas + camera view |
| 📷 **View gallery** | Middle + ring + pinky (not index) | Toggles a full-screen photo viewer showing your captured photos |
| ✊ **Idle** | Any other combination (e.g. a closed fist) | Does nothing — also used to **exit the gallery viewer** back to drawing mode |

A gesture must be held steady for a few frames before it "locks in" (this prevents flickering between gestures due to shaky detection).

## Photo viewer controls

Once you trigger the **view gallery** gesture (middle + ring + pinky), the screen switches to a full-screen photo browser:

- **Wave your hand** left or right to scroll through your captured photos.
- Or use the keyboard: **`a`** (previous photo) / **`d`** (next photo).
- Make a **fist** (idle gesture) to return to the drawing screen.

If you haven't captured any photos yet, the viewer will just show a prompt telling you to use the pinky-finger capture gesture.

## Keyboard shortcuts

These work in addition to the hand gestures:

| Key | Action |
|---|---|
| `1` – `5` | Change brush color (red, green, blue, yellow, magenta) |
| `+` / `=` | Increase brush thickness |
| `-` / `_` | Decrease brush thickness |
| `c` | Clear the canvas |
| `s` | Save the current canvas as a PNG to the `drawings/` folder |
| `a` | Previous photo (works in gallery mode) |
| `d` | Next photo (works in gallery mode) |
| `q` | Quit the app |

## Output files

The app automatically creates two folders in the same directory as the script:

- **`captures/`** — Photos taken with the pinky-finger capture gesture (`photo_<timestamp>.png`)
- **`drawings/`** — Canvases you manually save with the `s` key (`drawing_<timestamp>.png`)

## Tips

- Keep your hand fully in frame and reasonably well-lit — detection confidence is tuned fairly high (`min_detection_confidence=0.7`) to avoid false positives.
- If your fingertip line jumps or breaks unexpectedly, that's intentional: if your tracked point jumps more than 100px between frames (e.g. detection glitch), the app starts a new stroke instead of connecting it with a line.
- The on-screen top bar shows your current detected gesture and live FPS so you can gauge tracking quality.
