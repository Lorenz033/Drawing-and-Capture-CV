import cv2
import numpy as np
import mediapipe as mp
import time
import os
import math
from collections import deque

#
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

MAX_HANDS = 1

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=MAX_HANDS,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.6,
)

TIP_IDS = [4, 8, 12, 16, 20]
PIP_IDS = [3, 6, 10, 14, 18]

COLORS = {
    ord('1'): (255, 0, 0),
    ord('2'): (0, 255, 0),
    ord('3'): (0, 0, 255),
    ord('4'): (0, 255, 255),
    ord('5'): (255, 0, 255),
}

HAND_DEFAULT_COLORS = [(255, 0, 0), (0, 0, 255)]

brush_thickness = 12
eraser_thickness = 100

WIDTH, HEIGHT = 1280, 720
TOP_BAR = 60  # UI bar height

SMOOTHENING = 5
DEBOUNCE_FRAMES = 3
MAX_JUMP_DISTANCE = 100  # if the fingertip jumps farther than this between frames,
                          # start a new stroke instead of drawing a connecting line

# --- Photo capture / gallery settings ---------------------------------------
CAPTURE_DELAY = 5.0        # seconds of countdown before the photo is taken
CAPTURES_DIR = "captures"
THUMB_W, THUMB_H = 160, 90
GALLERY_VISIBLE = 6        # how many thumbnails fit in the strip at once
GALLERY_BAR_HEIGHT = 130   # strip height at the bottom of the screen

# --- Wave-to-scroll settings (photo viewer only) ----------------------------
WAVE_WINDOW = 0.3          # seconds of recent hand-x history to look at
WAVE_DISTANCE_THRESHOLD = 100  # px of horizontal movement within that window to count as a wave
WAVE_COOLDOWN = 0.3        # seconds before the same hand can trigger another scroll


def fingers_up(hand_landmarks, handedness_label):
    lm = hand_landmarks.landmark
    fingers = []

    if handedness_label == "Right":
        fingers.append(lm[TIP_IDS[0]].x < lm[PIP_IDS[0]].x)
    else:
        fingers.append(lm[TIP_IDS[0]].x > lm[PIP_IDS[0]].x)

    for tip, pip in zip(TIP_IDS[1:], PIP_IDS[1:]):
        fingers.append(lm[tip].y < lm[pip].y)

    return fingers


def classify_gesture(fingers):
    """Map a fingers_up() reading to: draw, hover, erase, capture, view, or idle."""
    thumb, index, middle, ring, pinky = fingers

    if index and not middle and not ring and not pinky:
        return "draw"
    if index and middle and not ring and not pinky:
        return "hover"
    if index and middle and ring and pinky:
        return "erase"
    if pinky and not index and not middle and not ring:
        return "capture"
    if middle and ring and pinky and not index:
        return "view"
    return "idle"


def ease_toward(prev, raw, smoothening):
    return prev + (raw - prev) / smoothening


class HandState:
    def __init__(self, color):
        self.color = color
        self.reset()

    def reset(self):
        self.prev_x, self.prev_y = 0, 0
        self.smooth_x, self.smooth_y = 0, 0
        self.pending_gesture = "idle"
        self.pending_count = 0
        self.active_gesture = "idle"
        self.seen_this_frame = False
        self.wave_history = deque()   # (timestamp, x) samples for wave detection
        self.last_wave_trigger = 0.0


def make_thumbnail(image):
    return cv2.resize(image, (THUMB_W, THUMB_H))


def draw_gallery_strip(combined, captured_images, selected_index, scroll_offset):
    bar_top = HEIGHT - GALLERY_BAR_HEIGHT
    overlay = combined.copy()
    cv2.rectangle(overlay, (0, bar_top), (WIDTH, HEIGHT), (20, 20, 20), cv2.FILLED)
    cv2.addWeighted(overlay, 0.85, combined, 0.15, 0, combined)

    if not captured_images:
        cv2.putText(combined, "No photos yet - show your pinky finger to capture one",
                    (20, bar_top + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        return

    spacing = 10
    start_x = 20
    visible = captured_images[scroll_offset:scroll_offset + GALLERY_VISIBLE]
    for i, item in enumerate(visible):
        real_idx = scroll_offset + i
        x = start_x + i * (THUMB_W + spacing)
        y = bar_top + 15
        combined[y:y + THUMB_H, x:x + THUMB_W] = item["thumb"]
        border_color = (0, 255, 255) if real_idx == selected_index else (120, 120, 120)
        thickness = 3 if real_idx == selected_index else 1
        cv2.rectangle(combined, (x, y), (x + THUMB_W, y + THUMB_H), border_color, thickness)

    cv2.putText(combined, f"Photo {selected_index + 1}/{len(captured_images)}  (wave hand, or a/d: scroll)",
                (WIDTH - 320, HEIGHT - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)


def main():
    global brush_thickness

    cap = cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

    if not cap.isOpened():
        print("Error: could not open webcam.")
        return

    canvas = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    hand_states = [HandState(HAND_DEFAULT_COLORS[i % len(HAND_DEFAULT_COLORS)])
                   for i in range(MAX_HANDS)]

    os.makedirs(CAPTURES_DIR, exist_ok=True)
    os.makedirs("drawings", exist_ok=True)

    captured_images = []   # list of {"thumb": ndarray, "full": ndarray, "path": str}
    selected_index = 0
    scroll_offset = 0

    capturing = False
    capture_trigger_time = None
    flash_until = 0  # timestamp until which to show a brief "Captured!" flash

    gallery_mode = False  # True = full-screen photo viewer, paused drawing

    p_time = 0

    print(f"Finger Drawing App started (tracking up to {MAX_HANDS} hands). Press 'q' to quit.")

    while True:
        success, frame = cap.read()
        if not success:
            print("Error: failed to grab frame from webcam.")
            break

        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (WIDTH, HEIGHT))
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        result = hands.process(rgb_frame)

        for state in hand_states:
            state.seen_this_frame = False

        status_lines = []

        if result.multi_hand_landmarks and result.multi_handedness:
            for idx, (hand_landmarks, handedness) in enumerate(
                zip(result.multi_hand_landmarks, result.multi_handedness)
            ):
                if idx >= MAX_HANDS:
                    break

                state = hand_states[idx]
                state.seen_this_frame = True
                handedness_label = handedness.classification[0].label

                if not gallery_mode:
                    mp_draw.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                        mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                        mp_draw.DrawingSpec(color=(255, 255, 255), thickness=2),
                    )

                fingers = fingers_up(hand_landmarks, handedness_label)
                raw_gesture = classify_gesture(fingers)

                if raw_gesture == state.pending_gesture:
                    state.pending_count += 1
                else:
                    state.pending_gesture = raw_gesture
                    state.pending_count = 1

                previous_active = state.active_gesture
                if state.pending_count >= DEBOUNCE_FRAMES:
                    state.active_gesture = raw_gesture

                # --- Toggle the full-screen photo viewer (edge-triggered) ---
                if state.active_gesture == "view" and previous_active != "view":
                    gallery_mode = not gallery_mode

                ix = int(hand_landmarks.landmark[8].x * WIDTH)
                iy = int(hand_landmarks.landmark[8].y * HEIGHT)

                if state.smooth_x == 0 and state.smooth_y == 0:
                    state.smooth_x, state.smooth_y = ix, iy
                else:
                    state.smooth_x = ease_toward(state.smooth_x, ix, SMOOTHENING)
                    state.smooth_y = ease_toward(state.smooth_y, iy, SMOOTHENING)

                sx, sy = int(state.smooth_x), int(state.smooth_y)
                label = f"Hand {idx + 1} ({handedness_label})"

                # --- Wave-to-scroll: track recent horizontal movement of this
                # hand and, while viewing photos, treat a fast enough swipe as
                # "wave right" / "wave left" to move through the gallery. ---
                now = time.time()
                state.wave_history.append((now, ix))
                while state.wave_history and now - state.wave_history[0][0] > WAVE_WINDOW:
                    state.wave_history.popleft()

                movement = 0
                if len(state.wave_history) >= 2:
                    movement = state.wave_history[-1][1] - state.wave_history[0][1]

                if (gallery_mode and captured_images
                        and now - state.last_wave_trigger > WAVE_COOLDOWN):
                    if movement > WAVE_DISTANCE_THRESHOLD:
                        selected_index = min(len(captured_images) - 1, selected_index + 1)
                        if selected_index >= scroll_offset + GALLERY_VISIBLE:
                            scroll_offset = selected_index - GALLERY_VISIBLE + 1
                        state.last_wave_trigger = now
                        state.wave_history.clear()
                    elif movement < -WAVE_DISTANCE_THRESHOLD:
                        selected_index = max(0, selected_index - 1)
                        if selected_index < scroll_offset:
                            scroll_offset = selected_index
                        state.last_wave_trigger = now
                        state.wave_history.clear()

                if gallery_mode:
           
                    status_lines.append(f"{label}: Viewing photos (wave: {int(movement)}px)")
                    state.prev_x, state.prev_y = 0, 0
                    continue

        
                if (state.active_gesture == "capture" and previous_active != "capture"
                        and not capturing):
                    capturing = True
                    capture_trigger_time = time.time()

                if state.active_gesture == "draw":
                    status_lines.append(f"{label}: Drawing")
                    cv2.circle(frame, (sx, sy), brush_thickness, state.color, cv2.FILLED)
                    if state.prev_x == 0 and state.prev_y == 0:
                        state.prev_x, state.prev_y = sx, sy
                    jump_dist = math.hypot(sx - state.prev_x, sy - state.prev_y)
                    if jump_dist <= MAX_JUMP_DISTANCE:
                        cv2.line(canvas, (int(state.prev_x), int(state.prev_y)), (sx, sy),
                                  state.color, brush_thickness)
                    state.prev_x, state.prev_y = sx, sy

                elif state.active_gesture == "hover":
                    status_lines.append(f"{label}: Hover")
                    cv2.circle(frame, (sx, sy), 12, state.color, 2)
                    state.prev_x, state.prev_y = 0, 0

                elif state.active_gesture == "erase":
                    status_lines.append(f"{label}: Erasing")
                    cv2.circle(frame, (sx, sy), eraser_thickness, (0, 0, 0), cv2.FILLED)
                    cv2.circle(canvas, (sx, sy), eraser_thickness, (0, 0, 0), cv2.FILLED)
                    state.prev_x, state.prev_y = 0, 0

                elif state.active_gesture == "capture":
                    status_lines.append(f"{label}: Capture")
                    state.prev_x, state.prev_y = 0, 0

                else:
                    status_lines.append(f"{label}: Idle")
                    state.prev_x, state.prev_y = 0, 0

        for state in hand_states:
            if not state.seen_this_frame:
                state.reset()

        if not status_lines:
            status_lines = ["No hand detected"]

        c_time = time.time()
        fps = 1 / (c_time - p_time) if p_time else 0
        p_time = c_time

        if gallery_mode:
            # --- Full-screen photo viewer ---
            if captured_images:
                combined = captured_images[selected_index]["full"].copy()
            else:
                combined = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
                cv2.putText(combined, "No photos yet - show your pinky finger to capture one",
                            (WIDTH // 2 - 340, HEIGHT // 2), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (200, 200, 200), 2)

            cv2.rectangle(combined, (0, 0), (WIDTH, TOP_BAR), (30, 30, 30), -1)
            cv2.putText(combined, "Viewing photos - make a fist to return to drawing",
                        (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            draw_gallery_strip(combined, captured_images, selected_index, scroll_offset)

        else:
            # --- Normal drawing view: merge canvas onto live frame ---
            gray_canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray_canvas, 10, 255, cv2.THRESH_BINARY_INV)
            mask_inv = cv2.bitwise_not(mask)

            frame_bg = cv2.bitwise_and(frame, frame, mask=mask)
            canvas_fg = cv2.bitwise_and(canvas, canvas, mask=mask_inv)
            combined = cv2.add(frame_bg, canvas_fg)

            if capturing:
                elapsed = time.time() - capture_trigger_time
                remaining = CAPTURE_DELAY - elapsed
                if remaining > 0:
                    countdown_num = int(math.ceil(remaining))
                    text = str(countdown_num)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    scale = 6
                    (tw, th), _ = cv2.getTextSize(text, font, scale, 8)
                    tx = (WIDTH - tw) // 2
                    ty = (HEIGHT + th) // 2
                    cv2.putText(combined, text, (tx, ty), font, scale, (0, 0, 0), 14)
                    cv2.putText(combined, text, (tx, ty), font, scale, (0, 255, 255), 8)
                else:
                    snapshot = combined.copy()
                    timestamp = int(time.time())
                    path = os.path.join(CAPTURES_DIR, f"photo_{timestamp}.png")
                    cv2.imwrite(path, snapshot)
                    captured_images.append({
                        "thumb": make_thumbnail(snapshot),
                        "full": snapshot,
                        "path": path,
                    })
                    selected_index = len(captured_images) - 1
                    if selected_index >= scroll_offset + GALLERY_VISIBLE:
                        scroll_offset = selected_index - GALLERY_VISIBLE + 1
                    capturing = False
                    flash_until = time.time() + 0.4
                    print(f"Captured: {path}")

            if time.time() < flash_until:
                flash = np.full_like(combined, 255)
                cv2.addWeighted(flash, 0.5, combined, 0.5, 0, combined)
                cv2.putText(combined, "Captured!", (WIDTH // 2 - 120, HEIGHT // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 128, 0), 4)

            cv2.rectangle(combined, (0, 0), (WIDTH, TOP_BAR), (30, 30, 30), -1)
            cv2.putText(combined, " | ".join(status_lines), (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(combined, f"FPS: {int(fps)}", (WIDTH - 150, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            for i, state in enumerate(hand_states):
                cv2.circle(combined, (WIDTH - 250 - i * 40, 30), 12, state.color, cv2.FILLED)

            draw_gallery_strip(combined, captured_images, selected_index, scroll_offset)

            cv2.putText(combined, "1-5: color | +/-: size | c: clear | s: save | q: quit",
                        (10, HEIGHT - GALLERY_BAR_HEIGHT - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (200, 200, 200), 1)

        cv2.imshow("Finger Drawing - press q to quit", combined)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c') and not gallery_mode:
            canvas = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        elif key in COLORS and not gallery_mode:
            hand_states[0].color = COLORS[key]
        elif key in (ord('+'), ord('=')) and not gallery_mode:
            brush_thickness = min(50, brush_thickness + 2)
        elif key in (ord('-'), ord('_')) and not gallery_mode:
            brush_thickness = max(2, brush_thickness - 2)
        elif key == ord('s') and not gallery_mode:
            filename = f"drawings/drawing_{int(time.time())}.png"
            cv2.imwrite(filename, canvas)
            print(f"Saved: {filename}")
        elif key == ord('a'):
            if captured_images:
                selected_index = max(0, selected_index - 1)
                if selected_index < scroll_offset:
                    scroll_offset = selected_index
        elif key == ord('d'):
            if captured_images:
                selected_index = min(len(captured_images) - 1, selected_index + 1)
                if selected_index >= scroll_offset + GALLERY_VISIBLE:
                    scroll_offset = selected_index - GALLERY_VISIBLE + 1

    cap.release()
    cv2.destroyAllWindows()
    hands.close()


if __name__ == "__main__":
    main()