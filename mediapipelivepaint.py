import cv2
import numpy as np
import mediapipe as mp
import time

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# Sketch effect function
def sketch(image, blur_strength=21):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    inverted = 255 - gray
    blurred = cv2.GaussianBlur(inverted, (blur_strength, blur_strength), 0)
    inverted_blur = 255 - blurred
    sketch_img = cv2.divide(gray, inverted_blur, scale=256.0)
    return sketch_img

# Function to count fingers 
def count_fingers(hand_landmarks):
    fingers = []

    # Tip IDs for 5 fingers (excluding wrist)
    tip_ids = [4, 8, 12, 16, 20]

    # Thumb
    if hand_landmarks.landmark[tip_ids[0]].x < hand_landmarks.landmark[tip_ids[0] - 1].x:
        fingers.append(1)
    else:
        fingers.append(0)

    # Other fingers
    for id in range(1, 5):
        if hand_landmarks.landmark[tip_ids[id]].y < hand_landmarks.landmark[tip_ids[id] - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)

    return sum(fingers)

# Open webcam
cap = cv2.VideoCapture(0)
save_count = 0
prev_finger_action = None
last_action_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    sketch_frame = sketch(frame)
    sketch_bgr = cv2.cvtColor(sketch_frame, cv2.COLOR_GRAY2BGR)
    combined = np.hstack((frame, sketch_bgr))

    # Convert for MediaPipe
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(combined, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            finger_count = count_fingers(hand_landmarks)

            # Debounce: only act if at least 2 seconds since last action
            if time.time() - last_action_time > 2:
                if finger_count == 5:
                    filename = f"gesture_sketch_{save_count}.png"
                    cv2.imwrite(filename, sketch_frame)
                    print(f"Saved sketch as {filename}")
                    save_count += 1
                    last_action_time = time.time()
                elif finger_count == 2:
                    print("Exit gesture detected. Quitting...")
                    cap.release()
                    cv2.destroyAllWindows()
                    exit()

            cv2.putText(combined, f"Fingers: {finger_count}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Live Sketch with Hand Control", combined)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
