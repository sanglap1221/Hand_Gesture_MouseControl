import cv2
import mediapipe as mp
import pyautogui

# Initialize mediapipe Hands and drawing utilities
capture_hands = mp.solutions.hands.Hands()
drawing_option = mp.solutions.drawing_utils
screen_width, screen_height = pyautogui.size()

# Start webcam capture
camera = cv2.VideoCapture(0)

x1, y1, x2, y2 = 0, 0, 0, 0

# ✨ Added smoothing variables
prev_mouse_x, prev_mouse_y = 0, 0
smoothening = 5  # larger = smoother but slower

while True:
    _, image = camera.read()
    image_height, image_width, _ = image.shape

    # Flip image horizontally for natural mirror view
    image = cv2.flip(image, 1)

    # Convert BGR to RGB (mediapipe requires RGB)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process image and detect hands
    output_hands = capture_hands.process(rgb_image)
    all_hands = output_hands.multi_hand_landmarks

    # Draw hand landmarks if found
    if all_hands:
        for hand_landmarks in all_hands:
            drawing_option.draw_landmarks(image, hand_landmarks)
            one_hand_landmarks = hand_landmarks.landmark
            for id, lm in enumerate(one_hand_landmarks):
                x = int(lm.x * image_width)
                y = int(lm.y * image_height)

                if id == 8:  # Index finger tip
                    # ✨ Compute target mouse position
                    target_mouse_x = int(screen_width * lm.x)
                    target_mouse_y = int(screen_height * lm.y)

                    # ✨ Apply smoothing
                    mouse_x = int(prev_mouse_x + (target_mouse_x - prev_mouse_x) / smoothening)
                    mouse_y = int(prev_mouse_y + (target_mouse_y - prev_mouse_y) / smoothening)

                    cv2.circle(image, (x, y), 10, (0, 255, 255), -1)

                    # ✨ Move the mouse
                    pyautogui.moveTo(mouse_x, mouse_y)

                    # ✨ Update previous position
                    prev_mouse_x, prev_mouse_y = mouse_x, mouse_y

                    x1 = x
                    y1 = y

                if id == 4:  # Thumb tip (just for distance, not moving cursor anymore)
                    x2 = x
                    y2 = y
                    cv2.circle(image, (x, y), 10, (0, 255, 255), -1)

        dist = y2 - y1
        # print(dist)
        if dist < 28:
            pyautogui.click()

    # Show the image with landmarks
    cv2.imshow('Hand movement video capture', image)

    # Exit on ESC key
    key = cv2.waitKey(1)
    if key == 27:  # ESC key
        break

camera.release()
cv2.destroyAllWindows()
