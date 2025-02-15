# Step 1 => Importing necessary libraries and initializing webcam, hand detector, and drawing utilities
import cv2
import mediapipe as mp
import pyautogui
webcam = cv2.VideoCapture(0)
if not webcam.isOpened():
    print("Error: Could not access the webcam.")
    exit()
hand_detector = mp.solutions.hands.Hands()
drawing_utils = mp.solutions.drawing_utils

# Step 2 => Getting screen resolution
screen_width, screen_height = pyautogui.size()

# Step 3 => Function to check for ESC key press and exit
def check_exit_key():
    if cv2.waitKey(1) & 0xFF == 27:
        return True
    return False

# Step 4 => Initializing y-coordinate for the index finger
index_finger_y = 0

# Step 5 => Main video capture loop
while True:
    try:
        # Step 6 => Capturing frame, flipping it, and getting dimensions
        ret, frame = webcam.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break
        flipped_frame = cv2.flip(frame, 1)
        frame_height, frame_width, _ = flipped_frame.shape

        # Step 7 => Converting frame to RGB and processing for hand detection
        rgb_frame = cv2.cvtColor(flipped_frame, cv2.COLOR_BGR2RGB)
        output = hand_detector.process(rgb_frame)
        hands = output.multi_hand_landmarks

        # Step 8 => Hand landmark detection and mouse control logic
        if hands:
            for hand in hands:
                drawing_utils.draw_landmarks(flipped_frame, hand)
                landmarks = hand.landmark

                # Step 9 => Iterating through landmarks and identifying thumb and index finger
                for id, landmark in enumerate(landmarks):
                    x = int(landmark.x * frame_width)
                    y = int(landmark.y * frame_height)

                    if id == 8:  # Index finger
                        cv2.circle(flipped_frame, (x, y), 10, (0, 255, 255), -1)
                        index_finger_x = screen_width / frame_width * x
                        index_finger_y = screen_height / frame_height * y

                    if id == 4:  # Thumb
                        cv2.circle(flipped_frame, (x, y), 10, (0, 255, 255), -1)
                        thumb_x = screen_width / frame_width * x
                        thumb_y = screen_height / frame_height * y

                        # Step 10 => Click and Mouse movement logic
                        if abs(index_finger_y - thumb_y) < 20: #Click condition
                            pyautogui.click()
                            pyautogui.sleep(1)
                        elif abs(index_finger_y - thumb_y) < 100: #Mouse movement condition
                            pyautogui.moveTo(index_finger_x, index_finger_y)

        # Step 11 => Displaying the frame and checking for exit key
        cv2.imshow('Virtual Mouse', flipped_frame)
        if check_exit_key():
            break

    except KeyboardInterrupt:
        print("Program interrupted. Exiting...")
        break

# Step 12 => Releasing resources and closing windows
webcam.release()
cv2.destroyAllWindows()
