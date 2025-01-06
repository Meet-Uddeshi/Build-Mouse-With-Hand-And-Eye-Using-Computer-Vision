import cv2
import mediapipe as mp
import pyautogui

# Initialize the webcam
webcam = cv2.VideoCapture(0)

# Check if the webcam is opened properly
if not webcam.isOpened():
    print("Error: Could not access the webcam.")
    exit()

# Set up hand detector from MediaPipe
hand_detector = mp.solutions.hands.Hands()

# Utility to draw landmarks
drawing_utils = mp.solutions.drawing_utils

# Get the screen resolution for mouse movement
screen_width, screen_height = pyautogui.size()

# Function to check for ESC key press and exit the program
def check_exit_key():
    """Check if the ESC key is pressed to exit the program."""
    if cv2.waitKey(1) & 0xFF == 27:  # 27 is the ESC key
        return True
    return False

# Initialize y-coordinate for the index finger
index_finger_y = 0

# Start the video capture loop
while True:
    try:
        # Capture frame from webcam
        ret, frame = webcam.read()

        if not ret:
            print("Error: Failed to capture frame.")
            break

        # Flip the frame horizontally for natural interaction
        flipped_frame = cv2.flip(frame, 1)

        # Get the dimensions of the frame
        frame_height, frame_width, _ = flipped_frame.shape

        # Convert the frame to RGB format (needed by MediaPipe)
        rgb_frame = cv2.cvtColor(flipped_frame, cv2.COLOR_BGR2RGB)

        # Process the frame to detect hands
        output = hand_detector.process(rgb_frame)

        # Get landmarks of the detected hands
        hands = output.multi_hand_landmarks

        if hands:
            # Loop through each detected hand
            for hand in hands:
                # Draw hand landmarks on the frame
                drawing_utils.draw_landmarks(flipped_frame, hand)

                # Extract the hand's landmarks
                landmarks = hand.landmark

                # Loop through each landmark
                for id, landmark in enumerate(landmarks):
                    # Calculate the position of the landmark on the frame
                    x = int(landmark.x * frame_width)
                    y = int(landmark.y * frame_height)

                    # If the landmark is the index finger (id 8), draw a circle and track its position
                    if id == 8:
                        cv2.circle(flipped_frame, (x, y), 10, (0, 255, 255), -1)
                        index_finger_x = screen_width / frame_width * x
                        index_finger_y = screen_height / frame_height * y

                    # If the landmark is the thumb (id 4), draw a circle and track its position
                    if id == 4:
                        cv2.circle(flipped_frame, (x, y), 10, (0, 255, 255), -1)
                        thumb_x = screen_width / frame_width * x
                        thumb_y = screen_height / frame_height * y

                        # Check if the thumb and index finger are close enough to trigger a click
                        if abs(index_finger_y - thumb_y) < 20:
                            pyautogui.click()
                            pyautogui.sleep(1)

                        # Move the mouse if the index finger is within a certain range of the thumb
                        elif abs(index_finger_y - thumb_y) < 100:
                            pyautogui.moveTo(index_finger_x, index_finger_y)

        # Display the frame with hand landmarks
        cv2.imshow('Virtual Mouse', flipped_frame)

        # Check if the user pressed ESC to exit
        if check_exit_key():
            break

    except KeyboardInterrupt:
        # Gracefully handle a manual interruption
        print("Program interrupted. Exiting...")
        break

# Release the webcam and close windows when done
webcam.release()
cv2.destroyAllWindows()
