import cv2
import mediapipe as mp
import pyautogui

# Initialize the webcam
webcam = cv2.VideoCapture(0)

# Set up face mesh detector from MediaPipe
face_mesh_detector = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)

# Get screen resolution
screen_width, screen_height = pyautogui.size()

# Function to check for ESC key press and exit the program
def check_exit_key():
    """Check if the ESC key is pressed to exit the program."""
    if cv2.waitKey(1) & 0xFF == 27:  # 27 is the ESC key
        return True
    return False

# Start the video capture loop
while True:
    # Capture frame from webcam
    ret, frame = webcam.read()

    # Flip the frame horizontally for natural interaction
    flipped_frame = cv2.flip(frame, 1)

    # Get the dimensions of the frame
    frame_height, frame_width, _ = flipped_frame.shape

    # Convert the frame to RGB format (needed by MediaPipe)
    rgb_frame = cv2.cvtColor(flipped_frame, cv2.COLOR_BGR2RGB)

    # Process the frame to detect faces
    processed_output = face_mesh_detector.process(rgb_frame)

    # Get landmarks of the detected face
    landmarks_list = processed_output.multi_face_landmarks

    if landmarks_list:
        # Extract the landmarks for the first face
        face_landmarks = landmarks_list[0].landmark

        # Loop through specific landmarks (near the eyes)
        for index, landmark in enumerate(face_landmarks[474:478]):
            # Calculate the position on the frame
            x_pos = int(landmark.x * frame_width)
            y_pos = int(landmark.y * frame_height)

            # Draw circle on the face landmarks
            cv2.circle(flipped_frame, (x_pos, y_pos), 3, (0, 255, 0))

            # If the second landmark in the region, move the mouse
            if index == 1:
                # Map the face landmark to screen coordinates
                screen_x = screen_width * landmark.x
                screen_y = screen_height * landmark.y

                # Move the mouse to the position
                pyautogui.moveTo(screen_x, screen_y)

        # Detect if the left eye landmarks are aligned for a click
        left_eye_landmarks = [face_landmarks[145], face_landmarks[159]]

        for landmark in left_eye_landmarks:
            # Get the position of each landmark
            x_pos = int(landmark.x * frame_width)
            y_pos = int(landmark.y * frame_height)

            # Draw a circle around the eye landmarks
            cv2.circle(flipped_frame, (x_pos, y_pos), 3, (0, 255, 255))

        # If the eyes are aligned in a way that suggests a blink (click action)
        if (left_eye_landmarks[0].y - left_eye_landmarks[1].y) < 0.004:
            # Perform a mouse click and pause for 1 second
            pyautogui.click()
            pyautogui.sleep(1)

    # Display the frame with the facial landmarks
    cv2.imshow('Eye Controlled Mouse', flipped_frame)

    # Check if the user pressed ESC to exit
    if check_exit_key():
        break

# Release the webcam and close windows when done
webcam.release()
cv2.destroyAllWindows()
