
import cv2
import mediapipe as mp
import pyautogui
webcam = cv2.VideoCapture(0)
face_mesh_detector = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)

# Step 2 => Getting screen resolution
screen_width, screen_height = pyautogui.size()

# Step 3 => Function to check for ESC key press and exit
def check_exit_key():
    if cv2.waitKey(1) & 0xFF == 27:  # 27 is the ESC key
        return True
    return False

# Step 4 => Main video capture loop
while True:
    # Step 5 => Capturing frame, flipping it, and getting dimensions
    ret, frame = webcam.read()
    flipped_frame = cv2.flip(frame, 1)
    frame_height, frame_width, _ = flipped_frame.shape

    # Step 6 => Converting frame to RGB and processing for face detection
    rgb_frame = cv2.cvtColor(flipped_frame, cv2.COLOR_BGR2RGB)
    processed_output = face_mesh_detector.process(rgb_frame)
    landmarks_list = processed_output.multi_face_landmarks

    # Step 7 => Face landmark extraction and mouse control
    if landmarks_list:
        face_landmarks = landmarks_list[0].landmark

        # Step 8 => Iterating through eye region landmarks and drawing circles
        for index, landmark in enumerate(face_landmarks[474:478]):
            x_pos = int(landmark.x * frame_width)
            y_pos = int(landmark.y * frame_height)
            cv2.circle(flipped_frame, (x_pos, y_pos), 3, (0, 255, 0))

            # Step 9 => Moving mouse based on landmark position
            if index == 1:  # Second landmark in the region (used for mouse movement)
                screen_x = screen_width * landmark.x
                screen_y = screen_height * landmark.y
                pyautogui.moveTo(screen_x, screen_y)

        # Step 10 => Eye blink detection and mouse click
        left_eye_landmarks = [face_landmarks[145], face_landmarks[159]]
        for landmark in left_eye_landmarks:
            x_pos = int(landmark.x * frame_width)
            y_pos = int(landmark.y * frame_height)
            cv2.circle(flipped_frame, (x_pos, y_pos), 3, (0, 255, 255))

        if (left_eye_landmarks[0].y - left_eye_landmarks[1].y) < 0.004:  # Blink detected
            pyautogui.click()
            pyautogui.sleep(1)

    # Step 11 => Displaying the frame and checking for exit key
    cv2.imshow('Eye Controlled Mouse', flipped_frame)
    if check_exit_key():
        break

# Step 12 => Releasing resources and closing windows
webcam.release()
cv2.destroyAllWindows()
