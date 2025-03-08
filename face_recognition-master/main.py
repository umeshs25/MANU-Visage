import cv2
from deepface import DeepFace
from simple_facerec import SimpleFacerec  # Assuming you're using SimpleFacerec for face recognition

# Initialize the face recognizer
sfr = SimpleFacerec()
sfr.load_encoding_images("images/")  # Folder containing known face images

# Initialize video capture
cap = cv2.VideoCapture(0)  # Use 0 for the default webcam

while True:
    success, frame = cap.read()  # Read a frame from the webcam
    if not success:
        print("Failed to capture image.")
        break

    # Detect Faces
    face_locations, face_names = sfr.detect_known_faces(frame)
    for face_loc, name in zip(face_locations, '''face_names'''):
        y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]

        # Draw rectangle around the face
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # Display the name of the person
        cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 0), 2)

    try:
        # Analyze emotions for each detected face
        for face_loc in face_locations:
            # Crop the face from the frame
            face_frame = frame[face_loc[0]:face_loc[2], face_loc[3]:face_loc[1]]
            if face_frame.size > 0:  # Ensure face_frame is valid
                face_analysis = DeepFace.analyze(img_path=face_frame, actions=['emotion'], enforce_detection=False)
                emotion = face_analysis[0]['dominant_emotion']

                # Display the emotion on the frame
                # cv2.putText(frame, emotion, (x1, y2 + 20), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 255), 2)

    except Exception as e:
        print(f"Error analyzing frame: {e}")

    # Display the frame
    cv2.imshow('Face Recognition & Emotion Detection', frame)

    # Exit loop on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
