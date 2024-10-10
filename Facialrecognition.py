import cv2
import os
from deepface import DeepFace
import time

# Load Haar-cascade for face detection
cascade_path = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
face_classifier = cv2.CascadeClassifier(cascade_path)

# Start video stream from the camera
video_capture = cv2.VideoCapture(0)

# Initialize last analysis time
last_analysis_time = 0
analysis_interval = 4  # Analyze every 4 seconds

# Initialize variables to store last analysis results
last_age = None
last_dominant_emotion = None
last_dominant_gender = None
last_dominant_race = None

def detect_bounding_box_and_analyze(vid, perform_analysis):
    global last_age, last_dominant_emotion, last_dominant_gender, last_dominant_race

    # Convert image to grayscale for face detection
    gray_image = cv2.cvtColor(vid, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))

    for (x, y, w, h) in faces:
        # Draw rectangle around detected face
        cv2.rectangle(vid, (x, y), (x + w, y + h), (0, 255, 0), 2)

        if perform_analysis:
            # Extract face region from the original image
            face_region = vid[y:y + h, x:x + w]

            try:
                # Use DeepFace to analyze the face region
                analysis = DeepFace.analyze(face_region, actions=['emotion', 'age', 'gender', 'race'])
                print(analysis)
                # Update last analysis results
                last_age = analysis[0]["age"]
                last_dominant_emotion = analysis[0]["dominant_emotion"]
                last_dominant_gender = analysis[0]["dominant_gender"]
                last_dominant_race = analysis[0]["dominant_race"]

            except Exception as e:
                print(f"Error during DeepFace analysis: {e}")

        # Display results on the video stream using the last known values
        if last_age is not None:
            cv2.putText(vid, f"Emotion: {last_dominant_emotion}", (x, y - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(vid, f"Age: {int(last_age)}", (x, y - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(vid, f"Gender: {last_dominant_gender}", (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(vid, f"Race: {last_dominant_race}", (x, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0), 1, cv2.LINE_AA)

    return faces

while True:
    result, video_frame = video_capture.read()
    if not result:
        break

    current_time = time.time()
    if current_time - last_analysis_time >= analysis_interval:
        perform_analysis = True
        last_analysis_time = current_time
    else:
        perform_analysis = False

    # Call the function with perform_analysis flag
    detect_bounding_box_and_analyze(video_frame, perform_analysis)

    # Show video stream with results
    cv2.imshow("Real-time Face and Emotion Detection", video_frame)

    # Break loop if user presses 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Clean up and release resources
video_capture.release()
cv2.destroyAllWindows()