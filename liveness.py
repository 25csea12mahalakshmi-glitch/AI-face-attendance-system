import os
import cv2

def load_cascades():
    # Dynamically pull XML files directly from installed OpenCV package data
    face_xml = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    eye_xml = cv2.data.haarcascades + 'haarcascade_eye.xml'

    face_cascade = cv2.CascadeClassifier(face_xml)
    eye_cascade = cv2.CascadeClassifier(eye_xml)

    # Verify cascades loaded correctly
    if face_cascade.empty():
        raise ValueError(f"Failed to load face cascade from: {face_xml}")
    if eye_cascade.empty():
        raise ValueError(f"Failed to load eye cascade from: {eye_xml}")

    return face_cascade, eye_cascade

# Initialize cascades
face_cascade, eye_cascade = load_cascades()

def detect_face_and_liveness(frame):
    """
    Detects faces and performs basic eye-detection liveness check on an input frame.
    Returns: (is_live, processed_frame, face_crop)
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, 
        scaleFactor=1.3, 
        minNeighbors=5, 
        minSize=(100, 100)
    )

    if len(faces) == 0:
        return False, frame, None

    # Process the primary detected face
    (x, y, w, h) = faces[0]
    face_crop = frame[y:y+h, x:x+w]
    roi_gray = gray[y:y+h, x:x+w]

    # Detect eyes within face region for basic liveness check
    eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=10)
    
    # Draw face bounding box
    color = (0, 255, 0) if len(eyes) > 0 else (0, 165, 255) # Green if eyes present, Orange if not
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    is_live = len(eyes) > 0
    return is_live, frame, face_crop