import os
import cv2

def load_cascades():
    # Look for local XML files in the repository first
    face_xml = "haarcascade_frontalface_default.xml"
    eye_xml = "haarcascade_eye.xml"

    # Fallback to OpenCV internal data if available
    if not os.path.exists(face_xml) and hasattr(cv2, 'data'):
        face_xml = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        eye_xml = cv2.data.haarcascades + 'haarcascade_eye.xml'

    face_cascade = cv2.CascadeClassifier(face_xml)
    eye_cascade = cv2.CascadeClassifier(eye_xml)

    return face_cascade, eye_cascade

face_cascade, eye_cascade = load_cascades()

def detect_face_and_liveness(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    if face_cascade.empty():
        return False, frame, None

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(100, 100))

    if len(faces) == 0:
        return False, frame, None

    (x, y, w, h) = faces[0]
    face_crop = frame[y:y+h, x:x+w]
    roi_gray = gray[y:y+h, x:x+w]

    eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=10) if not eye_cascade.empty() else []
    
    color = (0, 255, 0) if len(eyes) > 0 else (0, 165, 255)
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    return len(eyes) > 0, frame, face_crop