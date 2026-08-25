import cv2
import os

def load_cascades():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    face_xml = os.path.join(base_dir, 'haarcascade_frontalface_default.xml')
    eye_xml = os.path.join(base_dir, 'haarcascade_eye.xml')

    face_cascade = cv2.CascadeClassifier(face_xml if os.path.exists(face_xml) else cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(eye_xml if os.path.exists(eye_xml) else cv2.data.haarcascades + 'haarcascade_eye.xml')

    return face_cascade, eye_cascade

face_cascade, eye_cascade = load_cascades()

def detect_face_and_liveness(frame):
    """
    Detects faces and verifies liveness via eye feature scanning.
    Returns: (has_face, faces, is_live)
    """
    if frame is None or face_cascade.empty():
        return False, [], False

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

    if len(faces) == 0:
        return False, [], False

    is_live = False
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        # Detect eyes within the upper face Region of Interest (ROI)
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=6, minSize=(20, 20))
        if len(eyes) >= 1:
            is_live = True
            break

    return True, faces, is_live