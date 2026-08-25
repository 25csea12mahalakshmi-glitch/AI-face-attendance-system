import os
import cv2

def load_cascades():
    # 1. Start with local filenames
    face_xml = "haarcascade_frontalface_default.xml"
    eye_xml = "haarcascade_eye.xml"

    # 2. Check if local files exist; if not, try resolving OpenCV built-in directory safely
    if not os.path.exists(face_xml):
        data_path = getattr(cv2, 'data', None)
        cascade_dir = getattr(data_path, 'haarcascades', None) if data_path else None
        
        if cascade_dir:
            face_xml = os.path.join(cascade_dir, 'haarcascade_frontalface_default.xml')
            eye_xml = os.path.join(cascade_dir, 'haarcascade_eye.xml')

    # 3. Load classifiers cleanly
    face_cascade = cv2.CascadeClassifier(face_xml if os.path.exists(face_xml) else "")
    eye_cascade = cv2.CascadeClassifier(eye_xml if os.path.exists(eye_xml) else "")

    return face_cascade, eye_cascade

face_cascade, eye_cascade = load_cascades()

def detect_face_and_liveness(frame):
    if frame is None or face_cascade.empty():
        return False, frame, None

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
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