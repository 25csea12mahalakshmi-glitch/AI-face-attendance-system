import os
import cv2

def load_cascades():
    # Instantiate empty classifiers first (never crashes)
    face_cascade = cv2.CascadeClassifier()
    eye_cascade = cv2.CascadeClassifier()

    # Determine valid path candidates
    paths_to_check = [
        "haarcascade_frontalface_default.xml",
        os.path.join(getattr(cv2, 'data', None).haarcascades, 'haarcascade_frontalface_default.xml') if hasattr(cv2, 'data') and cv2.data and hasattr(cv2.data, 'haarcascades') else ""
    ]

    for path in paths_to_check:
        if path and os.path.exists(path):
            face_cascade.load(path)
            eye_path = path.replace('haarcascade_frontalface_default.xml', 'haarcascade_eye.xml')
            if os.path.exists(eye_path):
                eye_cascade.load(eye_path)
            break

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