import os
import cv2

# Global cache to prevent re-instantiating classifiers on every frame
_FACE_CASCADE = None
_EYE_CASCADE = None

def get_cascades():
    global _FACE_CASCADE, _EYE_CASCADE
    if _FACE_CASCADE is not None:
        return _FACE_CASCADE, _EYE_CASCADE

    try:
        # Load cascades dynamically when first needed
        cascade_dir = getattr(cv2.data, 'haarcascades', '') if hasattr(cv2, 'data') else ''
        face_path = os.path.join(cascade_dir, 'haarcascade_frontalface_default.xml') if cascade_dir else 'haarcascade_frontalface_default.xml'
        eye_path = os.path.join(cascade_dir, 'haarcascade_eye.xml') if cascade_dir else 'haarcascade_eye.xml'

        _FACE_CASCADE = cv2.CascadeClassifier(face_path if os.path.exists(face_path) else "")
        _EYE_CASCADE = cv2.CascadeClassifier(eye_path if os.path.exists(eye_path) else "")
    except Exception:
        _FACE_CASCADE = None
        _EYE_CASCADE = None

    return _FACE_CASCADE, _EYE_CASCADE

def detect_face_and_liveness(frame):
    if frame is None:
        return False, frame, None

    face_cascade, eye_cascade = get_cascades()
    if face_cascade is None or face_cascade.empty():
        return False, frame, None

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(100, 100))

    if len(faces) == 0:
        return False, frame, None

    (x, y, w, h) = faces[0]
    face_crop = frame[y:y+h, x:x+w]
    roi_gray = gray[y:y+h, x:x+w]

    eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=10) if eye_cascade and not eye_cascade.empty() else []
    
    color = (0, 255, 0) if len(eyes) > 0 else (0, 165, 255)
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    return len(eyes) > 0, frame, face_crop