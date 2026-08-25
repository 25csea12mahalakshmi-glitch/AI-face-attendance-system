import cv2

# Hardcode the default OpenCV cascade paths directly as strings 
# (Passing explicit string paths guarantees cv2.CascadeClassifier never receives None)
FACE_CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml' if hasattr(cv2, 'data') else ""
EYE_CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_eye.xml' if hasattr(cv2, 'data') else ""

face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
eye_cascade = cv2.CascadeClassifier(EYE_CASCADE_PATH)

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