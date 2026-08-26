import os
import cv2
import numpy as np

# Global cache for Haar Classifiers to maximize FPS performance
_FACE_CASCADE = None
_EYE_CASCADE = None


def get_cascades():
    global _FACE_CASCADE, _EYE_CASCADE
    if _FACE_CASCADE is not None:
        return _FACE_CASCADE, _EYE_CASCADE

    try:
        cascade_dir = getattr(cv2.data, 'haarcascades', '') if hasattr(cv2, 'data') else ''
        face_path = os.path.join(cascade_dir, 'haarcascade_frontalface_default.xml') if cascade_dir else 'haarcascade_frontalface_default.xml'
        eye_path = os.path.join(cascade_dir, 'haarcascade_eye.xml') if cascade_dir else 'haarcascade_eye.xml'

        _FACE_CASCADE = cv2.CascadeClassifier(face_path if os.path.exists(face_path) else "")
        _EYE_CASCADE = cv2.CascadeClassifier(eye_path if os.path.exists(eye_path) else "")
    except Exception:
        _FACE_CASCADE = None
        _EYE_CASCADE = None

    return _FACE_CASCADE, _EYE_CASCADE


# ==========================================
# 1. FACE DETECTION & 2. QUALITY ASSESSMENT
# ==========================================

def calculate_blur_score(image):
    """Computes focus clarity using Laplacian variance."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def check_face_quality(face_crop, min_size=80, blur_threshold=80.0):
    """
    2. Face Quality Assessment
    Validates face resolution and sharpness/focus clarity.
    """
    if face_crop is None or face_crop.size == 0:
        return False, "Empty face crop"

    h, w, _ = face_crop.shape
    if w < min_size or h < min_size:
        return False, f"Low Resolution ({w}x{h}px)"

    blur_score = calculate_blur_score(face_crop)
    if blur_score < blur_threshold:
        return False, f"Blurry Input ({int(blur_score)})"

    return True, f"Good Quality ({int(blur_score)})"


def detect_face_and_liveness(frame, blur_threshold=80.0):
    """
    1. Face Detection & Anti-Spoofing Pipeline
    """
    if frame is None:
        return False, frame, None, (0, 0, 0, 0), "No Frame"

    face_cascade, eye_cascade = get_cascades()
    if face_cascade is None or face_cascade.empty():
        return False, frame, None, (0, 0, 0, 0), "Cascade Error"

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(100, 100))

    if len(faces) == 0:
        return False, frame, None, (0, 0, 0, 0), "No Face Detected"

    (x, y, w, h) = faces[0]
    bbox = (x, y, w, h)
    face_crop = frame[y:y+h, x:x+w]
    roi_gray = gray[y:y+h, x:x+w]

    # Quality Check
    is_quality_ok, quality_msg = check_face_quality(face_crop, blur_threshold=blur_threshold)

    # Eye Liveness Check
    eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=10) if eye_cascade and not eye_cascade.empty() else []
    has_eyes = len(eyes) >= 1

    is_live = is_quality_ok and has_eyes

    if is_live:
        color = (0, 255, 0)
        status_text = f"LIVE USER ({quality_msg})"
    elif not is_quality_ok:
        color = (0, 0, 255)
        status_text = f"SPOOF: {quality_msg}"
    else:
        color = (0, 165, 255)
        status_text = "SPOOF: No Eye Signal"

    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    cv2.putText(frame, status_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    return is_live, frame, face_crop, bbox, quality_msg


# ==========================================
# 3. EMBEDDINGS & 4. IDENTITY MATCHING
# ==========================================

def extract_face_embedding(face_crop):
    """
    3. Face Embeddings
    Extracts a normalized spatial feature vector from a cropped face.
    """
    resized = cv2.resize(face_crop, (64, 64))
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [18, 25], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()


def match_identity(target_embedding, enrolled_database, threshold=0.75):
    """
    4. Identity Matching (1:N search)
    Compares target vector against enrolled database.
    """
    if not enrolled_database:
        return "Unknown User", 0.0

    best_match = "Unknown User"
    highest_sim = -1.0

    for user_name, stored_embedding in enrolled_database.items():
        similarity = cv2.compareHist(target_embedding, stored_embedding, cv2.HISTCMP_CORREL)
        if similarity > highest_sim:
            highest_sim = similarity
            best_match = user_name

    if highest_sim >= threshold:
        return best_match, float(highest_sim)
    return "Unknown User", float(highest_sim)


# ==========================================
# 5. ENROLLMENT & 6. VERIFICATION
# ==========================================

def enroll_user(user_name, frame, enrolled_database):
    """
    5. Enrollment Process
    Detects face, verifies quality, extracts embedding, and registers user.
    """
    is_live, _, face_crop, _, msg = detect_face_and_liveness(frame)
    if not is_live or face_crop is None:
        return False, f"Enrollment Failed: {msg}"

    embedding = extract_face_embedding(face_crop)
    enrolled_database[user_name] = embedding
    return True, f"User '{user_name}' successfully enrolled."


def verify_user(claimed_name, frame, enrolled_database, threshold=0.75):
    """
    6. Verification Process (1:1 Verification)
    Verifies if target face matches claimed identity.
    """
    if claimed_name not in enrolled_database:
        return False, f"User '{claimed_name}' is not enrolled in database."

    is_live, _, face_crop, _, msg = detect_face_and_liveness(frame)
    if not is_live or face_crop is None:
        return False, f"Verification Failed: {msg}"

    target_embedding = extract_face_embedding(face_crop)
    enrolled_embedding = enrolled_database[claimed_name]

    similarity = cv2.compareHist(target_embedding, enrolled_embedding, cv2.HISTCMP_CORREL)
    is_matched = similarity >= threshold

    return is_matched, f"Match Score: {similarity:.2f} (Threshold: {threshold})"