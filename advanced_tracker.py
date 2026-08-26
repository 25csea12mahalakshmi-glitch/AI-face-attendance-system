import cv2
import time
import numpy as np
import psycopg2
import psycopg2.extras
from pgvector.psycopg2 import register_vector
from insightface.app import FaceAnalysis

# ==========================================
# 1. DATABASE SETUP (Neon PostgreSQL + pgvector)
# ==========================================
DATABASE_URL = "postgresql://neondb_owner:npg_x9ocp1KSXwTY@ep-purple-voice-ayjb8bh2-pooler.c-5.us-east-2.aws.neon.tech/neondb?sslmode=require"

try:
    conn = psycopg2.connect(DATABASE_URL)
    register_vector(conn)
    cursor = conn.cursor()
    print("Successfully connected to Neon PostgreSQL DB with pgvector support.")
except Exception as e:
    print(f"Database connection error: {e}")
    exit(1)

# ==========================================
# 2. INSIGHTFACE MODEL INITIALIZATION
# ==========================================
# Fixed model name to 'buffalo_l' (RetinaFace + ArcFace)
app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

# ==========================================
# 3. PIPELINE FUNCTIONS
# ==========================================

def assess_face_quality(face, min_size=80, min_det_score=0.6):
    """
    2. Face Quality Assessment
    Checks bbox size, detection confidence, and extreme head pose angles.
    """
    bbox = face.bbox.astype(int)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]

    if w < min_size or h < min_size:
        return False, f"Too Small ({w}x{h}px)"

    if face.det_score < min_det_score:
        return False, f"Low Score ({face.det_score:.2f})"

    pitch, yaw, roll = face.pose
    if abs(yaw) > 30 or abs(pitch) > 30:
        return False, f"Side Pose (Yaw:{int(yaw)})"

    return True, "Good Quality"


def match_identity(embedding, threshold=0.4):
    """
    3. Embeddings & 4. Identity Matching
    Uses Neon pgvector cosine distance (<=>) for 1:N match.
    """
    # L2 Normalization for Cosine distance
    norm_embedding = (embedding / np.linalg.norm(embedding)).tolist()

    match_query = """
        SELECT user_name, face_embedding <=> %s::vector AS distance 
        FROM recognized_face_logs 
        WHERE user_name != 'Unknown User'
        ORDER BY distance ASC LIMIT 1;
    """
    cursor.execute(match_query, (norm_embedding,))
    result = cursor.fetchone()

    if result and result[1] < threshold:
        return result[0], float(1.0 - result[1])  # Returns matched name and similarity score
    
    return "Unknown User", float(1.0 - (result[1] if result else 1.0))


def enroll_user(user_name, face):
    """
    5. Enrollment
    Saves a registered profile embedding into the database.
    """
    is_good, reason = assess_face_quality(face)
    if not is_good:
        print(f"[ENROLLMENT ERROR] Quality check failed: {reason}")
        return False

    norm_embedding = (face.embedding / np.linalg.norm(face.embedding)).tolist()
    bbox = face.bbox.astype(int)
    bbox_json = {"x": int(bbox[0]), "y": int(bbox[1]), "w": int(bbox[2]-bbox[0]), "h": int(bbox[3]-bbox[1])}

    insert_query = """
        INSERT INTO recognized_face_logs (user_name, confidence, face_embedding, bounding_box)
        VALUES (%s, %s, %s::vector, %s);
    """
    cursor.execute(insert_query, (user_name, float(face.det_score), norm_embedding, psycopg2.extras.Json(bbox_json)))
    conn.commit()
    print(f"[ENROLLMENT SUCCESS] Registered '{user_name}' to Neon database.")
    return True


def log_verification(name, face):
    """
    6. Verification & Audit Logging
    Logs detection events to Neon without blocking real-time video stream.
    """
    norm_embedding = (face.embedding / np.linalg.norm(face.embedding)).tolist()
    bbox = face.bbox.astype(int)
    bbox_json = {"x": int(bbox[0]), "y": int(bbox[1]), "w": int(bbox[2]-bbox[0]), "h": int(bbox[3]-bbox[1])}

    insert_query = """
        INSERT INTO recognized_face_logs (user_name, confidence, face_embedding, bounding_box)
        VALUES (%s, %s, %s::vector, %s);
    """
    cursor.execute(insert_query, (name, float(face.det_score), norm_embedding, psycopg2.extras.Json(bbox_json)))
    conn.commit()


# ==========================================
# 4. REAL-TIME VIDEO STREAM LOOP
# ==========================================
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

last_log_time = time.time()
log_interval = 3.0  # Log to database every 3 seconds to avoid FPS drops
current_faces = []

print("\n--- CONTROLS ---")
print("Press 'e' -> Enroll detected face to database")
print("Press 'q' -> Quit video stream\n")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 1. Face Detection
    faces = app.get(frame)
    current_faces = faces

    for face in faces:
        bbox = face.bbox.astype(int)

        # 2. Quality Assessment
        is_quality_ok, quality_msg = assess_face_quality(face)

        if is_quality_ok:
            # 3 & 4. Embeddings & Identity Matching
            name, similarity = match_identity(face.embedding)
            
            if name != "Unknown User":
                color = (0, 255, 0)
                display_text = f"{name} ({similarity:.2f})"
            else:
                color = (0, 255, 255)
                display_text = "Unknown User"

            # Periodic background logging (6. Verification Logging)
            if time.time() - last_log_time > log_interval:
                log_verification(name, face)
                last_log_time = time.time()
        else:
            color = (0, 0, 255)
            display_text = f"Rejected: {quality_msg}"

        # Draw UI overlay
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
        cv2.rectangle(frame, (bbox[0], bbox[1] - 25), (bbox[2], bbox[1]), color, -1)
        cv2.putText(frame, display_text, (bbox[0] + 5, bbox[1] - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    # Instructions overlay
    cv2.putText(frame, "Press 'e' to Enroll | 'q' to Quit", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow('AI Face Engine (ArcFace + Neon pgvector)', frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('e'):
        if current_faces:
            new_name = input("\nEnter user name for enrollment: ").strip()
            if new_name:
                enroll_user(new_name, current_faces[0])
        else:
            print("\n[WARNING] No face detected to enroll.")

# Cleanup
cap.release()
cv2.destroyAllWindows()
cursor.close()
conn.close()