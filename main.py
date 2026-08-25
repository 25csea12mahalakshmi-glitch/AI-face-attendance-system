import io
import os
import sqlite3
from datetime import datetime
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from liveness import detect_face_and_liveness

app = FastAPI(title="Face Attendance API Engine", version="1.0.0")

DB_FILE = "attendance.db"
KNOWN_FACES_DIR = "known_faces"

def init_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            person_id TEXT NOT NULL,
            timestamp DATETIME NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

init_db()

# Global recognizer variables
recognizer = None
labels_map = {}

def train_recognizer():
    global recognizer, labels_map
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    labels_map = {}
    faces_data = []
    labels_data = []

    if not os.path.exists(KNOWN_FACES_DIR):
        os.makedirs(KNOWN_FACES_DIR)

    current_id = 0
    cascade_path = os.path.join(os.path.dirname(__file__), "haarcascade_frontalface_default.xml")
    face_cascade = cv2.CascadeClassifier(cascade_path if os.path.exists(cascade_path) else cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    for file in os.listdir(KNOWN_FACES_DIR):
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            name = os.path.splitext(file)[0]
            labels_map[current_id] = name
            img_path = os.path.join(KNOWN_FACES_DIR, file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            
            if img is not None:
                detected = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5)
                for (x, y, w, h) in detected:
                    faces_data.append(img[y:y+h, x:x+w])
                    labels_data.append(current_id)
                if len(detected) == 0:
                    faces_data.append(img)
                    labels_data.append(current_id)
            current_id += 1

    if faces_data:
        recognizer.train(faces_data, np.array(labels_data))

@app.on_event("startup")
def startup_event():
    train_recognizer()

@app.get("/api/v1/health")
def health_check():
    return {"status": "online", "model_loaded": recognizer is not None}

@app.get("/api/v1/logs")
def get_logs():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT person_id, timestamp FROM attendance ORDER BY id DESC LIMIT 15")
    rows = cursor.fetchall()
    conn.close()
    return [{"PersonID": row[0], "Timestamp": row[1]} for row in rows]

@app.post("/api/v1/verify")
async def verify_attendance(file: UploadFile = File(...)):
    contents = await file.read()
    np_img = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    has_face, faces, is_live = detect_face_and_liveness(frame)

    if not has_face:
        return JSONResponse(status_code=400, content={"status": "error", "message": "No face detected in camera frame."})

    if not is_live:
        return JSONResponse(status_code=403, content={"status": "rejected", "message": "Anti-Spoof Alert: Liveness verification failed."})

    if recognizer is None:
        return JSONResponse(status_code=500, content={"status": "error", "message": "Model not initialized."})

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    (x, y, w, h) = faces[0]
    face_roi = gray[y:y+h, x:x+w]

    label_id, confidence = recognizer.predict(face_roi)

    if confidence < 115 and label_id in labels_map:
        matched_name = labels_map[label_id]
        
        # Log into DB
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute("INSERT INTO attendance (person_id, timestamp) VALUES (?, ?)", (matched_name, timestamp))
        conn.commit()
        conn.close()

        return {"status": "success", "person_name": matched_name, "confidence": float(confidence)}
    
    return JSONResponse(status_code=404, content={"status": "unrecognized", "message": "Unrecognized face."})