import os
import psycopg2
from psycopg2.extras import Json
from pgvector.psycopg2 import register_vector
from fastapi import FastAPI, File, UploadFile, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import cv2
import numpy as np
from insightface.app import FaceAnalysis
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Enterprise Face Attendance API", version="2.0.0")

# --- CORS Settings for React ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATABASE_URL = os.getenv("DATABASE_URL")

# --- Initialize InsightFace Model ---
face_app = FaceAnalysis(name='buffalo_s', providers=['CPUExecutionProvider'])
face_app.prepare(ctx_id=0, det_size=(640, 640))

# --- Pydantic Models ---
class Employee(BaseModel):
    employee_id: str
    name: str
    department: str

# --- Helper DB Connection ---
def get_db():
    conn = psycopg2.connect(DATABASE_URL)
    register_vector(conn)
    return conn

# --- Authentication Endpoint ---
@app.post("/token")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    if form_data.username == "admin" and form_data.password == "admin123":
        return {"access_token": "demo_jwt_token_secret", "token_type": "bearer"}
    raise HTTPException(status_code=400, detail="Incorrect credentials")

# --- Employee Management Endpoints ---
@app.post("/api/v1/employees")
async def add_employee(emp: Employee):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO employees (employee_id, name, department) VALUES (%s, %s, %s);",
        (emp.employee_id, emp.name, emp.department)
    )
    conn.commit()
    conn.close()
    return {"message": f"Employee {emp.name} added successfully."}

@app.get("/api/v1/employees")
async def get_employees():
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT employee_id, name, department FROM employees;")
    rows = cursor.fetchall()
    conn.close()
    return [{"employee_id": r[0], "name": r[1], "department": r[2]} for r in rows]

# --- Logs Endpoint (Neon PostgreSQL) ---
@app.get("/api/v1/logs")
def get_logs():
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT user_name, confidence, detected_at FROM recognized_face_logs ORDER BY detected_at DESC LIMIT 15;")
    rows = cursor.fetchall()
    conn.close()
    return [{"PersonID": row[0], "Confidence": row[1], "Timestamp": row[2].strftime("%Y-%m-%d %H:%M:%S")} for row in rows]

# --- Verification & Vector Match Endpoint ---
@app.post("/api/v1/verify")
async def verify_attendance(file: UploadFile = File(...)):
    contents = await file.read()
    np_img = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    faces = face_app.get(frame)
    if not faces:
        return JSONResponse(status_code=400, content={"status": "error", "message": "No face detected."})

    target_face = faces[0]
    embedding = target_face.embedding.tolist()

    conn = get_db()
    cursor = conn.cursor()

    # Perform pgvector Cosine Distance Search in Neon
    match_query = """
        SELECT user_name, face_embedding <=> %s::vector AS distance 
        FROM recognized_face_logs 
        ORDER BY distance ASC LIMIT 1;
    """
    cursor.execute(match_query, (embedding,))
    result = cursor.fetchone()

    name = "Unknown User"
    if result and result[1] < 0.4:
        name = result[0]

    # Insert verified frame into Neon
    bbox = target_face.bbox.astype(int)
    bbox_json = {"x": int(bbox[0]), "y": int(bbox[1]), "w": int(bbox[2]-bbox[0]), "h": int(bbox[3]-bbox[1])}
    cursor.execute(
        "INSERT INTO recognized_face_logs (user_name, confidence, face_embedding, bounding_box) VALUES (%s, %s, %s::vector, %s);",
        (name, float(target_face.det_score), embedding, Json(bbox_json))
    )
    conn.commit()
    conn.close()

    return {"status": "success", "person_name": name, "confidence": float(target_face.det_score)}