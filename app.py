import os
import sqlite3
from datetime import datetime
import cv2
import numpy as np
import pandas as pd
import psycopg2
import streamlit as st
from dotenv import load_dotenv
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer

from liveness import detect_face_and_liveness

# 1. Environment & Database Setup
load_dotenv()

PG_DATABASE_URL = os.getenv("DATABASE_URL", "")
DB_FILE = "attendance.db"
KNOWN_FACES_DIR = "known_faces"

def get_db_connection():
    if PG_DATABASE_URL:
        return psycopg2.connect(PG_DATABASE_URL)
    return sqlite3.connect(DB_FILE)

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    query = '''
        CREATE TABLE IF NOT EXISTS attendance (
            id SERIAL PRIMARY KEY,
            person_id TEXT NOT NULL,
            timestamp TIMESTAMP NOT NULL
        )
    ''' if PG_DATABASE_URL else '''
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            person_id TEXT NOT NULL,
            timestamp DATETIME NOT NULL
        )
    '''
    cursor.execute(query)
    conn.commit()
    conn.close()

init_db()

# 2. Page Config & CSS Styling
st.set_page_config(page_title="Global AI Attendance Platform", layout="wide")

st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
        color: #f8fafc;
    }
    .metric-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 18px;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .metric-title { color: #94a3b8; font-size: 14px; font-weight: 600; text-transform: uppercase; }
    .metric-value { color: #f8fafc; font-size: 24px; font-weight: 700; margin-top: 4px; }
    .stButton > button {
        background: linear-gradient(90deg, #6366f1 0%, #4f46e5 100%);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: 0.6rem 1.2rem;
        transition: all 0.2s ease-in-out;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.4);
    }
    </style>
""", unsafe_allow_html=True)

st.title("🛡️ Global AI Attendance Platform")
st.markdown("##### *24/7 Edge Face Recognition & Anti-Spoof Liveness Engine*")
st.divider()

# 3. Model Training
@st.cache_resource
def train_opencv_recognizer():
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
        return recognizer, labels_map
    return None, {}

recognizer, labels_map = train_opencv_recognizer()

# 4. Helper Database Operations
def fetch_logs():
    conn = get_db_connection()
    query = "SELECT person_id AS Person, timestamp AS Timestamp FROM attendance ORDER BY id DESC"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def log_attendance(person_name):
    conn = get_db_connection()
    cursor = conn.cursor()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    placeholder = "%s, %s" if PG_DATABASE_URL else "?, ?"
    cursor.execute(f"INSERT INTO attendance (person_id, timestamp) VALUES ({placeholder})", (person_name, timestamp))
    conn.commit()
    conn.close()

# 5. Top Metric Bar
logs_df = fetch_logs()
today_str = datetime.now().strftime("%Y-%m-%d")
logs_today = logs_df[logs_df['Timestamp'].astype(str).str.contains(today_str)] if not logs_df.empty else pd.DataFrame()

m1, m2, m3 = st.columns(3)
with m1:
    st.markdown(f'<div class="metric-card"><div class="metric-title">Check-ins Today</div><div class="metric-value">{len(logs_today)}</div></div>', unsafe_allow_html=True)
with m2:
    st.markdown(f'<div class="metric-card"><div class="metric-title">Registered Enrollees</div><div class="metric-value">{len(labels_map)}</div></div>', unsafe_allow_html=True)
with m3:
    db_type = "Cloud Postgres" if PG_DATABASE_URL else "Local SQLite"
    status_text = f"Guarded ({db_type})" if recognizer else "No Models Loaded"
    st.markdown(f'<div class="metric-card"><div class="metric-title">Engine Status</div><div class="metric-value" style="color:#10b981;">{status_text}</div></div>', unsafe_allow_html=True)

st.write("")

# 6. WebRTC Processor
class AttendanceVideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.latest_frame = None
        self.is_live = False
        self.has_face = False
        self.faces = []

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        self.has_face, self.faces, self.is_live = detect_face_and_liveness(img)
        self.latest_frame = img.copy()

        for (x, y, w, h) in self.faces:
            color = (0, 255, 0) if self.is_live else (0, 165, 255)
            cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)

        status_msg = "LIVE SUBJECT VERIFIED" if self.is_live else ("ANTI-SPOOF CHECK..." if self.has_face else "POSITION FACE IN FRAME")
        status_color = (0, 255, 0) if self.is_live else (0, 165, 255)
        cv2.putText(img, status_msg, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, status_color, 2)

        return img

# 7. Navigation Tabs
tab_live, tab_logs = st.tabs(["🎥 Live Verification Terminal", "📊 Audit Logs & Analytics"])

with tab_live:
    col_cam, col_info = st.columns([2, 1])

    with col_cam:
        ctx = webrtc_streamer(
            key="attendance-webrtc",
            video_processor_factory=AttendanceVideoProcessor,
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            media_stream_constraints={"video": True, "audio": False}
        )

    with col_info:
        st.subheader("Control Panel")
        mark_btn = st.button("📸 Capture & Verify Identity")
        st.divider()
        st.info("💡 **Anti-Spoof Protocol:** Ensure proper lighting and blink to verify liveness before capture.")

        if mark_btn:
            if ctx and ctx.video_processor and ctx.video_processor.latest_frame is not None:
                processor = ctx.video_processor
                if not processor.has_face:
                    st.warning("No face detected in video stream!")
                elif not processor.is_live:
                    st.error("Anti-Spoof Alert: Liveness verification failed.")
                elif recognizer is None:
                    st.error("No facial embeddings registered in system database.")
                else:
                    gray = cv2.cvtColor(processor.latest_frame, cv2.COLOR_BGR2GRAY)
                    (x, y, w, h) = processor.faces[0]
                    face_roi = gray[y:y+h, x:x+w]
                    
                    label_id, confidence = recognizer.predict(face_roi)
                    
                    if confidence < 115 and label_id in labels_map:
                        matched_name = labels_map[label_id]
                        log_attendance(matched_name)
                        st.balloons()
                        st.success(f"Identity Verified! Logged entry for **{matched_name}**.")
                    else:
                        st.warning("Access Denied: Unrecognized biometric signature.")
            else:
                st.warning("Webcam stream disconnected. Please allow camera permissions.")

with tab_logs:
    st.subheader("Attendance Log Audit Trail")
    df_current = fetch_logs()
    
    if not df_current.empty:
        col_table, col_actions = st.columns([3, 1])
        with col_table:
            st.dataframe(df_current, width="stretch", height=400)
        with col_actions:
            st.markdown("### Data Operations")
            csv_data = df_current.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Export Audit Report (CSV)", data=csv_data, file_name="attendance_report.csv", mime="text/csv")
            
            st.divider()
            if st.button("🗑️ Wipe Attendance Database"):
                conn = get_db_connection()
                cursor = conn.cursor()
                cursor.execute("DELETE FROM attendance")
                conn.commit()
                conn.close()
                st.rerun()
    else:
        st.info("No audit logs recorded in database.")