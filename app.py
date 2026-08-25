import os
import cv2
import numpy as np
import pandas as pd
import psycopg2
import streamlit as st
from dotenv import load_dotenv
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer, RTCConfiguration
from liveness import detect_face_and_liveness

# 1. Environment & Configuration Setup
load_dotenv()

DATABASE_URL = st.secrets.get("DATABASE_URL", os.getenv("DATABASE_URL"))
AZURE_FACE_KEY = st.secrets.get("AZURE_FACE_KEY", os.getenv("AZURE_FACE_KEY"))

# WebRTC STUN Server configuration to allow P2P video streaming across cloud networks
RTC_CONFIG = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

st.set_page_config(page_title="AI Face Attendance System", layout="wide")
st.title("⚡ AI-Powered Cloud Face Attendance System")

# 2. Database Connection Helper
def get_db_connection():
    try:
        conn = psycopg2.connect(DATABASE_URL)
        return conn
    except Exception as e:
        st.error(f"Database Connection Error: {e}")
        return None

# 3. OpenCV LBPH Recognizer Initialization
@st.cache_resource
def train_opencv_recognizer():
    try:
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        # Fallback empty labels map; populate with database training images if available
        labels_map = {}
        return recognizer, labels_map
    except Exception as e:
        st.warning(f"Recognizer initialization deferred: {e}")
        return None, {}

recognizer, labels_map = train_opencv_recognizer()

# 4. Helper Database Operations
def fetch_logs():
    conn = get_db_connection()
    if conn:
        try:
            df = pd.read_sql_query("SELECT * FROM attendance_logs ORDER BY timestamp DESC LIMIT 50;", conn)
            conn.close()
            return df
        except Exception:
            conn.close()
    return pd.DataFrame(columns=["Log ID", "Name", "Timestamp", "Status"])

# 5. Video Processing Transformer for Live Webcam Stream
class VideoProcessor(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Run liveness and face detection logic from liveness.py
        is_live, processed_frame, face_crop = detect_face_and_liveness(img)
        
        return processed_frame

# 6. Streamlit User Interface Layout
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📷 Live Stream & Recognition")
    webrtc_streamer(
        key="face-attendance",
        video_processor_factory=VideoProcessor,
        rtc_configuration=RTC_CONFIG,
        media_stream_constraints={"video": True, "audio": False}
    )

with col2:
    st.subheader("📋 Real-time Logs")
    if st.button("Refresh Logs"):
        st.rerun()
    
    logs_df = fetch_logs()
    st.dataframe(logs_df, use_container_width=True)