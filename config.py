import os

# Azure Credentials
AZURE_ENDPOINT = "https://face-recognition-rg.cognitiveservices.azure.com/"
AZURE_KEY = os.getenv("AZURE_FACE_KEY", "")
PERSON_GROUP_ID = "company-employees"

# Liveness Settings
# Lower EAR means eyes are closed; a value below this triggers a blink detection
EAR_THRESHOLD = 0.21

# Files
ATTENDANCE_FILE = "attendance.csv"