import cv2
import numpy as np
from liveness import detect_face_and_liveness
from advanced_tracker import load_database, extract_face_embedding, match_identity, enroll_user

def run_realtime_stream():
    # Load enrolled user database
    enrolled_db = load_database()
    
    # Initialize OpenCV VideoCapture
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Frame skipping controls for real-time performance (keeps FPS high)
    frame_count = 0
    process_interval = 3
    cached_result = None

    print("\n==========================================")
    print("REAL-TIME FACE RECOGNITION ACTIVE")
    print("Commands:")
    print("  - Press 'e': Enroll current face")
    print("  - Press 'q': Exit continuous stream")
    print("==========================================\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab video frame.")
            break

        frame_count += 1

        # Run heavy AI analysis every N frames to keep performance smooth
        if frame_count % process_interval == 0:
            is_live, _, face_crop, bbox, msg = detect_face_and_liveness(frame)

            if is_live and face_crop is not None:
                emb = extract_face_embedding(face_crop)
                matched_name, score = match_identity(emb, enrolled_db)
                
                label = f"{matched_name} ({score:.2f})"
                color = (0, 255, 0) if matched_name != "Unknown" else (0, 255, 255)
            else:
                label = f"Rejected: {msg}"
                color = (0, 0, 255)
                matched_name = "Unknown"

            cached_result = {
                "bbox": bbox,
                "label": label,
                "color": color,
                "face_crop": face_crop,
                "is_live": is_live
            }

        # Render Bounding Box & Label Overlay safely using cached results
        if cached_result and cached_result.get("bbox") is not None:
            bbox = cached_result["bbox"]
            
            # Verify bbox structure (x, y, w, h) before drawing
            if isinstance(bbox, (list, tuple, np.ndarray)) and len(bbox) == 4 and sum(bbox) > 0:
                x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                color = cached_result["color"]

                # Main face bounding box
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                
                # Header text banner (max bounds check prevents off-screen clipping)
                cv2.rectangle(frame, (x, max(0, y - 25)), (x + w, y), color, -1)
                cv2.putText(
                    frame, cached_result["label"], (x + 5, max(12, y - 7)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA
                )

        # On-screen Control Instructions
        cv2.putText(
            frame, "Press 'e' to Enroll | 'q' to Quit", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
        )

        # Display Frame
        cv2.imshow("Real-Time Face Recognition System", frame)

        # Keyboard Event Handling
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        elif key == ord('e'):
            if cached_result and cached_result["is_live"] and cached_result["face_crop"] is not None:
                print("\n[ENROLLMENT INITIATED]")
                new_user_name = input("Enter Name for detected face: ").strip()
                if new_user_name:
                    success, message = enroll_user(new_user_name, cached_result["face_crop"], enrolled_db)
                    print(f"Status: {message}\n")
                    # Refresh active in-memory database after enrollment
                    enrolled_db = load_database()
            else:
                print("\n[ENROLLMENT FAILED] Ensure a valid, live face is in the frame.\n")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_realtime_stream()