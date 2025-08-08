import cv2
import numpy as np
import pandas as pd
from datetime import datetime
import os
from PIL import Image
import sys
from flask import Flask, request, jsonify, render_template
import threading
import time

app = Flask(__name__)

# Initialize face recognizer and Haar Cascade
recognizer = cv2.face.LBPHFaceRecognizer_create()
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Define font for OpenCV display
font = cv2.FONT_HERSHEY_SIMPLEX

# Attendance file setup
file_name = "data/attendance.csv"

# Global variables for camera control
camera_running = False
current_cap = None

# Overwrite attendance file with headers at program start
os.makedirs(os.path.dirname(file_name), exist_ok=True)
df = pd.DataFrame(columns=["Name", "Date", "Time"])
df.to_csv(file_name, index=False)
print(f"[INFO] Initialized new attendance file: {file_name}")

# Global names dictionary
names = {0: "None"}

def try_open_webcam(max_attempts=3):
    """Attempt to open webcam with different indices and backends."""
    for index in range(max_attempts):
        for backend in [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]:
            cap = cv2.VideoCapture(index, backend)
            if cap.isOpened():
                print(f"[INFO] Webcam opened successfully with index {index} and backend {backend}")
                return cap, None
            cap.release()
    return None, "[ERROR] Cannot open webcam. Ensure it is connected and not in use."

def stop_camera():
    """Stop the current camera operation."""
    global camera_running, current_cap
    camera_running = False
    if current_cap is not None:
        current_cap.release()
        current_cap = None
    cv2.destroyAllWindows()
    print("[INFO] Camera stopped by user.")

def capture_faces(name):
    """Capture up to 5 face images for a given user and save them to dataset/<name>."""
    global camera_running, current_cap
    
    path = f"dataset/{name}"
    os.makedirs(path, exist_ok=True)
    
    current_cap, error = try_open_webcam()
    if error:
        print(error)
        return error
    
    camera_running = True
    count = 0
    max_images = 2
    images_captured = 0
    
    print(f"[INFO] Capturing faces for {name}. Looking for faces...")
    
    while images_captured < max_images and camera_running:
        ret, frame = current_cap.read()
        if not ret:
            stop_camera()
            print("[ERROR] Failed to capture image.")
            return "[ERROR] Failed to capture image."
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))
        
        # Draw rectangles around detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"Face {images_captured + 1}/{max_images}", (x, y-10), font, 0.7, (0, 255, 0), 2)
        
        cv2.putText(frame, f"Captured: {images_captured}/{max_images}", (10, 30), font, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, "Press 's' to save face, 'q' to quit", (10, frame.shape[0]-10), font, 0.5, (255, 255, 255), 1)
        cv2.imshow("Capturing Faces", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s') and len(faces) > 0:
            # Save the largest face detected
            largest_face = max(faces, key=lambda face: face[2] * face[3])
            x, y, w, h = largest_face
            face_img = frame[y:y+h, x:x+w]
            filename = os.path.join(path, f"{name}_{images_captured}.jpg")
            cv2.imwrite(filename, face_img)
            print(f"[INFO] Saved: {filename}")
            images_captured += 1
            time.sleep(0.5)  # Brief pause after saving
        elif key == ord('q'):
            print("[INFO] Stopping face capture early.")
            break
        
        count += 1
    
    stop_camera()
    return f"Face capture complete. Captured {images_captured} images for {name}."

def get_images_and_labels(path):
    """Retrieve face images and their corresponding labels from dataset directory."""
    face_samples = []
    ids = []
    name_to_id = {}
    current_id = 1
    
    for person_name in os.listdir(path):
        person_path = os.path.join(path, person_name)
        if not os.path.isdir(person_path):
            continue
        
        if person_name not in name_to_id:
            name_to_id[person_name] = current_id
            current_id += 1
        
        image_paths = [os.path.join(person_path, f) for f in os.listdir(person_path) if f.endswith('.jpg')]
        for image_path in image_paths:
            try:
                pil_image = Image.open(image_path).convert('L')
                image_np = np.array(pil_image, 'uint8')
                
                # Use the entire image as it's already a cropped face
                if image_np.size > 0:
                    face_samples.append(image_np)
                    ids.append(name_to_id[person_name])
            except Exception as e:
                print(f"[WARNING] Could not process image {image_path}: {e}")
                continue
    
    return face_samples, ids, name_to_id

def train_model():
    """Train the face recognizer using images in the dataset directory."""
    path = 'dataset'
    if not os.path.exists(path):
        return "[ERROR] Dataset directory not found. Capture faces first."
    
    print("[INFO] Training faces...")
    faces, ids, name_to_id = get_images_and_labels(path)
    
    if not faces:
        return "[ERROR] No faces found to train. Please capture faces first."
    
    recognizer.train(faces, np.array(ids))
    
    os.makedirs('trainer', exist_ok=True)
    recognizer.write('trainer/trainer.yml')
    
    global names
    names = {v: k for k, v in name_to_id.items()}
    names[0] = "Unknown"
    
    print(f"[INFO] Training completed with {len(set(ids))} unique faces")
    return f"Training completed! Trained {len(set(ids))} unique faces. Model saved."

def mark_attendance(name):
    """Mark attendance for a recognized person in the CSV file and return True if marked."""
    now = datetime.now()
    date = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")
    
    try:
        # Read existing attendance
        if os.path.exists(file_name):
            df = pd.read_csv(file_name)
        else:
            df = pd.DataFrame(columns=["Name", "Date", "Time"])
        
        # Check if attendance already marked today
        today_attendance = df[(df['Name'] == name) & (df['Date'] == date)]
        
        if today_attendance.empty:
            # Create new entry
            new_entry = pd.DataFrame({
                "Name": [name],
                "Date": [date], 
                "Time": [time_str]
            })
            
            # Append to existing data
            df = pd.concat([df, new_entry], ignore_index=True)
            df.to_csv(file_name, index=False)
            print(f"[INFO] Attendance marked for {name} at {time_str}")
            return True
        else:
            print(f"[INFO] Attendance already marked for {name} today")
            return False
            
    except Exception as e:
        print(f"[ERROR] Failed to mark attendance: {e}")
        return False

def run_attendance():
    """Run the attendance system, return message after one attendance."""
    global camera_running, current_cap
    
    if not os.path.exists('trainer/trainer.yml'):
        return "[ERROR] Trained model not found. Please train the model first."
    
    try:
        recognizer.read('trainer/trainer.yml')
    except Exception as e:
        return f"[ERROR] Failed to load trained model: {e}"
    
    current_cap, error = try_open_webcam()
    if error:
        print(error)
        return error
    
    camera_running = True
    print("[INFO] Starting face recognition attendance system.")
    print("[INFO] Press ESC to stop the system.")
    
    attendance_marked = False
    
    while camera_running and not attendance_marked:
        ret, frame = current_cap.read()
        if not ret:
            stop_camera()
            print("[ERROR] Failed to capture image.")
            return "[ERROR] Failed to capture image."
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))
        
        for (x, y, w, h) in faces:
            face_roi = gray[y:y+h, x:x+w]
            
            try:
                id, confidence = recognizer.predict(face_roi)
                confidence_percent = round(100 - confidence)
                
                if confidence < 70:  # Lower is better for confidence
                    name = names.get(id, "Unknown")
                    color = (0, 255, 0)  # Green for recognized
                    
                    if name != "Unknown" and mark_attendance(name):
                        attendance_marked = True
                        stop_camera()
                        return f"Attendance marked successfully for {name}! Check {file_name}"
                else:
                    name = "Unknown"
                    color = (0, 0, 255)  # Red for unknown
                
                # Draw rectangle and text
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, f"{name}", (x+5, y-5), font, 0.8, color, 2)
                cv2.putText(frame, f"Confidence: {confidence_percent}%", (x+5, y+h-5), font, 0.6, (255, 255, 255), 1)
                
            except Exception as e:
                print(f"[ERROR] Recognition failed: {e}")
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                cv2.putText(frame, "Error", (x+5, y-5), font, 0.8, (0, 0, 255), 2)
        
        # Add instructions on frame
        cv2.putText(frame, "Looking for faces... Press ESC to exit", (10, 30), font, 0.7, (255, 255, 255), 2)
        cv2.imshow("Attendance System", frame)
        
        if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
            print("[INFO] Attendance system stopped by user.")
            break
    
    stop_camera()
    
    if not attendance_marked:
        return "Attendance system stopped. No attendance marked."
    
    return "Attendance system completed."

@app.route('/')
def index():
    """Serve the main UI."""
    return render_template('ui.html')

@app.route('/capture', methods=['POST'])
def capture():
    data = request.get_json()
    name = data.get('name', '').strip()
    if not name:
        return jsonify({'message': '[ERROR] Name cannot be empty.'}), 400
    
    # Clean name for filename
    clean_name = "".join(c for c in name if c.isalnum() or c in (' ', '-', '_')).strip()
    if not clean_name:
        return jsonify({'message': '[ERROR] Please enter a valid name.'}), 400
    
    message = capture_faces(clean_name)
    return jsonify({'message': message})

@app.route('/train', methods=['POST'])
def train():
    message = train_model()
    return jsonify({'message': message})

@app.route('/attendance', methods=['POST'])
def attendance():
    message = run_attendance()
    return jsonify({'message': message})

@app.route('/stop_camera', methods=['POST'])
def stop_camera_route():
    stop_camera()
    return jsonify({'message': 'Camera stopped successfully.'})

@app.route('/exit', methods=['POST'])
def exit_app():
    print("[INFO] Exiting program.")
    stop_camera()  # Stop camera before exiting
    threading.Thread(target=lambda: os._exit(0), daemon=True).start()
    return jsonify({'message': 'Exiting program.'})

if __name__ == "__main__":
    app.run(debug=True, threaded=True)