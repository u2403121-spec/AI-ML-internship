import streamlit as st
import cv2
import numpy as np
import pandas as pd
import os
from PIL import Image
from datetime import datetime

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

KNOWN_FACES_DIR = "known_faces"
ATTENDANCE_FILE = "attendance.csv"
TRAINER_FILE = "trainer.yml"

os.makedirs(KNOWN_FACES_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# LOAD FACE DETECTOR
# ─────────────────────────────────────────────

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

recognizer = cv2.face.LBPHFaceRecognizer_create()

# ─────────────────────────────────────────────
# TRAIN MODEL
# ─────────────────────────────────────────────

def train_model():

    faces = []
    labels = []
    label_map = {}

    current_label = 0

    for filename in os.listdir(KNOWN_FACES_DIR):

        if filename.endswith((".jpg", ".png", ".jpeg")):

            path = os.path.join(KNOWN_FACES_DIR, filename)

            name = os.path.splitext(filename)[0]

            img = Image.open(path).convert("L")

            img_np = np.array(img, "uint8")

            detected_faces = face_cascade.detectMultiScale(
                img_np,
                scaleFactor=1.1,
                minNeighbors=5
            )

            for (x, y, w, h) in detected_faces:

                faces.append(img_np[y:y+h, x:x+w])

                labels.append(current_label)

            label_map[current_label] = name

            current_label += 1

    if len(faces) > 0:

        recognizer.train(faces, np.array(labels))

        recognizer.save(TRAINER_FILE)

    return label_map

# ─────────────────────────────────────────────
# MARK ATTENDANCE
# ─────────────────────────────────────────────

def mark_attendance(name):

    now = datetime.now()

    date = now.strftime("%Y-%m-%d")

    time = now.strftime("%H:%M:%S")

    if os.path.exists(ATTENDANCE_FILE):

        df = pd.read_csv(ATTENDANCE_FILE)

        already_marked = (
            (df["Name"] == name) &
            (df["Date"] == date)
        ).any()

        if already_marked:
            return

    new_entry = pd.DataFrame(
        [[name, time, date]],
        columns=["Name", "Login Time", "Date"]
    )

    if os.path.exists(ATTENDANCE_FILE):

        new_entry.to_csv(
            ATTENDANCE_FILE,
            mode="a",
            header=False,
            index=False
        )

    else:

        new_entry.to_csv(
            ATTENDANCE_FILE,
            index=False
        )

# ─────────────────────────────────────────────
# STREAMLIT UI
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="Face Detection Attendance System",
    layout="wide"
)

st.title("🎓 Face Detection Attendance System")

st.write(
    "Current Time:",
    datetime.now().strftime("%Y-%m-%d %H:%M:%S")
)

menu = st.sidebar.radio(
    "Navigation",
    [
        "📋 Register Face",
        "📷 Mark Attendance",
        "📊 View Report"
    ]
)

# ─────────────────────────────────────────────
# REGISTER FACE
# ─────────────────────────────────────────────

if menu == "📋 Register Face":

    st.header("Register New Student")

    name = st.text_input("Enter Student Name")

    img_file = st.camera_input(
        "Capture Student Image"
    )

    if img_file and name:

        if st.button("Register Student"):

            img = Image.open(img_file)

            save_path = os.path.join(
                KNOWN_FACES_DIR,
                f"{name}.jpg"
            )

            img.save(save_path)

            train_model()

            st.success(
                f"✅ {name} registered successfully!"
            )

# ─────────────────────────────────────────────
# MARK ATTENDANCE
# ─────────────────────────────────────────────

elif menu == "📷 Mark Attendance":

    st.header("Live Attendance")

    if not os.path.exists(TRAINER_FILE):

        st.warning(
            "⚠️ Please register students first."
        )

    else:

        label_map = train_model()

        recognizer.read(TRAINER_FILE)

        run = st.checkbox("Start Camera")

        FRAME_WINDOW = st.image([])

        camera = cv2.VideoCapture(0)

        while run:

            ret, frame = camera.read()

            if not ret:
                st.error("❌ Camera not accessible.")
                break

            frame = cv2.flip(frame, 1)

            gray = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2GRAY
            )

            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.2,
                minNeighbors=5,
                minSize=(100, 100)
            )

            for (x, y, w, h) in faces:

                face_roi = gray[y:y+h, x:x+w]

                label, confidence = recognizer.predict(
                    face_roi
                )

                if confidence < 70:

                    name = label_map[label]

                    color = (0, 255, 0)

                    mark_attendance(name)

                else:

                    name = "Unknown"

                    color = (0, 0, 255)

                # FACE RECTANGLE
                cv2.rectangle(
                    frame,
                    (x, y),
                    (x+w, y+h),
                    color,
                    3
                )

                # NAME BACKGROUND
                cv2.rectangle(
                    frame,
                    (x, y - 40),
                    (x+w, y),
                    color,
                    -1
                )

                # NAME TEXT
                cv2.putText(
                    frame,
                    name,
                    (x + 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2
                )

            FRAME_WINDOW.image(
                cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB
                ),
                channels="RGB"
            )

        camera.release()

# ─────────────────────────────────────────────
# VIEW REPORT
# ─────────────────────────────────────────────

elif menu == "📊 View Report":

    st.header("Attendance Report")

    if os.path.exists(ATTENDANCE_FILE):

        df = pd.read_csv(ATTENDANCE_FILE)

        st.dataframe(
            df,
            use_container_width=True
        )

        csv = df.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "⬇ Download CSV",
            csv,
            "attendance.csv",
            "text/csv"
        )

    else:

        st.info("No attendance records found.")