import cv2
import os
import numpy as np
import pickle
import csv
from datetime import datetime

# ===================== OBJECT DETECTION =====================
class VideoCamera:
    def __init__(self):
        self.thres = 0.45
        self.cap = cv2.VideoCapture(0)

        with open("coco.names", "rt") as f:
            self.classNames = f.read().rstrip("\n").split("\n")

        self.net = cv2.dnn_DetectionModel(
            "frozen_inference_graph.pb",
            "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
        )
        self.net.setInputSize(320, 320)
        self.net.setInputScale(1.0 / 127.5)
        self.net.setInputMean((127.5, 127.5, 127.5))
        self.net.setInputSwapRB(True)

    def __del__(self):
        if self.cap.isOpened():
            self.cap.release()

    def get_frame_and_objects(self):
        success, img = self.cap.read()
        detected_objects = []

        if not success or img is None:
            return None, []

        try:
            classIds, confs, bbox = self.net.detect(img, confThreshold=self.thres)
        except:
            return None, []

        if classIds is not None:
            for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
                name = self.classNames[classId - 1].upper()

                # ✅ ONLY PERSON & COW
                if name in ["PERSON", "COW"]:
                    if name not in detected_objects:
                        detected_objects.append(name)

                    # draw ONLY for these
                    cv2.rectangle(img, box, (0, 0, 255), 2)
                    cv2.putText(img, name, (box[0], box[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        ret, jpeg = cv2.imencode('.jpg', img)
        return jpeg.tobytes(), detected_objects


# ===================== FACE RECOGNITION =====================
class FaceRecognitionSystem:
    def __init__(self):
        self.data_path = "opencv"
        self.face_size = (200, 200)
        self.model_path = "face_model.yml"
        self.label_path = "labels.pkl"
        self.attendance_path = "attendance.csv"

        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.label_map = {}

    # ================= CAPTURE =================
    def capture_faces(self, person_name, source=0, max_images=10):
        save_path = os.path.join(self.data_path, person_name)
        os.makedirs(save_path, exist_ok=True)

        cap = cv2.VideoCapture(source)
        count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                count += 1
                face = gray[y:y+h, x:x+w]
                face = cv2.resize(face, self.face_size)
                cv2.imwrite(f"{save_path}/{count}.jpg", face)

            if count >= max_images:
                break

        cap.release()
        print(f"✅ Captured {count} images for {person_name}")

    # ================= TRAIN =================
    def train_model(self):
        faces, labels = [], []
        label_id = 0
        self.label_map = {}

        for person in os.listdir(self.data_path):
            path = os.path.join(self.data_path, person)
            if not os.path.isdir(path):
                continue

            for img_name in os.listdir(path):
                img = cv2.imread(os.path.join(path, img_name), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue

                faces.append(cv2.resize(img, self.face_size))
                labels.append(label_id)

            self.label_map[label_id] = person
            label_id += 1

        if not faces:
            return

        self.recognizer.train(faces, np.array(labels))
        self.recognizer.save(self.model_path)

        with open(self.label_path, "wb") as f:
            pickle.dump(self.label_map, f)

        print("✅ Model trained")

    # ================= LOAD =================
    def load_model(self):
        if os.path.exists(self.model_path):
            self.recognizer.read(self.model_path)

        if os.path.exists(self.label_path):
            with open(self.label_path, "rb") as f:
                self.label_map = pickle.load(f)

    # ================= ATTENDANCE (UPDATED LOGIC) =================
    def mark_attendance(self, name):
        now = datetime.now()
        date_str = now.strftime('%Y-%m-%d')
        time_str = now.strftime('%H:%M:%S')

        rows = []
        updated = False

        # ✅ read existing data
        if os.path.exists(self.attendance_path):
            with open(self.attendance_path, 'r') as f:
                reader = csv.reader(f)
                rows = list(reader)

        # ✅ update if same name + same date
        for i, row in enumerate(rows):
            if len(row) == 3:
                if row[0] == name and row[2] == date_str:
                    rows[i][1] = time_str   # 🔥 update time
                    updated = True

        # ✅ if not found → add new
        if not updated:
            rows.append([name, time_str, date_str])

        # ✅ rewrite full file
        with open(self.attendance_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(rows)

        print(f"[ATTENDANCE UPDATED] {name} → {time_str}")