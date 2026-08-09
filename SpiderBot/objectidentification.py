import cv2
import os
import time
from ultralytics import YOLO
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

stop_requested = False

class _Stop(BaseHTTPRequestHandler):
    def do_POST(self):
        global stop_requested
        stop_requested = True
        self.send_response(200); self.end_headers()
    def log_message(self, *a): pass

threading.Thread(target=lambda: HTTPServer(("0.0.0.0", 8001), _Stop).serve_forever(), daemon=True).start()

SAVE_FOLDER = "stream/Photos"
os.makedirs(SAVE_FOLDER, exist_ok=True)
for filename in os.listdir(SAVE_FOLDER):
    file_path = os.path.join(SAVE_FOLDER, filename)
    if os.path.isfile(file_path):
        os.remove(file_path)

os.makedirs("stream", exist_ok=True)
open("stream/photos.txt", "w").close()

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)

if not cap.isOpened():
    print("Could not open camera")
    exit()

survivor_count = 0
seen_ids = set()
saved_histograms = []
similarity = 0.75

def get_histogram(image):
    hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    return cv2.normalize(hist, hist).flatten()

while True:
    if stop_requested:
        break
    ret, frame = cap.read()

    if not ret:
        print("Failed to read frame")
        break

    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        imgsz=320,
        verbose=False
    )

    annotated_frame = results[0].plot()

    if results[0].boxes.id is not None:

        boxes = results[0].boxes

        for box, track_id, cls in zip(
            boxes.xyxy,
            boxes.id,
            boxes.cls
        ):

            if int(cls) != 0:
                continue

            track_id = int(track_id)

            if track_id not in seen_ids:
                seen_ids.add(track_id)

                x1, y1, x2, y2 = map(int, box)

                person_crop = frame[y1:y2, x1:x2]
                if person_crop.size == 0:
                    continue

                hist = get_histogram(person_crop)

                if any(cv2.compareHist(hist, h, cv2.HISTCMP_CORREL) >= similarity for h in saved_histograms):
                    continue

                survivor_count += 1

                saved_histograms.append(hist)

                cv2.imwrite(
                    os.path.join(SAVE_FOLDER, f"person_{survivor_count}.jpg"),
                    frame
                )

                print(f"New survivor counted: {survivor_count}")

                with open("stream/photos.txt", "a") as f:
                    f.write(f"person_{survivor_count}.jpg\n")


    cv2.imwrite("stream/frame.jpg", annotated_frame)
    with open("stream/heartbeat.txt", "w") as f:
        f.write(str(time.time()))
cap.release()
cv2.destroyAllWindows()