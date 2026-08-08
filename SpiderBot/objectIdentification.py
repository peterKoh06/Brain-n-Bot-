import cv2
import os
from ultralytics import YOLO

SAVE_FOLDER = "Photos"
os.makedirs(SAVE_FOLDER, exist_ok=True)

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
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
    ret, frame = cap.read()

    if not ret:
        print("Failed to read frame")
        break

    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
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
 
                # Skip if this person matches someone already saved
                if any(cv2.compareHist(hist, h, cv2.HISTCMP_CORREL) >= similarity for h in saved_histograms):
                    continue

                survivor_count += 1

                saved_histograms.append(hist)
 
                cv2.imwrite(
                    os.path.join(SAVE_FOLDER, f"person_{survivor_count}.jpg"),
                    frame
                    # person_crop
                )

                print(f"New survivor counted: {survivor_count}")



    cv2.imshow("YOLO Survivor Counter", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("e"):
        break

cap.release()
cv2.destroyAllWindows()
