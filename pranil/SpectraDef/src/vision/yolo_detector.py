import av
import cv2
from ultralytics import YOLO


# ==============================
# 1. Load YOLO model
# ==============================

model = YOLO("yolov8n.pt")


# ==============================
# 2. RTSP camera URL
# ==============================

RTSP_URL = "rtsp://10.153.142.207:554/stream"


# ==============================
# 3. Open RTSP stream
# ==============================

print("Connecting to RTSP stream...")

container = av.open(
    RTSP_URL,
    options={
        "rtsp_transport": "tcp",
        "timeout": "10000000"
    }
)

print("RTSP CONNECTED")


# ==============================
# 4. Read live frames
# ==============================

for frame in container.decode(video=0):

    # Convert PyAV frame → OpenCV image
    image = frame.to_ndarray(format="bgr24")

    # ==============================
    # 5. Run YOLO
    # ==============================

    results = model(image, verbose=False)

    # ==============================
    # 6. Print detections
    # ==============================

    for result in results:

        for box in result.boxes:

            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            class_name = model.names[class_id]

            print(
                f"Detected: {class_name} "
                f"| Confidence: {confidence:.2f}"
            )

    # ==============================
    # 7. Draw bounding boxes
    # ==============================

    annotated = results[0].plot()

    # ==============================
    # 8. Display live result
    # ==============================

    cv2.imshow(
        "SpectraDef - Live YOLO Detection",
        annotated
    )

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ==============================
# 9. Cleanup
# ==============================

container.close()
cv2.destroyAllWindows()

print("Stream stopped.")