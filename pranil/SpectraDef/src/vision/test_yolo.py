from ultralytics import YOLO
import cv2
import av
import threading
import time


# ==============================
# 1. Load YOLO11m model
# ==============================

print("Loading YOLO11m model...")

model = YOLO("yolo11m.pt")

print("YOLO11m loaded!")


# ==============================
# 2. RTSP camera stream
# ==============================

RTSP_URL = "rtsp://10.153.142.207:554/stream"

print("Connecting to RTSP stream...")


container = av.open(
    RTSP_URL,
    options={
        "rtsp_transport": "tcp",
        "timeout": "10000000"
    }
)


print("RTSP connected!")
print("Starting YOLO detection...")
print("Press Q to quit.")


# ==============================
# 3. Latest-frame buffer
# ==============================

latest_frame = None

frame_lock = threading.Lock()

stop_thread = False


def read_frames():
    global latest_frame
    global stop_thread

    try:

        for frame in container.decode(video=0):

            if stop_thread:
                break

            image = frame.to_ndarray(
                format="bgr24"
            )

            # Only keep the newest frame.
            # Old frames are automatically discarded.
            with frame_lock:
                latest_frame = image

    except Exception as e:

        print("Stream reader stopped:", e)


# ==============================
# 4. Start camera reader
# ==============================

reader_thread = threading.Thread(
    target=read_frames,
    daemon=True
)

reader_thread.start()


# ==============================
# 5. YOLO processing loop
# ==============================

last_time = time.time()

while True:

    # ------------------------------
    # Get newest frame
    # ------------------------------

    with frame_lock:

        if latest_frame is None:
            continue

        image = latest_frame.copy()


    # ------------------------------
    # Run YOLO11m
    # ------------------------------

    results = model(
        image,
        imgsz=320,
        conf=0.40,
        verbose=False
    )


    # ------------------------------
    # Draw detections
    # ------------------------------

    annotated = results[0].plot()


    # ------------------------------
    # Calculate FPS
    # ------------------------------

    current_time = time.time()

    fps = 1 / (current_time - last_time)

    last_time = current_time


    # Display FPS on screen

    cv2.putText(
        annotated,
        f"FPS: {fps:.1f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )


    # ------------------------------
    # Display
    # ------------------------------

    cv2.imshow(
        "SpectraDef - YOLO11m Live Detection",
        annotated
    )


    # ------------------------------
    # Quit
    # ------------------------------

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ==============================
# 6. Cleanup
# ==============================

stop_thread = True

container.close()

cv2.destroyAllWindows()

print("Stream stopped.")