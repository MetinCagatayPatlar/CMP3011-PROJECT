import cv2
import time
from ultralytics import YOLO
from collections import Counter


def main():
    cap = cv2.VideoCapture("traffic.mp4")
    model = YOLO("yolov8n.pt")
    model.to("cuda:0")
    print(f"Using device: {model.device}")

    # COCO class name mapping
    target_classes = {
        2: "car",
        5: "bus",
        3: "motorcycle",
        7: "truck"
    }

    # Setup font
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    font_color = (0, 255, 0)
    thickness = 2

    frame_count = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (1280, 720))

        # Run inference
        # results = model.predict(frame, device="cuda", verbose=False)[0]
        results = model.track(frame, device="cuda", verbose=False)[0]

        # Count target classes
        detections = results.boxes
        class_ids = detections.cls.int().tolist() if detections is not None else []
        counts = Counter()

        for cls_id in class_ids:
            if cls_id in target_classes:
                counts[target_classes[cls_id]] += 1

        # Draw detections on frame
        annotated = results.plot()

        # Draw counters
        y_offset = 30
        for cls in ["car", "bus", "motorcycle", "truck"]:
            label = f"{cls}: {counts[cls]}"
            cv2.putText(annotated, label, (10, y_offset), font, font_scale, font_color, thickness)
            y_offset += 30

        # Draw FPS
        frame_count += 1
        elapsed = time.time() - start_time
        fps = frame_count / elapsed if elapsed > 0 else 0
        cv2.putText(annotated, f"FPS: {fps:.2f}", (10, y_offset), font, font_scale, (255, 255, 0), thickness)

        # Show frame
        cv2.imshow("YOLOv8 Vehicle Counter", annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()