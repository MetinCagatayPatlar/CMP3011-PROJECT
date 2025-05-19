from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.pt") 

# Train the model
results = model.train(data="coco8.yaml", device="0", epochs=100, imgsz=640)