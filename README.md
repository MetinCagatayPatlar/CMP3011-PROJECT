# CMP3011-PROJECT


  **Structure**
├── coco8.yaml
├── traffic_train.py
├── traffic_app_counter.py
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/


   **Description**
coco8.yaml
Dataset configuration file. Specifies the paths to your training/validation images and labels, and lists the class names.

traffic_train.py
Loads the YOLOv8n pre-trained weights, trains on the dataset defined in coco8.yaml, and saves the best checkpoint to:
runs/detect/train/weights/best.pt

  **Prerequisites**
Python 3.8 or higher

ultralytics (YOLOv8)

OpenCV

NumPy

  **Dataset Preparation**
               
               .. Arrange your dataset like this:
images/
  train/
  val/
labels/
  train/
  val/
               
               ..Your coco8.yaml should look like:
train: images/train
val:   images/val

names:
  0: car
  1: bus
  2: van
  3: others

**Training**

Run the training script with:
python traffic_train.py \
     --data coco8.yaml \
     --weights yolov8n.pt \
     --epochs 50 \
     --imgsz 640

 --data: path to coco8.yaml

--weights: initial weights (yolov8n.pt)

--epochs: number of training epochs

--imgsz: input image size (pixels)

After training, find your best model at:

runs/detect/train/weights/best.pt

**Inference & Counting**
1.Open traffic_app_counter.py and update the model path:
model = YOLO("runs/detect/train/weights/best.pt")
2.Run the script:
python traffic_app_counter.py \
  --source path/to/video_or_camera \
  --conf 0.25
--source: path to a video file or camera index (0 for default webcam)

--conf: confidence threshold for detections

  **Classes**
|  ID |  Class |
| :-: | :----: |
|  0  |   car  |
|  1  |   bus  |
|  2  |   van  |
|  3  | others |







