# Train dataset

from ultralytics import YOLO

model = YOLO("yolov8n.pt")
model.train(data="datasets/facemasks/data.yaml", workers=0, epochs=100, batch=16)