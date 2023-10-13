# Train dataset

from ultralytics import YOLO

detaset_name = "facemasks" # Dataset name

model = YOLO("yolov8n.pt")
model.train(data = "datasets/" + detaset_name + "/data.yaml", workers = 0, epochs = 50, batch = 24)