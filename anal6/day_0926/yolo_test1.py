# pip install ultralytics, opencv-python

from ultralytics import YOLO

try:
    model = YOLO('yolov8n.pt')


except Exception as e:
    print('error')

# print(ultralytics.checks)
print(model.names)      # COCO dataset(class 80개)
print(len(model.names))