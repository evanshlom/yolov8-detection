from ultralytics import YOLO

model = YOLO("yolov8m.pt")

# Get annotated jpg with bounding boxes of detected objects, and labels txt file
model.predict(source="img1.jpg",
              
              # compute
              device=0, # default cuda:0
              conf=0.5, # confidence threshold for object detection
              
              # output
              save=True, # save results in jpg format to 'runs/detect/predict'
              save_txt=True, # save labels in txt format to 'runs/detect/predict/labels'
              save_conf=True, # save confidences as float to 'runs/detect/predict/conf'
              
              # format
              line_width=1) # bounding box width (pixels)