from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')

# Export the model to ONNX format
model.export(format='edgetpu')  # creates 'yolov8n.onnx'
