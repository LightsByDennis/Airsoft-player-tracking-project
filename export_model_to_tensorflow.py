from ultralytics import YOLO

# Load the YOLO11 model
model = YOLO("yolo11n.pt")

# Export the model to TFLite Edge TPU format
model.export(format="edgetpu")  # creates 'yolo11n_full_integer_quant_edgetpu.tflite'

print("Done")
