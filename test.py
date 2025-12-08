from ultralytics import YOLO
import cv2

model = YOLO('best.pt') # Ensure best.pt is in the folder
results = model.predict('test.jpg', show=True, save=True)
cv2.waitKey(0)