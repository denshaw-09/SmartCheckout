from ultralytics import YOLO
import os
import shutil

def train_model():
    # 1. Load the model
    model = YOLO('yolov8n.pt') 

    # 2. Train
    print("Starting Training...")
    model.train(
        data='data.yaml',
        epochs=30,
        imgsz=640,
        batch=8,
        name='smart_checkout_run'
    )
    
    # 3. Move the best model to the main folder automatically
    source_path = "runs/detect/smart_checkout_run/weights/best.pt"
    dest_path = "best.pt"

    if os.path.exists(source_path):
        shutil.copy(source_path, dest_path)
        print(f"Success! Trained model copied to '{dest_path}'")
    else:
        print("Training finished, but couldn't auto-copy 'best.pt'. Please check 'runs' folder.")

if __name__ == '__main__':
    train_model()