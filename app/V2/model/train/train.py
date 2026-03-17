from ultralytics import YOLO # type: ignore

def main(): 
    model = YOLO('yolov8n.pt') 
    print(" Starting YOLOv8 Training on GPU...")
    results = model.train(
        data='../../../../YOLO/ui_detect/data.yaml',   
        epochs=100,         
        imgsz=640,         
        batch=4,            
        device=0,          
        plots=True          
    )

if __name__ == '__main__':
    main()