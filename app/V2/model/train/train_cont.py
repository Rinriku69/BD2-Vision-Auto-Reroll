from ultralytics import YOLO # type: ignore

def main():
    model = YOLO(r'runs\detect\train\weights\last.pt') 
    print("Resuming YOLOv8 Training...")
    results = model.train(resume=True)

if __name__ == '__main__':
    main()