from ultralytics import YOLO # type: ignore

def main():
    model = YOLO(r'runs\detect\ui\train\weights\best.pt')

    print("Starting Model Evaluation on TEST dataset...")
    metrics = model.val(
        data='../../../../YOLO/ui_detect/data.yaml', 
        split='test',  
        plots=True    
    )

    print("\n Testing Completed! Check the 'runs/detect/val' folder for results.")

if __name__ == '__main__':
    main()