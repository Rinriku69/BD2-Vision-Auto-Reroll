import cv2 #type: ignore
import numpy as np
import pyautogui #type: ignore
from ultralytics import YOLO #type: ignore


def main():
    model = YOLO(r'runs\detect\train\weights\best.pt')

    print("Bot Started!")

    screenshot = pyautogui.screenshot()
    frame = np.array(screenshot)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)


    results = model(frame, conf=0.8)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls_id = int(box.cls[0])
            
            if cls_id == 0: 
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                print(f"Found 5 star at: {x1}, {y1}, {x2}, {y2}")

                
                cropped_5star = frame[y1:y2, x1:x2]

                cv2.imshow("Found 5 Star!", cropped_5star)
                cv2.waitKey(0)

if __name__ == "__main__":
    main()