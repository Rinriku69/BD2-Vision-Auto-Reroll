# Import libraries and model
import pyautogui 
import time 
import pydirectinput # type: ignore
import keyboard
import winsound 
import cv2 
from cv2.typing import MatLike
import numpy as np 
from ultralytics import YOLO 
from typing import cast, TypedDict
from ultralytics.engine.results import Results



class CenterPosition(TypedDict):
    x:int
    y:int


class CharacterDetection():
    
    #--- Constructor -----
    def __init__(self, char_model_path:str, ui_model_path:str, characters:list[str], toggle_key:str, stop_key:str):
        print(f"Loading Model...")
        # Script State
        self.script_alive: bool = True
        self.script_active: bool = False
        # Load Model
        try:
            self.char_model: YOLO = YOLO(char_model_path)
            self.ui_model: YOLO = YOLO(ui_model_path)
            print(f"Model loaded successfully!")
        except Exception as e:
            print(f"Error Loading Model {e}")
        
        # Register hotkeys
        self.register_hotkeys(toggle_key,stop_key)
        
        # Get target characters images
        base_path = 'public/characters/'
        character_image_paths = [f"{base_path}{char_file_name}.png" for char_file_name in characters]
        print('\n'.join([f"Getting Character path : {path}," for path in character_image_paths]))
        
        # Convert Images to BGR2GRAY
        self.chars_need = len(character_image_paths)
        
        self.template_gray: list[MatLike] = []
        for char_path in character_image_paths:
            img = cv2.imread(char_path)
    
            if img is None:
                raise ValueError(f"Failed to get the image: {char_path}")
            print(f"Successfully Loaded {char_path}")
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            self.template_gray.append(gray)
    #--------------------------------------------------------------
    
    
    # Script State Control    
    def exit_script(self) -> None :
        self.script_active = False
        self.script_alive = False

    def toggle_script(self)  -> None :
        self.script_active = not self.script_active
        
        if self.script_active:
            print("--- Script ACTIVATED by Hotkey ---")
        else:
            print("--- Script DEACTIVATED by Hotkey ---")
            
    def register_hotkeys(self,toggle_key:str,stop_key:str)  -> None :
        try:
            keyboard.add_hotkey(toggle_key, self.toggle_script)
            keyboard.add_hotkey(stop_key, self.exit_script)

            print(
                f"Script loaded. Press {toggle_key.upper()} to Start/Stop. "
                f"Press {stop_key.upper()} to Exit."
            )

        except Exception as e:
            print(
                f"Failed to register hotkeys. Try running the script as an administrator. Error: {e}"
            )
            exit()
    #--------------------
    
       
    #  Get object screenshot 
    def char_detecting(self) :
        screenshot = pyautogui.screenshot()
        frame = np.array(screenshot)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        results: list[Results]  = cast(list[Results],self.char_model(frame, conf=0.8,verbose=False))
        found_character: list[np.ndarray] = []
        for r in results:
            slots = r.boxes
            if slots == None:
                return None    
            for slot in slots:
                cls_id = int(slot.cls[0])
                if cls_id == 0: 
                    x1, y1, x2, y2 = map(int, slot.xyxy[0])
                    cropped_character: np.ndarray = frame[y1:y2, x1:x2]
                    found_character.append(cropped_character)
        print(f"Finished Scan found {len(found_character)} 5-star characters")
        return found_character
    
    def ui_detecting(self,target_btn:str) -> None:
        screenshot = pyautogui.screenshot()
        frame = np.array(screenshot)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        results:list[Results]  = cast(list[Results],self.ui_model(frame, conf=0.8,verbose=False))
        center: CenterPosition = {"x":0,"y":0}
        for r in results:
            buttons = r.boxes
            if buttons == None:
                print("Buttons not found")
                break
            
            # Classes : draw_again = 0, skip = 1
            for btn in buttons:
                cls_btn = int(btn.cls[0])
                if cls_btn == 0 and target_btn == "redraw":
                    x1, y1, x2, y2 = map(int,btn.xyxy)
                    center["x"], center["y"] =  (x1 + x2)//2,(y1+y2)//2
                    print(f"Redraw button found at {center["x"], center["y"]}")
                    pyautogui.click(center["x"],center["y"])         
                elif cls_btn == 1 and target_btn == "skip":
                    x1, y1, x2, y2 = map(int,btn.xyxy)
                    center["x"], center["y"] =  (x1 + x2)//2,(y1+y2)//2
                    print(f"Skip button found at {center["x"], center["y"]}")
                    pyautogui.click(center["x"],center["y"])        

    
    def verify_target_match(self,cropped_5star_img: np.ndarray, threshold:float = 0.6) -> bool:
        cropped_gray = cv2.cvtColor(cropped_5star_img, cv2.COLOR_BGR2GRAY)

        for template_gray in self.template_gray:
            h_temp, w_temp = template_gray.shape
            h_crop, w_crop = cropped_gray.shape
                
            if h_temp > h_crop or w_temp > w_crop:
                print("Template picture is too big ")
                return False
            result = cv2.matchTemplate(cropped_gray, template_gray, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(result)

            if max_val >= threshold:
                print(f"Found With : {max_val*100:.2f}%")
                return True
            else:
                print(f"Not Found : {max_val*100:.2f}%")
        return False      
        
    def start_reroll(self):  
        try:  
            while self.script_alive:
                if self.script_active: 
                    condition_met: bool = False
                    # UI DETECT    
                    self.ui_detecting("redraw")
                    time.sleep(DELAY_AFTER_CLICK)
                    pydirectinput.press('enter')            # type: ignore
                    time.sleep(DELAY_AFTER_CLICK)
                    for _ in range(4): 
                        self.ui_detecting("skip")
                        time.sleep(0.2)
                    time.sleep(DELAY_AFTER_PULL_SEQUENCE)
                    #---------
                    
                    character_images:list[np.ndarray] | None = self.char_detecting()
                    if character_images == None:
                        time.sleep(0.5)
                        continue
                    
                    target_found: int = 0
                    for c in character_images:
                        target_char_check = self.verify_target_match(c)
                        if target_char_check:
                            target_found += 1
                            print(f"Found {target_found} target now")
                            if target_found == self.chars_need:
                                condition_met = True
                                break
        
                    # Stop Condition
                    if condition_met:
                        print("*** Target Founded ****")
                        self.script_active = False
                        print("--- Script Stopped ---")
                        sound_file = "public/sound/tuturu.wav"
                        try:
                            winsound.PlaySound(sound_file, winsound.SND_FILENAME)
                        except Exception as e_sound:
                            print(f"Error playing sound: {e_sound},{sound_file}")
                            winsound.Beep(1000, 500)
                    else:
                        time.sleep(DELAY_BEFORE_RETRY)
                time.sleep(0.1)
        except KeyboardInterrupt: 
            print("\n--- Script interrupted by user (Ctrl+C). Exiting. ---")

        finally:
            print("--- Cleaning up hotkeys... ---")
            keyboard.unhook_all_hotkeys() 
            print("--- Script fully exited. ---")
                
                
if __name__ == "__main__":
    CHAR_MODEL_PATH = 'model/train/runs/detect/char/train/weights/best.pt'
    UI_MODEL_PATH = 'model/train/runs/detect/ui/train2/weights/best.pt'
    DELAY_AFTER_CLICK = 0.5
    DELAY_AFTER_PULL_SEQUENCE = 0.5 
    DELAY_BEFORE_RETRY = 0.2
    
    bot = CharacterDetection(CHAR_MODEL_PATH,UI_MODEL_PATH,["blade1","alec"],'f9','esc')
    bot.start_reroll()